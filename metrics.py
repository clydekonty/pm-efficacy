"""Statistical Evaluation Module for Forecast Efficiency, Scoring Rules, and Calibration.

Implements:
- Dynamic cross-sectional Brier Score over countdown time t in [-365, -1].
- Cross-sectional Log Loss (Cross-Entropy) with probability clipping epsilon = 0.005.
- Calibration binning, Empirical Win Rates, 95% Wilson Score Confidence Intervals.
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
- Liquidity barrier filtering and multi-office comparative evaluation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_LIQUIDITY_THRESHOLD,
    LOG_LOSS_EPSILON,
    WILSON_CONFIDENCE_LEVEL,
)

logger = logging.getLogger("metrics")


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = WILSON_CONFIDENCE_LEVEL
) -> Tuple[float, float]:
    """Calculate the Wilson Score confidence interval for a binomial proportion.

    Handles edge cases (0 trials, 0 or 100% successes) gracefully without degenerate intervals.
    """
    if trials <= 0:
        return (np.nan, np.nan)

    p_hat = successes / trials
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z_sq = z * z

    denominator = 1.0 + z_sq / trials
    center = (p_hat + z_sq / (2.0 * trials)) / denominator
    half_width = (z / denominator) * np.sqrt(
        (p_hat * (1.0 - p_hat) / trials) + (z_sq / (4.0 * trials * trials))
    )

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (float(lower), float(upper))


def calculate_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the Brier Score: BS = (1/N) * sum((p_i - y_i)^2)."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def calculate_log_loss(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    epsilon: float = LOG_LOSS_EPSILON
) -> float:
    """Compute Binary Cross-Entropy (Log Loss), clipping probabilities to [eps, 1 - eps]."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return np.nan

    p_clipped = np.clip(p, epsilon, 1.0 - epsilon)
    loss = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
    return float(loss)


def compute_dynamic_scoring(
    panel_df: pd.DataFrame,
    office: Optional[str] = None,
    liquidity_tier: Optional[str] = None,
    min_trailing_7d_vol: Optional[float] = None,
) -> pd.DataFrame:
    """Compute dynamic cross-sectional Brier Score and Log Loss grouped by days_to_election.

    Parameters:
      panel_df: Master panel with ['race_id', 'days_to_election', 'implied_prob', 'actual_winner', ...]
      office: Optional filter for 'Senate', 'House', or 'Governor'.
      liquidity_tier: 'top_quartile', 'above_median', or None ('full_sample').
      min_trailing_7d_vol: Absolute volume threshold barrier (e.g. $25,000).

    Returns:
      DataFrame indexed by days_to_election with ['days_to_election', 'n_races', 'brier_score', 'log_loss', 'mean_prob', 'mean_volume']
    """
    if panel_df.empty:
        return pd.DataFrame()

    df = panel_df.copy()

    # Filter by office type
    if office and office.lower() != "all":
        df = df[df["office"].str.lower() == office.lower()]

    # Filter by absolute liquidity barrier if specified
    if min_trailing_7d_vol is not None and "trailing_7d_vol" in df.columns:
        df = df[df["trailing_7d_vol"] >= min_trailing_7d_vol]

    # Filter by relative liquidity tier if specified
    if liquidity_tier and "volume" in df.columns and len(df) > 0:
        if liquidity_tier == "top_quartile":
            q75 = df["volume"].quantile(0.75)
            df = df[df["volume"] >= q75]
        elif liquidity_tier == "above_median":
            q50 = df["volume"].median()
            df = df[df["volume"] >= q50]

    if df.empty:
        return pd.DataFrame()

    metrics_list = []
    for dte, group in df.groupby("days_to_election"):
        n_obs = len(group)
        if n_obs == 0:
            continue

        p = group["implied_prob"].to_numpy()
        y = group["actual_winner"].to_numpy()

        bs = calculate_brier_score(p, y)
        ll = calculate_log_loss(p, y)

        vol_mean = group["volume"].mean() if "volume" in group.columns else 0.0
        vol_median = group["volume"].median() if "volume" in group.columns else 0.0

        metrics_list.append({
            "days_to_election": int(dte),
            "n_races": n_obs,
            "brier_score": bs,
            "log_loss": ll,
            "mean_prob": float(np.mean(p)),
            "win_rate": float(np.mean(y)),
            "mean_volume": float(vol_mean),
            "median_volume": float(vol_median),
        })

    res_df = pd.DataFrame(metrics_list)
    if not res_df.empty:
        res_df = res_df.sort_values(by="days_to_election").reset_index(drop=True)
    return res_df


def compute_calibration_curve(
    panel_df: pd.DataFrame,
    horizon_days: int = 30,
    n_bins: int = 5,
    min_trailing_7d_vol: Optional[float] = None,
    office: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate calibration table and Expected Calibration Error (ECE) at a fixed horizon T-H.

    Subsets to favorite-side probabilities p in [0.50, 0.99] at days_to_election == -horizon_days.
    Bins probabilities into adaptive or equal-width bins, computing:
      - N_bin: sample size
      - p_bar: mean predicted probability
      - y_hat: empirical win rate
      - lower_ci_95, upper_ci_95: 95% Wilson Score confidence interval bounds
      - ECE: Expected Calibration Error
      - MCE: Maximum Calibration Error
    """
    if panel_df.empty:
        return {"bins_df": pd.DataFrame(), "ece": np.nan, "mce": np.nan, "n_obs": 0}

    df = panel_df.copy()

    # Filter by office
    if office and office.lower() != "all":
        df = df[df["office"].str.lower() == office.lower()]

    # Filter by liquidity threshold
    if min_trailing_7d_vol is not None and "trailing_7d_vol" in df.columns:
        df = df[df["trailing_7d_vol"] >= min_trailing_7d_vol]

    # Target specific countdown day: t = -horizon_days
    target_dte = -abs(horizon_days)
    horizon_data = df[df["days_to_election"] == target_dte].copy()

    # If exact day missing for some contracts, allow closest day within +/- 2 days
    if len(horizon_data) == 0:
        window_df = df[(df["days_to_election"] >= target_dte - 2) & (df["days_to_election"] <= target_dte + 2)].copy()
        if not window_df.empty:
            # Pick the observation closest to target_dte for each race
            window_df["dte_diff"] = (window_df["days_to_election"] - target_dte).abs()
            horizon_data = window_df.sort_values("dte_diff").groupby("race_id").first().reset_index()

    if horizon_data.empty:
        logger.warning("No data found for horizon T-%d", horizon_days)
        return {"bins_df": pd.DataFrame(), "ece": np.nan, "mce": np.nan, "n_obs": 0}

    # Restrict to favorite probabilities [0.50, 1.00]
    p_vals = horizon_data["implied_prob"].to_numpy()
    y_vals = horizon_data["actual_winner"].to_numpy()
    vols = horizon_data["volume"].to_numpy() if "volume" in horizon_data.columns else np.zeros(len(p_vals))

    # Define bins across [0.50, 1.00]
    bin_edges = np.linspace(0.50, 1.00, n_bins + 1)
    bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(n_bins)]

    bin_assignments = np.digitize(p_vals, bin_edges, right=False) - 1
    # Handle upper boundary edge case (p == 1.0 falls into last bin)
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    bin_records = []
    total_n = len(p_vals)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = (bin_assignments == i)
        n_k = int(np.sum(mask))

        if n_k > 0:
            p_bin = p_vals[mask]
            y_bin = y_vals[mask]
            v_bin = vols[mask]

            p_bar = float(np.mean(p_bin))
            wins = int(np.sum(y_bin))
            y_hat = float(wins / n_k)
            vol_total = float(np.sum(v_bin))
            vol_mean = float(np.mean(v_bin))

            lower_ci, upper_ci = wilson_score_interval(wins, n_k, confidence=WILSON_CONFIDENCE_LEVEL)
            cal_diff = abs(y_hat - p_bar)

            ece += (n_k / total_n) * cal_diff
            mce = max(mce, cal_diff)
        else:
            p_bar = float((bin_edges[i] + bin_edges[i+1]) / 2.0)
            y_hat = np.nan
            lower_ci, upper_ci = np.nan, np.nan
            vol_total = 0.0
            vol_mean = 0.0
            cal_diff = np.nan

        bin_records.append({
            "bin_idx": i,
            "bin_range": bin_labels[i],
            "bin_lower": float(bin_edges[i]),
            "bin_upper": float(bin_edges[i+1]),
            "n_obs": n_k,
            "mean_pred_prob": p_bar,
            "empirical_win_rate": y_hat,
            "ci_lower_95": lower_ci,
            "ci_upper_95": upper_ci,
            "total_volume": vol_total,
            "mean_volume": vol_mean,
            "abs_calibration_gap": cal_diff,
        })

    bins_df = pd.DataFrame(bin_records)
    return {
        "bins_df": bins_df,
        "ece": float(ece),
        "mce": float(mce),
        "n_obs": total_n,
        "horizon_days": horizon_days,
        "brier_score": calculate_brier_score(p_vals, y_vals),
        "log_loss": calculate_log_loss(p_vals, y_vals),
    }


def evaluate_liquidity_barrier_sensitivity(
    panel_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Analyze how Brier score, Log Loss, and ECE vary across different liquidity barriers.

    Tests filtering contracts by trailing 7-day cumulative volume thresholds.
    """
    if panel_df.empty:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = [0.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0]

    sensitivity_records = []
    for thresh in thresholds:
        cal_res = compute_calibration_curve(
            panel_df,
            horizon_days=horizon_days,
            min_trailing_7d_vol=thresh if thresh > 0 else None,
        )
        n_obs = cal_res["n_obs"]
        if n_obs > 0:
            sensitivity_records.append({
                "threshold_vol": float(thresh),
                "horizon_days": horizon_days,
                "n_races": n_obs,
                "brier_score": cal_res["brier_score"],
                "log_loss": cal_res["log_loss"],
                "ece": cal_res["ece"],
                "mce": cal_res["mce"],
            })

    return pd.DataFrame(sensitivity_records)
