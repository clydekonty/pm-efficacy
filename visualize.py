"""Visualization Module for Kalshi Prediction Market Forecast Efficacy.

Generates publication-grade figures:
1. Dynamic Brier Score & Log Loss trajectories over countdown days t in [-365, -1].
2. Multi-horizon Calibration Reliability Dot Plots with 95% Wilson Score error bars
   and volume-proportional point sizing.
3. Liquidity barrier sensitivity curves.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

# Set non-interactive backend and writable cache dir before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR
from metrics import compute_calibration_curve, compute_dynamic_scoring, evaluate_liquidity_barrier_sensitivity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visualize")

# Apply clean professional plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
})


def plot_scoring_curves_over_time(
    panel_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "brier_logloss_curves.png"
) -> Path:
    """Plot cross-sectional Brier Score and Log Loss curves over countdown days t in [-365, -1].

    Includes office breakdowns (Senate, House, Governor) and liquidity tier comparison.
    """
    if panel_df.empty:
        logger.warning("Empty panel DataFrame. Skipping scoring curves plot.")
        return output_path

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig.suptitle("Kalshi Election Forecast Efficiency Leading Up to Election Day (T-365 to T-1)", fontsize=16, weight="bold")

    # 1. Full Sample Dynamic Scoring
    full_scores = compute_dynamic_scoring(panel_df)

    # 2. Office Breakdowns
    offices = ["Senate", "House", "Governor"]
    office_colors = {"Senate": "#1f77b4", "House": "#2ca02c", "Governor": "#d62728"}

    # Top-Left: Brier Score by Office
    ax_bs_off = axes[0, 0]
    if not full_scores.empty:
        ax_bs_off.plot(full_scores["days_to_election"], full_scores["brier_score"],
                       label="All Races (Aggregate)", color="black", linewidth=2.5, linestyle="-")

    for off in offices:
        off_scores = compute_dynamic_scoring(panel_df, office=off)
        if not off_scores.empty:
            ax_bs_off.plot(off_scores["days_to_election"], off_scores["brier_score"],
                           label=f"{off} Races", color=office_colors.get(off), alpha=0.85)

    ax_bs_off.axhline(0.25, color="gray", linestyle="--", alpha=0.7, label="Uninformative Prior (0.25)")
    ax_bs_off.set_title("Cross-Sectional Brier Score by Office Type")
    ax_bs_off.set_ylabel(r"Brier Score: $\frac{1}{N} \sum (p_i - y_i)^2$")
    ax_bs_off.legend(loc="upper left", frameon=True)
    ax_bs_off.grid(True, linestyle=":", alpha=0.6)

    # Top-Right: Log Loss by Office
    ax_ll_off = axes[0, 1]
    if not full_scores.empty:
        ax_ll_off.plot(full_scores["days_to_election"], full_scores["log_loss"],
                       label="All Races (Aggregate)", color="black", linewidth=2.5, linestyle="-")

    for off in offices:
        off_scores = compute_dynamic_scoring(panel_df, office=off)
        if not off_scores.empty:
            ax_ll_off.plot(off_scores["days_to_election"], off_scores["log_loss"],
                           label=f"{off} Races", color=office_colors.get(off), alpha=0.85)

    ax_ll_off.axhline(np.log(2.0), color="gray", linestyle="--", alpha=0.7, label=r"Uninformative Prior ($\ln 2 \approx 0.693$)")
    ax_ll_off.set_title("Cross-Sectional Log Loss (Cross-Entropy) by Office Type")
    ax_ll_off.set_ylabel(r"Log Loss: $-\frac{1}{N} \sum [y \ln p + (1-y) \ln(1-p)]$")
    ax_ll_off.legend(loc="upper left", frameon=True)
    ax_ll_off.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Left: Liquidity Tier Comparison (Top Quartile Volume vs Full Sample)
    ax_liq = axes[1, 0]
    top_q_scores = compute_dynamic_scoring(panel_df, liquidity_tier="top_quartile")
    med_scores = compute_dynamic_scoring(panel_df, liquidity_tier="above_median")

    if not full_scores.empty:
        ax_liq.plot(full_scores["days_to_election"], full_scores["brier_score"],
                    label="Full Universe", color="gray", linestyle="-", alpha=0.8)
    if not med_scores.empty:
        ax_liq.plot(med_scores["days_to_election"], med_scores["brier_score"],
                    label="Above-Median Volume", color="#ff7f0e", linestyle="-")
    if not top_q_scores.empty:
        ax_liq.plot(top_q_scores["days_to_election"], top_q_scores["brier_score"],
                    label="Top Quartile Volume (Most Liquid)", color="#2ca02c", linewidth=2.5)

    ax_liq.set_title("Brier Score Sensitivity across Liquidity Tiers")
    ax_liq.set_xlabel(r"Days to Election ($t \in [-365, -1]$)")
    ax_liq.set_ylabel("Brier Score")
    ax_liq.legend(loc="upper left", frameon=True)
    ax_liq.grid(True, linestyle=":", alpha=0.6)

    # Bottom-Right: Active Contract Breadth & Daily Traded Volume
    ax_breadth = axes[1, 1]
    if not full_scores.empty:
        color_breadth = "#4169e1"
        ax_breadth.plot(full_scores["days_to_election"], full_scores["n_races"],
                        color=color_breadth, label=r"Active Market Breadth ($N_t$)", linewidth=2.0)
        ax_breadth.set_ylabel(r"Number of Active Races ($N_t$)", color=color_breadth)
        ax_breadth.tick_params(axis="y", labelcolor=color_breadth)

        # Secondary Y-Axis for Median Volume
        ax_vol = ax_breadth.twinx()
        color_vol = "#e377c2"
        ax_vol.plot(full_scores["days_to_election"], full_scores["median_volume"],
                    color=color_vol, linestyle=":", label="Median Daily Volume ($)", linewidth=1.8)
        ax_vol.set_ylabel("Median Volume ($)", color=color_vol)
        ax_vol.tick_params(axis="y", labelcolor=color_vol)
        ax_vol.grid(False)

    ax_breadth.set_title("Cross-Sectional Breadth & Liquidity Depth Over Time")
    ax_breadth.set_xlabel(r"Days to Election ($t \in [-365, -1]$)")
    ax_breadth.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scoring curves plot: %s", output_path)
    return output_path


def plot_calibration_dot_plot(
    panel_df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    n_bins: int = 5,
    min_trailing_7d_vol: Optional[float] = None,
    output_path: Path = FIGURES_DIR / "calibration_dot_plot.png"
) -> Path:
    """Plot multi-horizon calibration reliability dot plots with 95% Wilson Score error bars.

    Point size is scaled proportionally to traded volume.
    Subplots for T-60, T-30, and T-7 days before Election Day.
    """
    if panel_df.empty:
        logger.warning("Empty panel DataFrame. Skipping calibration plot.")
        return output_path

    if horizons is None:
        horizons = [60, 30, 7]

    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 5.5), sharey=True)
    if n_horizons == 1:
        axes = [axes]

    fig.suptitle("Favorite-Side Calibration & Reliability Curves at Fixed Countdown Horizons",
                 fontsize=15, weight="bold")

    for idx, (h, ax) in enumerate(zip(horizons, axes)):
        cal_res = compute_calibration_curve(
            panel_df,
            horizon_days=h,
            n_bins=n_bins,
            min_trailing_7d_vol=min_trailing_7d_vol,
        )
        bins_df = cal_res["bins_df"]
        ece = cal_res["ece"]
        n_obs = cal_res["n_obs"]

        # 45-degree reference line
        ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", linewidth=1.5, label="Perfect Calibration")

        if not bins_df.empty:
            valid_bins = bins_df.dropna(subset=["empirical_win_rate"]).copy()

            if not valid_bins.empty:
                p_bar = valid_bins["mean_pred_prob"].to_numpy()
                y_hat = valid_bins["empirical_win_rate"].to_numpy()
                ci_l = valid_bins["ci_lower_95"].to_numpy()
                ci_u = valid_bins["ci_upper_95"].to_numpy()
                vols = valid_bins["total_volume"].to_numpy()

                # Calculate asymmetric Wilson CI error bars
                y_err_lower = y_hat - ci_l
                y_err_upper = ci_u - y_hat
                y_err = [y_err_lower, y_err_upper]

                # Scale scatter point size by volume
                max_vol = np.max(vols) if len(vols) > 0 and np.max(vols) > 0 else 1.0
                point_sizes = 80 + 350 * (vols / max_vol)

                # Error bars
                ax.errorbar(
                    p_bar, y_hat, yerr=y_err,
                    fmt="none", ecolor="#1f77b4", elinewidth=1.8, capsize=4, capthick=1.5, alpha=0.75,
                    label="95% Wilson CI"
                )

                # Dot scatter with volume sizing
                scatter = ax.scatter(
                    p_bar, y_hat,
                    s=point_sizes, c="#1f77b4", edgecolors="black", linewidths=1.2,
                    alpha=0.9, zorder=5, label=r"Observed Win Rate (size $\propto$ vol)"
                )

                # Connecting line between calibration points
                sort_idx = np.argsort(p_bar)
                ax.plot(p_bar[sort_idx], y_hat[sort_idx], color="#1f77b4", linestyle=":", alpha=0.6)

        # Labels and cosmetics
        filter_text = f" (Vol >= ${min_trailing_7d_vol:,.0f})" if min_trailing_7d_vol else ""
        ax.set_title(f"Horizon: T-{h} Days{filter_text}\n$N = {n_obs}$, $\\mathrm{{ECE}} = {ece:.3f}$")
        ax.set_xlabel("Mean Implied Probability $\\bar{p}$ (Favorite)")
        if idx == 0:
            ax.set_ylabel("Empirical Win Rate $\\hat{y}$")

        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(0.40, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration dot plot: %s", output_path)
    return output_path


def plot_liquidity_barrier_sensitivity(
    panel_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    horizon_days: int = 30,
    output_path: Path = FIGURES_DIR / "liquidity_barrier_sensitivity.png"
) -> Path:
    """Plot Brier Score and Expected Calibration Error (ECE) across liquidity barriers."""
    if panel_df.empty:
        logger.warning("Empty panel DataFrame. Skipping liquidity sensitivity plot.")
        return output_path

    sens_df = evaluate_liquidity_barrier_sensitivity(panel_df, thresholds=thresholds, horizon_days=horizon_days)
    if sens_df.empty:
        logger.warning("No sensitivity data generated.")
        return output_path

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x_labels = [f"${int(t):,}" for t in sens_df["threshold_vol"]]
    x = np.arange(len(x_labels))

    color_ece = "#d62728"
    color_bs = "#1f77b4"

    l1 = ax1.plot(x, sens_df["ece"], color=color_ece, marker="o", linewidth=2.2, label="ECE (Calibration Error)")
    ax1.set_ylabel("Expected Calibration Error (ECE)", color=color_ece)
    ax1.tick_params(axis="y", labelcolor=color_ece)
    ax1.set_xlabel("Trailing 7-Day Liquidity Barrier Threshold ($)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=25)

    l2 = ax2.plot(x, sens_df["brier_score"], color=color_bs, marker="s", linestyle="--", linewidth=2.2, label="Brier Score")
    ax2.set_ylabel("Brier Score", color=color_bs)
    ax2.tick_params(axis="y", labelcolor=color_bs)
    ax2.grid(False)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title(f"Forecast Accuracy vs. Retail Liquidity Barrier (Horizon: T-{horizon_days} Days)", weight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved liquidity barrier sensitivity plot: %s", output_path)
    return output_path


def main() -> None:
    from config import PANEL_PARQUET
    if not PANEL_PARQUET.exists():
        logger.error("Panel parquet not found at %s. Run pipeline first.", PANEL_PARQUET)
        return
    panel_df = pd.read_parquet(PANEL_PARQUET)
    plot_scoring_curves_over_time(panel_df)
    plot_calibration_dot_plot(panel_df)
    plot_liquidity_barrier_sensitivity(panel_df)
    print("Visualizations created successfully in", FIGURES_DIR)


if __name__ == "__main__":
    main()
