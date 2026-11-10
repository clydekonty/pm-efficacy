"""End-to-End Orchestrator for Kalshi Election Market Forecasting Efficacy Pipeline.

Provides a unified CLI interface to:
1. Discover and catalog US Senate, House, and Governor election contracts.
2. Harvest daily candlesticks and build the continuous panel DataFrame.
3. Compute dynamic Brier scores, Log Loss, and calibration tables.
4. Render publication-ready visualizations.
5. Generate synthetic benchmark data for offline econometric testing.
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_LIQUIDITY_THRESHOLD,
    FIGURES_DIR,
    METADATA_PARQUET,
    METADATA_SQLITE,
    PANEL_PARQUET,
    US_STATES,
)
from discovery import MarketDiscoveryService
from metrics import (
    compute_calibration_curve,
    compute_dynamic_scoring,
    evaluate_liquidity_barrier_sensitivity,
)
from pipeline import PanelBuilder
from visualize import (
    plot_calibration_dot_plot,
    plot_liquidity_barrier_sensitivity,
    plot_scoring_curves_over_time,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def generate_benchmark_dataset(
    n_races: int = 60,
    days: int = 365,
    seed: int = 42,
    output_path: Path = PANEL_PARQUET
) -> pd.DataFrame:
    """Generate realistic synthetic election race panel data for offline testing and econometric validation.

    Simulates realistic prediction market price paths:
      - 50 states and districts across Senate, House, Governor.
      - Mean-reverting random walk with drift toward actual outcome as t -> 0.
      - Realistic volume distributions with power-law retail and institutional trading spikes near Election Day.
      - Strict canonical schema: ['race_id', 'office', 'state_district', 'date', 'days_to_election', 'implied_prob', 'volume', 'actual_winner'].
    """
    logger.info("Generating realistic benchmark dataset (%d races over %d days)...", n_races, days)
    np.random.seed(seed)

    state_list = list(US_STATES.keys())
    offices = ["Senate", "House", "Governor"]

    races = []
    for i in range(n_races):
        office = np.random.choice(offices, p=[0.35, 0.45, 0.20])
        st = np.random.choice(state_list)
        if office == "House":
            dist = np.random.randint(1, 15)
            st_dist = f"{st}{dist:02d}"
        else:
            st_dist = st

        race_id = f"{office}_{st_dist}"
        # Latent true probability of the favorite winning
        true_fav_prob = np.random.uniform(0.52, 0.95)
        # Outcome realized from Bernoulli(true_fav_prob)
        actual_winner = int(np.random.rand() < true_fav_prob)
        base_volume = np.random.lognormal(mean=9.5, sigma=1.2)  # ~$10k to $100k daily

        races.append({
            "race_id": race_id,
            "office": office,
            "state_district": st_dist,
            "true_fav_prob": true_fav_prob,
            "actual_winner": actual_winner,
            "base_volume": base_volume,
        })

    end_date = date(2026, 11, 3)
    start_date = end_date - timedelta(days=days)

    records = []
    for r in races:
        winner = r["actual_winner"]
        # Initial consensus at T-365
        p = np.random.uniform(0.50, 0.85)
        # Drift target: moves toward 1.0 if winner=1, toward 0.50 or lower if winner=0
        target = 0.95 if winner == 1 else 0.52

        for step in range(days):
            dte = -(days - step)  # -365 to -1
            curr_date = start_date + timedelta(days=step)

            # Volatility increases as election approaches; drift pulls toward true outcome
            time_decay = (step + 1) / days
            drift = 0.005 * (target - p)
            noise = np.random.normal(0.0, 0.02 * (1.0 + 0.5 * time_decay))
            p = np.clip(p + drift + noise, 0.50, 0.99)

            # Volume surges as countdown decreases
            vol = r["base_volume"] * (0.3 + 2.5 * (time_decay ** 2)) * np.random.lognormal(0.0, 0.4)

            records.append({
                "race_id": r["race_id"],
                "office": r["office"],
                "state_district": r["state_district"],
                "date": curr_date,
                "days_to_election": dte,
                "implied_prob": float(p),
                "volume": float(vol),
                "actual_winner": winner,
            })

    df = pd.DataFrame(records)
    # Add trailing 7-day volume
    df = df.sort_values(by=["race_id", "date"]).reset_index(drop=True)
    df["trailing_7d_vol"] = df.groupby("race_id")["volume"].transform(
        lambda s: s.rolling(window=7, min_periods=1).sum()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved benchmark panel to %s (%d rows)", output_path, len(df))
    return df


def run_pipeline(
    discover: bool = False,
    harvest: bool = False,
    evaluate: bool = False,
    visualize: bool = False,
    benchmark_mock: bool = False,
    horizons: list = DEFAULT_EVALUATION_HORIZONS,
    liquidity_barrier: float = DEFAULT_LIQUIDITY_THRESHOLD,
) -> None:
    """Execute end-to-end pipeline stages."""
    print("=" * 70)
    print("  KALSHI ELECTION FORECAST EFFICIENCY & PANEL HARVESTER PIPELINE  ")
    print("=" * 70)

    # 1. Benchmark Mock Data Mode
    if benchmark_mock:
        logger.info("Running in BENCHMARK MOCK mode...")
        panel_df = generate_benchmark_dataset()
        evaluate = True
        visualize = True

    # 2. Market Discovery
    if discover:
        logger.info("[Stage 1/4] Starting Market Discovery & Universe Selection...")
        discovery_service = MarketDiscoveryService()
        meta_df = discovery_service.run_discovery(use_series_acceleration=True)
        discovery_service.save_metadata(meta_df, parquet_path=METADATA_PARQUET, sqlite_path=METADATA_SQLITE)
        print(f"[*] Discovery complete: {len(meta_df)} contracts cataloged.")

    # 3. Candlestick Harvester & Panel Alignment
    if harvest:
        logger.info("[Stage 2/4] Harvesting Daily Candlesticks & Aligning Master Panel...")
        if not METADATA_PARQUET.exists():
            logger.error("Metadata file %s not found. Please run discovery first (--discover).", METADATA_PARQUET)
            sys.exit(1)
        meta_df = pd.read_parquet(METADATA_PARQUET)
        builder = PanelBuilder()
        panel_df = builder.build_panel(meta_df)
        builder.save_panel(panel_df, path=PANEL_PARQUET)
        print(f"[*] Panel construction complete: {len(panel_df)} panel observations.")

    # 4. Statistical Evaluation
    if evaluate:
        logger.info("[Stage 3/4] Evaluating Forecast Efficiency & Calibration...")
        if not PANEL_PARQUET.exists():
            logger.error("Panel file %s not found. Please run pipeline first (--pipeline).", PANEL_PARQUET)
            sys.exit(1)
        panel_df = pd.read_parquet(PANEL_PARQUET)

        # Dynamic Scoring Table Sample
        dyn_scoring = compute_dynamic_scoring(panel_df)
        print("\n--- Cross-Sectional Scoring Rules (Countdown Samples) ---")
        sample_dtes = [-365, -180, -90, -60, -30, -14, -7, -1]
        sample_df = dyn_scoring[dyn_scoring["days_to_election"].isin(sample_dtes)]
        print(sample_df[["days_to_election", "n_races", "brier_score", "log_loss", "mean_prob", "win_rate"]].to_string(index=False))

        # Fixed Horizon Calibration Curves
        print("\n--- Calibration & Expected Calibration Error (ECE) at Fixed Horizons ---")
        for h in horizons:
            cal = compute_calibration_curve(panel_df, horizon_days=h)
            print(f"\n[Horizon T-{h} Days] Total Races: {cal['n_obs']} | Brier Score: {cal['brier_score']:.4f} | ECE: {cal['ece']:.4f} | MCE: {cal['mce']:.4f}")
            print(cal["bins_df"][["bin_range", "n_obs", "mean_pred_prob", "empirical_win_rate", "ci_lower_95", "ci_upper_95", "abs_calibration_gap"]].to_string(index=False))

        # Liquidity Barrier Sensitivity
        print(f"\n--- Liquidity Barrier Sensitivity Analysis (Threshold = ${liquidity_barrier:,.0f}) ---")
        sens_df = evaluate_liquidity_barrier_sensitivity(panel_df, horizon_days=30)
        print(sens_df.to_string(index=False))

    # 5. Visualization Generation
    if visualize:
        logger.info("[Stage 4/4] Generating Publication-Quality Figures...")
        if not PANEL_PARQUET.exists():
            logger.error("Panel file %s not found. Please run pipeline first (--pipeline).", PANEL_PARQUET)
            sys.exit(1)
        panel_df = pd.read_parquet(PANEL_PARQUET)

        p1 = plot_scoring_curves_over_time(panel_df)
        p2 = plot_calibration_dot_plot(panel_df, horizons=horizons)
        p3 = plot_liquidity_barrier_sensitivity(panel_df, horizon_days=30)

        print(f"\n[*] Visualizations saved:")
        print(f"    - Scoring curves: {p1}")
        print(f"    - Calibration dot plots: {p2}")
        print(f"    - Liquidity barrier sensitivity: {p3}")

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY  ")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi Election Forecast Efficiency & Panel Harvester Pipeline")
    parser.add_argument("--discover", action="store_true", help="Run market discovery & universe selection")
    parser.add_argument("--pipeline", action="store_true", help="Harvest candlesticks and construct panel_df.parquet")
    parser.add_argument("--evaluate", action="store_true", help="Compute scoring rules, ECE, Wilson CI, and calibration")
    parser.add_argument("--visualize", action="store_true", help="Generate publication-grade diagnostic plots")
    parser.add_argument("--all", action="store_true", help="Run full pipeline end-to-end")
    parser.add_argument("--benchmark-mock", action="store_true", help="Generate benchmark dataset and test all econometric modules")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_EVALUATION_HORIZONS, help="Evaluation horizons in days to election")
    parser.add_argument("--liquidity-barrier", type=float, default=DEFAULT_LIQUIDITY_THRESHOLD, help="Trailing 7-day liquidity threshold in dollars")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        # Default behavior: run complete workflow
        run_pipeline(
            discover=True,
            harvest=True,
            evaluate=True,
            visualize=True,
            horizons=args.horizons,
            liquidity_barrier=args.liquidity_barrier,
        )
    else:
        run_pipeline(
            discover=args.discover,
            harvest=args.pipeline,
            evaluate=args.evaluate,
            visualize=args.visualize,
            benchmark_mock=args.benchmark_mock,
            horizons=args.horizons,
            liquidity_barrier=args.liquidity_barrier,
        )


if __name__ == "__main__":
    main()
