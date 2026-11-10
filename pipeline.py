"""Time-Series Extraction & Panel Alignment Module for Kalshi Election Markets.

Pulls daily candlesticks (1440m period) for discovered election contracts,
resamples to continuous daily dates t in [T-365, T-1], forward-fills prices,
computes trailing liquidity, folds two-party races to the favorite side (p >= 0.50),
and emits the master panel DataFrame.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BASE_URL,
    CANDLE_PERIOD_MINUTES,
    MAX_CANDLES_PER_QUERY,
    MAX_LOOKBACK_DAYS,
    METADATA_PARQUET,
    PANEL_PARQUET,
    SECONDS_PER_DAY,
)
from discovery import KalshiAPIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


class CandlestickHarvester:
    """Harvests historical daily candlesticks from Kalshi Trade API v2."""

    def __init__(self, client: Optional[KalshiAPIClient] = None):
        self.client = client or KalshiAPIClient()

    def fetch_candlesticks_window(
        self,
        ticker: str,
        series_ticker: Optional[str],
        is_historical: bool,
        start_ts: int,
        end_ts: int,
        period_interval: int = CANDLE_PERIOD_MINUTES,
    ) -> List[Dict[str, Any]]:
        """Fetch candlesticks for a specific time window, handling Kalshi endpoint routing."""
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        # Route to appropriate endpoint based on settlement / historical status
        urls_to_try = []
        if is_historical:
            urls_to_try.append(f"{BASE_URL}/historical/markets/{ticker}/candlesticks")
            if series_ticker:
                urls_to_try.append(f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks")
        else:
            if series_ticker:
                urls_to_try.append(f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks")
            urls_to_try.append(f"{BASE_URL}/markets/{ticker}/candlesticks")
            urls_to_try.append(f"{BASE_URL}/historical/markets/{ticker}/candlesticks")

        candlesticks = []
        for url in urls_to_try:
            try:
                resp = self.client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    candlesticks = data.get("candlesticks", [])
                    return candlesticks
            except Exception as e:
                logger.debug("Failed endpoint %s for %s: %s", url, ticker, e)

        return candlesticks

    def fetch_market_candlesticks(
        self,
        ticker: str,
        series_ticker: Optional[str],
        is_historical: bool,
        start_ts: int,
        end_ts: int,
    ) -> List[Dict[str, Any]]:
        """Fetch candlesticks with chunking if time range exceeds MAX_CANDLES_PER_QUERY."""
        total_days = max(1, (end_ts - start_ts) // SECONDS_PER_DAY)
        if total_days <= MAX_CANDLES_PER_QUERY:
            return self.fetch_candlesticks_window(ticker, series_ticker, is_historical, start_ts, end_ts)

        # Chunk across multiple requests
        all_candles = []
        chunk_seconds = MAX_CANDLES_PER_QUERY * SECONDS_PER_DAY
        curr_start = start_ts
        while curr_start < end_ts:
            curr_end = min(curr_start + chunk_seconds, end_ts)
            candles = self.fetch_candlesticks_window(ticker, series_ticker, is_historical, curr_start, curr_end)
            all_candles.extend(candles)
            curr_start = curr_end

        # Deduplicate candles by end_period_ts
        seen_ts = set()
        deduped = []
        for c in all_candles:
            ts = c.get("end_period_ts")
            if ts not in seen_ts:
                seen_ts.add(ts)
                deduped.append(c)
        return deduped


def parse_candle_price_to_prob(candle: Dict[str, Any]) -> Optional[float]:
    """Parse close price from candlestick into probability p in [0.0, 1.0].

    Safely checks 'price.close_dollars', 'price.close', and mid-market spread.
    Converts cents to dollars when price > 1.0.
    """
    price_obj = candle.get("price") or {}
    val = price_obj.get("close_dollars")
    if val is None:
        val = price_obj.get("close")
    if val is None:
        val = price_obj.get("previous_dollars")

    # If no trade recorded, try mid-quote from yes_bid and yes_ask
    if val is None:
        bid = (candle.get("yes_bid") or {}).get("close_dollars")
        ask = (candle.get("yes_ask") or {}).get("close_dollars")
        if bid is not None and ask is not None:
            try:
                val = (float(bid) + float(ask)) / 2.0
            except (ValueError, TypeError):
                val = None
        elif bid is not None:
            val = bid
        elif ask is not None:
            val = ask

    if val is None:
        return None

    try:
        p = float(val)
        if p > 1.0:  # cents representation (e.g. 65 cents -> 0.65)
            p = p / 100.0
        return max(0.0, min(1.0, p))
    except (ValueError, TypeError):
        return None


def parse_candle_volume(candle: Dict[str, Any]) -> float:
    """Parse daily contract volume from candle."""
    val = candle.get("volume_fp")
    if val is None:
        val = candle.get("volume")
    if val is None:
        return 0.0
    try:
        return max(0.0, float(val))
    except (ValueError, TypeError):
        return 0.0


def parse_candle_open_interest(candle: Dict[str, Any]) -> float:
    """Parse open interest from candle."""
    val = candle.get("open_interest_fp")
    if val is None:
        val = candle.get("open_interest")
    if val is None:
        return 0.0
    try:
        return max(0.0, float(val))
    except (ValueError, TypeError):
        return 0.0


class PanelBuilder:
    """Cleans, aligns, and builds the daily panel DataFrame."""

    def __init__(self, harvester: Optional[CandlestickHarvester] = None):
        self.harvester = harvester or CandlestickHarvester()

    def process_market_series(self, market_meta: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch and align time series for a single market contract."""
        ticker = market_meta["ticker"]
        series_ticker = market_meta.get("series_ticker")
        is_historical = market_meta.get("is_historical", False)

        open_time_str = market_meta.get("open_time")
        close_time_str = market_meta.get("close_time")
        if not close_time_str:
            return None

        try:
            close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        except Exception:
            return None

        election_day = close_dt.date()
        # Election Day - 1 day (T-1)
        t_minus_1_date = election_day - timedelta(days=1)
        # Election Day - 365 days (T-365)
        t_minus_365_date = election_day - timedelta(days=MAX_LOOKBACK_DAYS)

        # Parse open_time
        if open_time_str:
            try:
                open_dt = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
                start_date = max(open_dt.date(), t_minus_365_date)
            except Exception:
                start_date = t_minus_365_date
        else:
            start_date = t_minus_365_date

        if start_date > t_minus_1_date:
            return None

        start_ts = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        # Query up to end of T-1 day
        end_ts = int(datetime.combine(election_day, datetime.min.time(), tzinfo=timezone.utc).timestamp())

        candles = self.harvester.fetch_market_candlesticks(
            ticker=ticker,
            series_ticker=series_ticker,
            is_historical=is_historical,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        # Build raw observations
        records = []
        for c in candles:
            end_ts_val = c.get("end_period_ts")
            if not end_ts_val:
                continue
            candle_dt = datetime.fromtimestamp(end_ts_val, tz=timezone.utc)
            candle_date = candle_dt.date()

            prob = parse_candle_price_to_prob(c)
            vol = parse_candle_volume(c)
            oi = parse_candle_open_interest(c)

            records.append({
                "date": candle_date,
                "implied_prob": prob,
                "volume": vol,
                "open_interest": oi,
            })

        # Build continuous daily date grid
        full_date_range = pd.date_range(start=start_date, end=t_minus_1_date, freq="D").date
        if len(full_date_range) == 0:
            return None

        if records:
            raw_df = pd.DataFrame(records).dropna(subset=["date"])
            # In case multiple candles appear on same date, take last candle
            raw_df = raw_df.groupby("date").agg({
                "implied_prob": "last",
                "volume": "sum",
                "open_interest": "last"
            }).reset_index()
            merged = pd.DataFrame({"date": full_date_range}).merge(raw_df, on="date", how="left")
        else:
            merged = pd.DataFrame({"date": full_date_range})
            merged["implied_prob"] = np.nan
            merged["volume"] = 0.0
            merged["open_interest"] = 0.0

        # Resample daily continuous & forward-fill implied_prob
        merged["implied_prob"] = merged["implied_prob"].ffill()
        # Backward fill any leading missing prices with the earliest traded consensus
        merged["implied_prob"] = merged["implied_prob"].bfill()

        # If completely unpriced, fall back to 0.50 prior
        merged["implied_prob"] = merged["implied_prob"].fillna(0.50)

        merged["volume"] = merged["volume"].fillna(0.0)
        merged["open_interest"] = merged["open_interest"].ffill().fillna(0.0)

        # Calculate trailing 7-day cumulative volume
        merged["trailing_7d_vol"] = merged["volume"].rolling(window=7, min_periods=1).sum()

        # Countdown days to election: t in [-365, -1]
        merged["days_to_election"] = [(d - election_day).days for d in merged["date"]]

        # Metadata columns
        merged["ticker"] = ticker
        merged["race_id"] = market_meta["race_id"]
        merged["office"] = market_meta["office"]
        merged["state_district"] = market_meta["state_district"]
        merged["party"] = market_meta.get("party", "Other")
        merged["settlement_value"] = market_meta.get("settlement_value")

        return merged

    def fold_two_party_races_to_favorite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fold binary two-party races to the favorite side (p >= 0.50).

        Creates an unambiguous [0.50, 0.99] probability distribution.
        For each (race_id, date), selects or transforms the favorite outcome:
          - p_fav = max(p, 1 - p), clipped to [0.50, 0.99]
          - actual_winner = 1 if the favorite won, 0 if the favorite lost.
        """
        if df.empty:
            return df

        folded_rows = []

        # Group by unique race and date
        for (race_id, date), group in df.groupby(["race_id", "date"]):
            # If multiple contracts exist (e.g. D and R), find the contract with p >= 0.50
            fav_candidate_row = group.loc[group["implied_prob"].idxmax()]

            p_raw = fav_candidate_row["implied_prob"]
            settle_val = fav_candidate_row["settlement_value"]

            # If p_raw >= 0.50, the candidate is the favorite
            if p_raw >= 0.50:
                p_fav = p_raw
                winner = int(settle_val == 1.0) if settle_val is not None else np.nan
            else:
                # If single contract had p < 0.50, fold to opposite favorite side
                p_fav = 1.0 - p_raw
                winner = int(settle_val == 0.0) if settle_val is not None else np.nan

            # Clamp probability to [0.50, 0.99]
            p_fav_clamped = min(0.99, max(0.50, float(p_fav)))

            # Aggregate total race volume on this day across legs
            day_volume = group["volume"].sum()
            day_trailing_7d = group["trailing_7d_vol"].sum()

            folded_rows.append({
                "race_id": race_id,
                "office": fav_candidate_row["office"],
                "state_district": fav_candidate_row["state_district"],
                "date": date,
                "days_to_election": fav_candidate_row["days_to_election"],
                "implied_prob": p_fav_clamped,
                "volume": day_volume,
                "trailing_7d_vol": day_trailing_7d,
                "actual_winner": winner,
            })

        panel_df = pd.DataFrame(folded_rows)
        # Drop rows where actual_winner is unfinalized / unknown
        panel_df = panel_df.dropna(subset=["actual_winner"]).copy()
        panel_df["actual_winner"] = panel_df["actual_winner"].astype(int)

        # Sort cleanly by race_id and date
        panel_df = panel_df.sort_values(by=["race_id", "date"]).reset_index(drop=True)
        return panel_df

    def build_panel(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Run candlestick harvesting and build master panel DataFrame."""
        if metadata_df.empty:
            logger.warning("Metadata DataFrame is empty.")
            return pd.DataFrame()

        logger.info("Processing time series for %d contracts...", len(metadata_df))
        contract_dfs = []
        for idx, row in metadata_df.iterrows():
            market_meta = row.to_dict()
            ticker = market_meta["ticker"]
            logger.info("[%d/%d] Fetching candlesticks for %s (%s)...",
                        idx + 1, len(metadata_df), ticker, market_meta.get("race_id"))
            res_df = self.process_market_series(market_meta)
            if res_df is not None and not res_df.empty:
                contract_dfs.append(res_df)

        if not contract_dfs:
            logger.warning("No candlestick data could be retrieved.")
            return pd.DataFrame()

        combined_raw = pd.concat(contract_dfs, ignore_index=True)
        logger.info("Raw candlestick series combined: %d total observations. Folding to favorite side...",
                    len(combined_raw))

        panel = self.fold_two_party_races_to_favorite(combined_raw)
        logger.info("Final master panel constructed: %d observations across %d races.",
                    len(panel), panel["race_id"].nunique())
        return panel

    def save_panel(self, df: pd.DataFrame, path: Path = PANEL_PARQUET) -> None:
        """Save master panel DataFrame to Parquet."""
        if df.empty:
            logger.warning("Panel DataFrame is empty. Skipping save.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Verify required columns exist
        required_cols = [
            "race_id", "office", "state_district", "date",
            "days_to_election", "implied_prob", "volume", "actual_winner"
        ]
        out_df = df[required_cols + [c for c in df.columns if c not in required_cols]]
        out_df.to_parquet(path, index=False)
        logger.info("Master panel saved to %s (%d rows)", path, len(out_df))


def main() -> None:
    if not METADATA_PARQUET.exists():
        logger.error("Metadata parquet not found at %s. Run discovery first.", METADATA_PARQUET)
        return
    meta_df = pd.read_parquet(METADATA_PARQUET)
    builder = PanelBuilder()
    panel_df = builder.build_panel(meta_df)
    builder.save_panel(panel_df)
    print("Panel build complete. Shape:", panel_df.shape)
    if not panel_df.empty:
        print(panel_df.head(10))


if __name__ == "__main__":
    main()
