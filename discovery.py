"""Market Discovery & Universe Selection module for Kalshi Election Markets.

Identifies U.S. Congressional (Senate, House) and Gubernatorial races across
both live and historical Kalshi endpoints, parses metadata, extracts candidate/party,
office, state/district, and settlement outcomes, and stores in Parquet and SQLite.
"""

import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import requests

from config import (
    BACKOFF_FACTOR,
    BASE_URL,
    CUTOFF_URL,
    HISTORICAL_MARKETS_URL,
    MARKETS_URL,
    MAX_RETRIES,
    METADATA_PARQUET,
    METADATA_SQLITE,
    REQUEST_TIMEOUT,
    REQUESTS_PER_SECOND,
    SERIES_URL,
    US_STATES,
    USER_AGENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("discovery")


class KalshiAPIClient:
    """Robust HTTP client with rate-limiting, exponential backoff, and cutoff routing."""

    def __init__(self, requests_per_second: float = REQUESTS_PER_SECOND):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json"
        })
        self.min_interval = 1.0 / max(requests_per_second, 0.1)
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        """Issue GET request with rate limiting and retry backoff on 429/5xx."""
        params = params or {}
        retries = 0

        while True:
            self._rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if response.status_code == 429:
                    retries += 1
                    if retries > MAX_RETRIES:
                        response.raise_for_status()
                    retry_after = float(response.headers.get("Retry-After", BACKOFF_FACTOR ** retries))
                    logger.warning("HTTP 429 Too Many Requests. Backing off for %.2f seconds (attempt %d/%d)",
                                   retry_after, retries, MAX_RETRIES)
                    time.sleep(retry_after)
                    continue
                elif response.status_code >= 500:
                    retries += 1
                    if retries > MAX_RETRIES:
                        response.raise_for_status()
                    backoff = BACKOFF_FACTOR ** retries
                    logger.warning("HTTP %d Server Error. Backing off for %.2f seconds (attempt %d/%d)",
                                   response.status_code, backoff, retries, MAX_RETRIES)
                    time.sleep(backoff)
                    continue

                return response
            except requests.RequestException as e:
                retries += 1
                if retries > MAX_RETRIES:
                    logger.error("Max retries exceeded for URL %s: %s", url, e)
                    raise
                backoff = BACKOFF_FACTOR ** retries
                logger.warning("Network error on %s: %s. Retrying in %.2f seconds", url, e, backoff)
                time.sleep(backoff)

    def get_market_settled_cutoff(self) -> Optional[datetime]:
        """Fetch dynamic market settlement cutoff timestamp from Kalshi."""
        try:
            resp = self.get(CUTOFF_URL)
            if resp.status_code == 200:
                data = resp.json()
                ts_str = data.get("market_settled_ts")
                if ts_str:
                    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception as e:
            logger.warning("Could not retrieve market_settled_ts cutoff: %s", e)
        return None


def parse_race_metadata(market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a Kalshi market dict to determine if it is a US House, Senate, or Governor election race.

    Extracts:
      - office ('Senate', 'House', 'Governor')
      - state_district ('AZ', 'NY17', etc.)
      - race_id (e.g. 'Senate_AZ', 'House_NY17')
      - candidate_or_party ('Democrat', 'Republican', or candidate name)
      - open_time, close_time, result, settlement_value
    """
    ticker = market.get("ticker", "").upper()
    title = market.get("title", "")
    event_ticker = market.get("event_ticker", "").upper()
    subtitle = market.get("subtitle", "") or ""
    yes_sub_title = market.get("yes_sub_title", "") or ""

    combined_text = f"{ticker} {event_ticker} {title} {subtitle} {yes_sub_title}".lower()

    # Filter out non-general election or non-race artifacts
    negative_keywords = [
        "parlay", "crosscategory", "bill", "pass", "reconciliation",
        "confirm", "yea", "recess", "shutdown", "approval", "margin",
        "mov", "popular vote", "turnout", "exit poll", "special session",
        "speaker", "majority leader", "whip", "chair", "committee",
        "brazil", "nigeria", "canada", "uk ", "france", "turkey", "spain",
        "philippines", "germany", "austria", "mexico", "japan", "scotland",
        "primary", "runoff", "nominee", "nomination"
    ]
    if any(k in combined_text for k in negative_keywords):
        return None

    office = None
    state_district = None

    # Check for House races: pattern HOUSE{STATE}{DIST} e.g. HOUSENY17, KXHOUSENC11
    m_house = re.search(r'(?:KX)?HOUSE([A-Z]{2})(\d{1,2})', ticker + " " + event_ticker)
    if m_house:
        st, dist = m_house.group(1), m_house.group(2)
        if st in US_STATES:
            office = "House"
            state_district = f"{st}{int(dist):02d}"

    # Check for Senate races: SENATEPARTY-{ST}, SENATEPARTY{ST}, SENATE{ST}, KXSENATE{ST}
    if not office:
        m_sen = re.search(r'(?:KX)?SENATE(?:PARTY-?|RACE-?)?([A-Z]{2})(?:-|\b)', ticker + " " + event_ticker)
        if m_sen:
            st = m_sen.group(1)
            if st in US_STATES:
                office = "Senate"
                state_district = st

    # Check for Governor races: GOVPARTY{ST}, GOV{ST}, KXGOV{ST}
    if not office:
        m_gov = re.search(r'(?:KX)?GOV(?:PARTY-?)?([A-Z]{2})(?:-|\b)', ticker + " " + event_ticker)
        if m_gov:
            st = m_gov.group(1)
            if st in US_STATES:
                office = "Governor"
                state_district = st

    # Additional text-based pattern recognition if ticker regex didn't catch it
    if not office:
        # Search title for US state and office
        for st_code, st_name in US_STATES.items():
            if st_name.lower() in title.lower():
                # Check house district in title e.g. "NY-17" or "California 41st"
                m_title_dist = re.search(rf'{st_code}[ -]?(\d{{1,2}})', title, re.IGNORECASE)
                if "house" in title.lower() and m_title_dist:
                    office = "House"
                    state_district = f"{st_code}{int(m_title_dist.group(1)):02d}"
                    break
                elif "senate" in title.lower() and "state senate" not in title.lower():
                    office = "Senate"
                    state_district = st_code
                    break
                elif "governor" in title.lower() or "governorship" in title.lower():
                    office = "Governor"
                    state_district = st_code
                    break

    if not office or not state_district:
        return None

    race_id = f"{office}_{state_district}"

    # Candidate or Party Representation
    candidate_or_party = yes_sub_title or ""
    if not candidate_or_party:
        # Check ticker suffix -D, -R
        if ticker.endswith("-D") or "-DEM" in ticker:
            candidate_or_party = "Democrat"
        elif ticker.endswith("-R") or "-REP" in ticker:
            candidate_or_party = "Republican"
        else:
            candidate_or_party = title

    # Normalize party name if applicable
    cand_lower = candidate_or_party.lower()
    if "democrat" in cand_lower or cand_lower == "d":
        party = "Democrat"
    elif "republican" in cand_lower or cand_lower == "r":
        party = "Republican"
    else:
        party = "Other"

    # Outcome / Settlement parsing
    raw_result = (market.get("result") or "").strip().lower()
    settlement_val_str = market.get("settlement_value_dollars") or market.get("settlement_value")
    settlement_value: Optional[float] = None

    if raw_result == "yes":
        settlement_value = 1.0
    elif raw_result == "no":
        settlement_value = 0.0
    elif settlement_val_str is not None:
        try:
            val = float(settlement_val_str)
            settlement_value = 1.0 if val >= 0.5 else 0.0
            if raw_result == "":
                raw_result = "yes" if settlement_value == 1.0 else "no"
        except (ValueError, TypeError):
            settlement_value = None

    # Series ticker detection
    series_ticker = market.get("series_ticker") or ""
    if not series_ticker and event_ticker:
        # Many Kalshi event tickers are {SERIES}-{DATE}
        series_ticker = event_ticker.split("-")[0]

    return {
        "ticker": market.get("ticker"),
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "office": office,
        "state_district": state_district,
        "race_id": race_id,
        "title": title,
        "candidate": candidate_or_party,
        "party": party,
        "status": market.get("status"),
        "result": raw_result if raw_result in ["yes", "no"] else None,
        "settlement_value": settlement_value,
        "settlement_ts": market.get("settlement_ts"),
        "open_time": market.get("open_time"),
        "close_time": market.get("close_time"),
        "volume_fp": float(market.get("volume_fp") or 0.0),
        "open_interest_fp": float(market.get("open_interest_fp") or 0.0),
    }


class MarketDiscoveryService:
    """Service to discover and catalog all election race contracts across Kalshi endpoints."""

    def __init__(self, client: Optional[KalshiAPIClient] = None):
        self.client = client or KalshiAPIClient()
        self.cutoff_dt = self.client.get_market_settled_cutoff()
        logger.info("Kalshi historical cutoff timestamp: %s", self.cutoff_dt)

    def is_historical_market(self, settlement_ts_str: Optional[str]) -> bool:
        """Determine if a market settled strictly before market_settled_ts cutoff."""
        if not settlement_ts_str or not self.cutoff_dt:
            return False
        try:
            settled_dt = datetime.fromisoformat(settlement_ts_str.replace("Z", "+00:00"))
            return settled_dt < self.cutoff_dt
        except Exception:
            return False

    def discover_via_series(self) -> List[Dict[str, Any]]:
        """Discover election markets by inspecting all election series (fast, targeted)."""
        logger.info("Scanning Kalshi /series for Congressional and Gubernatorial series...")
        resp = self.client.get(SERIES_URL)
        if resp.status_code != 200:
            logger.error("Failed to query series: %d", resp.status_code)
            return []

        all_series = resp.json().get("series", [])
        election_series_tickers = set()

        for s in all_series:
            st = s.get("ticker", "").upper()
            title = s.get("title", "")
            tags = s.get("tags") or []
            cat = s.get("category", "")

            # Exclude non-relevant series
            if any(t in ["International elections", "Brazil", "Congress", "Primaries", "Combo", "Mixed Combos"] for t in tags):
                continue

            # Check if this series is for Senate, House, or Governor
            is_senate = any("senate" in x.lower() for x in [st, title] + tags)
            is_house = any("house" in x.lower() for x in [st, title] + tags)
            is_gov = any("gov" in x.lower() for x in [st, title] + tags)

            if (is_senate or is_house or is_gov) and cat in ["Elections", "Politics", ""]:
                election_series_tickers.add(st)

        logger.info("Found %d election-related series. Pulling contracts...", len(election_series_tickers))
        discovered = []
        for st in election_series_tickers:
            # Query /markets for this series
            m_resp = self.client.get(MARKETS_URL, params={"series_ticker": st})
            if m_resp.status_code == 200:
                for m in m_resp.json().get("markets", []):
                    parsed = parse_race_metadata(m)
                    if parsed:
                        parsed["is_historical"] = self.is_historical_market(parsed.get("settlement_ts"))
                        discovered.append(parsed)

            # Also check /historical/markets
            hm_resp = self.client.get(HISTORICAL_MARKETS_URL, params={"series_ticker": st})
            if hm_resp.status_code == 200:
                for m in hm_resp.json().get("markets", []):
                    parsed = parse_race_metadata(m)
                    if parsed:
                        parsed["is_historical"] = True
                        discovered.append(parsed)

        return discovered

    def discover_via_pagination(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Paginate /markets and /historical/markets with cursors to discover race markets."""
        discovered: Dict[str, Dict[str, Any]] = {}

        for endpoint_name, endpoint_url in [("live", MARKETS_URL), ("historical", HISTORICAL_MARKETS_URL)]:
            cursor = ""
            pages = 0
            logger.info("Paginating %s markets...", endpoint_name)

            while True:
                params: Dict[str, Any] = {"limit": 1000}
                if cursor:
                    params["cursor"] = cursor

                resp = self.client.get(endpoint_url, params=params)
                if resp.status_code != 200:
                    logger.warning("Pagination error on %s: HTTP %d", endpoint_name, resp.status_code)
                    break

                data = resp.json()
                markets = data.get("markets", [])
                cursor = data.get("cursor", "")

                for m in markets:
                    parsed = parse_race_metadata(m)
                    if parsed:
                        parsed["is_historical"] = (endpoint_name == "historical") or self.is_historical_market(parsed.get("settlement_ts"))
                        discovered[parsed["ticker"]] = parsed

                pages += 1
                if max_pages and pages >= max_pages:
                    break
                if not cursor or not markets:
                    break

        return list(discovered.values())

    def run_discovery(self, use_series_acceleration: bool = True) -> pd.DataFrame:
        """Run full discovery pipeline and return metadata DataFrame."""
        if use_series_acceleration:
            markets = self.discover_via_series()
        else:
            markets = self.discover_via_pagination()

        # Deduplicate by ticker, preferring richer data
        unique_markets: Dict[str, Dict[str, Any]] = {}
        for m in markets:
            t = m["ticker"]
            if t not in unique_markets:
                unique_markets[t] = m
            else:
                if m.get("result") is not None and unique_markets[t].get("result") is None:
                    unique_markets[t] = m

        df = pd.DataFrame(list(unique_markets.values()))
        if df.empty:
            logger.warning("No election markets discovered.")
            return df

        # Sort cleanly
        df = df.sort_values(by=["office", "state_district", "ticker"]).reset_index(drop=True)
        logger.info("Successfully discovered %d election contracts across %d unique races.",
                    len(df), df["race_id"].nunique())
        return df

    def save_metadata(self, df: pd.DataFrame, parquet_path: Path = METADATA_PARQUET, sqlite_path: Path = METADATA_SQLITE) -> None:
        """Save discovered metadata to Parquet and SQLite."""
        if df.empty:
            logger.warning("DataFrame is empty, skipping save.")
            return

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved metadata to Parquet: %s (%d rows)", parquet_path, len(df))

        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as conn:
            df.to_sql("markets_metadata", conn, if_exists="replace", index=False)
        logger.info("Saved metadata to SQLite: %s (table 'markets_metadata')", sqlite_path)


def main() -> None:
    service = MarketDiscoveryService()
    df = service.run_discovery(use_series_acceleration=True)
    service.save_metadata(df)
    print(f"Discovery complete. Discovered {len(df)} contracts.")
    if not df.empty:
        print(df[["ticker", "office", "state_district", "party", "result", "status"]].head(10))


if __name__ == "__main__":
    main()
