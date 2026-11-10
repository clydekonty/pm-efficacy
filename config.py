"""Configuration settings, endpoints, file paths, and constants for Kalshi Election Pipeline."""

import os
from pathlib import Path
from typing import Dict

# Kalshi Trade API v2 Base URLs
BASE_URL = os.environ.get("KALSHI_API_URL", "https://api.elections.kalshi.com/trade-api/v2")
CUTOFF_URL = f"{BASE_URL}/historical/cutoff"
MARKETS_URL = f"{BASE_URL}/markets"
HISTORICAL_MARKETS_URL = f"{BASE_URL}/historical/markets"
SERIES_URL = f"{BASE_URL}/series"
EVENTS_URL = f"{BASE_URL}/events"

# Directory Structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
CACHE_DIR = DATA_DIR / "cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
METADATA_PARQUET = DATA_DIR / "markets_metadata.parquet"
METADATA_SQLITE = DATA_DIR / "markets_metadata.db"
PANEL_PARQUET = DATA_DIR / "panel_df.parquet"

# Rate Limiting & Networking
REQUESTS_PER_SECOND = float(os.environ.get("KALSHI_RPS", "10.0"))
MAX_RETRIES = int(os.environ.get("KALSHI_MAX_RETRIES", "5"))
BACKOFF_FACTOR = float(os.environ.get("KALSHI_BACKOFF", "1.5"))
REQUEST_TIMEOUT = int(os.environ.get("KALSHI_TIMEOUT", "15"))
USER_AGENT = "Kalshi-Efficacy-Pipeline/1.0 (Quant-Research)"

# Market Ingestion Constraints
CANDLE_PERIOD_MINUTES = 1440  # 1 day in minutes
MAX_CANDLES_PER_QUERY = 5000
MAX_LOOKBACK_DAYS = 365
SECONDS_PER_DAY = 86400

# Scoring & Metric Parameters
LOG_LOSS_EPSILON = 0.005  # clip probabilities to [0.005, 0.995]
WILSON_CONFIDENCE_LEVEL = 0.95
DEFAULT_EVALUATION_HORIZONS = [7, 30, 60]  # Days to election
DEFAULT_LIQUIDITY_THRESHOLD = 25000.0      # Trailing 7-day cumulative volume barrier ($25k)

# US State Postal Codes and Names
US_STATES: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
}
