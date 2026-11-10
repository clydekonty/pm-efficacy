"""Unit and econometric tests for the Kalshi Election Forecasting Pipeline."""

import math
from datetime import date, timedelta
import numpy as np
import pandas as pd

from discovery import parse_race_metadata
from metrics import (
    calculate_brier_score,
    calculate_log_loss,
    compute_calibration_curve,
    compute_dynamic_scoring,
    wilson_score_interval,
)
from pipeline import PanelBuilder, parse_candle_price_to_prob


def test_wilson_score_interval():
    """Verify Wilson score confidence interval properties and edge cases."""
    # 0 trials edge case
    lower, upper = wilson_score_interval(0, 0)
    assert math.isnan(lower) and math.isnan(upper)

    # 10 successes out of 20 trials (50% empirical)
    l, u = wilson_score_interval(10, 20, confidence=0.95)
    assert 0.0 < l < 0.50
    assert 0.50 < u < 1.0
    # Symmetric around 0.5 for p=0.5
    assert abs((l + u) / 2.0 - 0.5) < 1e-4

    # 0 successes out of 10 trials
    l_zero, u_zero = wilson_score_interval(0, 10, confidence=0.95)
    assert l_zero == 0.0
    assert u_zero > 0.0

    # 10 successes out of 10 trials
    l_full, u_full = wilson_score_interval(10, 10, confidence=0.95)
    assert l_full < 1.0
    assert math.isclose(u_full, 1.0, abs_tol=1e-5)


def test_scoring_rules_math():
    """Verify Brier Score and Log Loss calculation against known analytic values."""
    # Perfect forecast: BS = 0, Log Loss ~ 0
    p_perfect = np.array([1.0, 0.0, 1.0])
    y_perfect = np.array([1, 0, 1])
    assert calculate_brier_score(p_perfect, y_perfect) == 0.0
    assert calculate_log_loss(p_perfect, y_perfect) < 0.01

    # Completely uninformative coin-flip: BS = 0.25, LL = ln(2) ~ 0.6931
    p_coin = np.array([0.5, 0.5, 0.5, 0.5])
    y_coin = np.array([1, 0, 1, 0])
    assert math.isclose(calculate_brier_score(p_coin, y_coin), 0.25, abs_tol=1e-5)
    assert math.isclose(calculate_log_loss(p_coin, y_coin), math.log(2.0), abs_tol=1e-3)

    # Inverted forecast: BS = 1.0
    p_bad = np.array([0.0, 1.0])
    y_bad = np.array([1, 0])
    assert calculate_brier_score(p_bad, y_bad) == 1.0


def test_candlestick_price_normalization():
    """Verify normalization of live close_dollars, historical cents, and mid-market quote."""
    # Live dollar format string
    c1 = {"price": {"close_dollars": "0.6800"}}
    assert math.isclose(parse_candle_price_to_prob(c1), 0.68, abs_tol=1e-4)

    # Historical cents format
    c2 = {"price": {"close": 74}}
    assert math.isclose(parse_candle_price_to_prob(c2), 0.74, abs_tol=1e-4)

    # Empty price, mid quote from bid / ask
    c3 = {
        "price": {},
        "yes_bid": {"close_dollars": "0.5800"},
        "yes_ask": {"close_dollars": "0.6200"}
    }
    assert math.isclose(parse_candle_price_to_prob(c3), 0.60, abs_tol=1e-4)


def test_fold_two_party_races_to_favorite():
    """Verify folding two-party races creates an unambiguous [0.50, 0.99] distribution."""
    builder = PanelBuilder()

    # Case A: Candidate is favorite (p = 0.75) and wins (settlement_value = 1.0)
    df_a = pd.DataFrame([{
        "race_id": "Senate_AZ",
        "office": "Senate",
        "state_district": "AZ",
        "date": date(2026, 10, 1),
        "days_to_election": -33,
        "implied_prob": 0.75,
        "volume": 1000.0,
        "trailing_7d_vol": 5000.0,
        "settlement_value": 1.0
    }])
    folded_a = builder.fold_two_party_races_to_favorite(df_a)
    assert len(folded_a) == 1
    assert math.isclose(folded_a.iloc[0]["implied_prob"], 0.75)
    assert folded_a.iloc[0]["actual_winner"] == 1

    # Case B: Candidate is underdog (p = 0.35) and underdog lost (settlement_value = 0.0)
    # The favorite was 1 - 0.35 = 0.65, and favorite won!
    df_b = pd.DataFrame([{
        "race_id": "House_NY17",
        "office": "House",
        "state_district": "NY17",
        "date": date(2026, 10, 1),
        "days_to_election": -33,
        "implied_prob": 0.35,
        "volume": 800.0,
        "trailing_7d_vol": 4000.0,
        "settlement_value": 0.0
    }])
    folded_b = builder.fold_two_party_races_to_favorite(df_b)
    assert len(folded_b) == 1
    assert math.isclose(folded_b.iloc[0]["implied_prob"], 0.65)
    assert folded_b.iloc[0]["actual_winner"] == 1  # The favorite won


def test_race_metadata_parsing():
    """Verify regex and title parsing for House, Senate, and Governor races."""
    # Governor test
    m_gov = {
        "ticker": "GOVPARTYNH-28-R",
        "title": "Will the Republican party win the governorship in New Hampshire",
        "event_ticker": "GOVPARTYNH-28",
        "yes_sub_title": "Kelly Ayotte",
        "result": "yes",
        "status": "finalized"
    }
    parsed_gov = parse_race_metadata(m_gov)
    assert parsed_gov is not None
    assert parsed_gov["office"] == "Governor"
    assert parsed_gov["state_district"] == "NH"
    assert parsed_gov["race_id"] == "Governor_NH"
    assert parsed_gov["result"] == "yes"
    assert parsed_gov["settlement_value"] == 1.0

    # House test
    m_house = {
        "ticker": "HOUSENY17-26-R",
        "title": "Will Republican win the House race for NY-17?",
        "event_ticker": "HOUSENY17-26",
        "yes_sub_title": "Mike Lawler",
        "result": "no",
        "status": "finalized"
    }
    parsed_house = parse_race_metadata(m_house)
    assert parsed_house is not None
    assert parsed_house["office"] == "House"
    assert parsed_house["state_district"] == "NY17"
    assert parsed_house["race_id"] == "House_NY17"
    assert parsed_house["result"] == "no"
    assert parsed_house["settlement_value"] == 0.0

    # Negative filter: international parlay should be rejected
    m_parlay = {
        "ticker": "KXMVECROSSCATEGORY-S2026",
        "title": "Parlay multi-event cross category",
        "event_ticker": "KXMVECROSSCATEGORY",
    }
    assert parse_race_metadata(m_parlay) is None


def test_calibration_and_ece():
    """Verify calibration binning and Expected Calibration Error (ECE) computation."""
    records = []
    # Create 100 observations perfectly calibrated
    for _ in range(50):
        records.append({
            "race_id": f"race_{_}",
            "office": "Senate",
            "state_district": "PA",
            "date": date(2026, 10, 4),
            "days_to_election": -30,
            "implied_prob": 0.60,
            "volume": 1000.0,
            "trailing_7d_vol": 5000.0,
            "actual_winner": 1 if _ < 30 else 0  # 30/50 = 0.60
        })
    for _ in range(50, 100):
        records.append({
            "race_id": f"race_{_}",
            "office": "House",
            "state_district": "CA41",
            "date": date(2026, 10, 4),
            "days_to_election": -30,
            "implied_prob": 0.80,
            "volume": 2000.0,
            "trailing_7d_vol": 10000.0,
            "actual_winner": 1 if _ < 90 else 0  # 40/50 = 0.80
        })

    df = pd.DataFrame(records)
    cal_res = compute_calibration_curve(df, horizon_days=30, n_bins=5)
    assert cal_res["n_obs"] == 100
    # Perfect calibration should yield ECE close to 0
    assert cal_res["ece"] < 0.01


if __name__ == "__main__":
    test_wilson_score_interval()
    print("✓ test_wilson_score_interval passed")
    test_scoring_rules_math()
    print("✓ test_scoring_rules_math passed")
    test_candlestick_price_normalization()
    print("✓ test_candlestick_price_normalization passed")
    test_fold_two_party_races_to_favorite()
    print("✓ test_fold_two_party_races_to_favorite passed")
    test_race_metadata_parsing()
    print("✓ test_race_metadata_parsing passed")
    test_calibration_and_ece()
    print("✓ test_calibration_and_ece passed")
    print("\nAll unit and econometric tests passed successfully!")

