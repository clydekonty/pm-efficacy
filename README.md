# Kalshi Election Forecast Efficiency & Panel Harvester Pipeline

A modular Python quantitative pipeline designed to pull daily time-series contract data for U.S. Congressional (Senate, House) and Gubernatorial races from Kalshi's Trade API v2, build a daily continuous panel of market-implied probabilities leading up to Election Day ($t \in [T-365, T-1]$), and evaluate forecast efficiency using proper scoring rules (Brier Score, Log Loss) and empirical calibration curves with Wilson Score confidence intervals.

---

## 1. Architecture & Timing Constraints

Kalshi partitions its exchange data across a dynamic settlement cutoff timestamp (`GET /trade-api/v2/historical/cutoff`):
- **Live / Unsettled Contracts** (or settled after `market_settled_ts`): Routed to `/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks` or `/trade-api/v2/markets/...`.
- **Historical Settled Contracts** (settled before `market_settled_ts`): Routed to `/trade-api/v2/historical/markets/{ticker}/candlesticks`.
- **Price Normalization**: Safely parses both `price.close_dollars` (live string/float in dollars) and `price.close` (historical integer cents $\div 100$), falling back to mid-market quote $((P_{bid} + P_{ask}) / 2)$ when days have zero executed trades. All probabilities are normalized strictly to $p \in [0.0, 1.0]$.

---

## 2. Project Structure

```text
pm-efficacy/
├── config.py             # API endpoints, rate limiting, directory paths, state constants
├── discovery.py          # Market discovery, cutoff routing, regex parsing, parquet/SQLite export
├── pipeline.py           # Candlestick harvester, daily resampling, forward-fill, favorite-side folding
├── metrics.py            # Brier Score, Log Loss, Wilson CI, Expected Calibration Error (ECE), liquidity barriers
├── visualize.py          # Publication-quality Matplotlib figures (Brier curves, calibration dots, liquidity sensitivity)
├── main.py               # Unified CLI orchestrator running end-to-end stages and synthetic benchmarks
├── test_pipeline.py      # Unit and econometric test suite verifying mathematical correctness
├── requirements.txt      # Project dependencies (requests, pandas, numpy, scipy, statsmodels, matplotlib, pyarrow)
├── data/                 # Data directory containing metadata and master panel
│   ├── markets_metadata.parquet
│   ├── markets_metadata.db
│   └── panel_df.parquet
└── figures/              # Generated publication figures
    ├── brier_logloss_curves.png
    ├── calibration_dot_plot.png
    └── liquidity_barrier_sensitivity.png
```

---

## 3. Mathematical Foundations

### Dynamic Cross-Sectional Brier Score
For each countdown day $t \in [-365, -1]$ leading up to Election Day:
$$\text{BS}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} (p_{i,t} - y_i)^2$$
Benchmark reference: an uninformative coin-flip prediction ($p = 0.50$) achieves $\text{BS} = 0.25$.

### Log Loss (Binary Cross-Entropy)
Probabilities are clipped to $[\epsilon, 1-\epsilon]$ with $\epsilon = 0.005$:
$$\text{LL}_t = -\frac{1}{N_t} \sum_{i=1}^{N_t} \left[ y_i \ln(p'_{i,t}) + (1 - y_i) \ln(1 - p'_{i,t}) \right]$$
Benchmark reference: an uninformative coin-flip prediction achieves $\text{LL} = \ln 2 \approx 0.6931$.

### Two-Party Favorite-Side Folding
In binary two-party races, implied probabilities are folded onto the favorite side:
$$p_{\text{fav}} = \max(p, 1 - p) \in [0.50, 0.99]$$
$$y_{\text{fav}} = \begin{cases} y & \text{if } p \ge 0.50 \\ 1 - y & \text{if } p < 0.50 \end{cases}$$

### Calibration & 95% Wilson Score Confidence Interval
For bin $k$ with $n_k$ observations and empirical win rate $\hat{y}_k = w_k / n_k$, the 95% Wilson Score interval is computed as:
$$\text{Center} = \frac{\hat{y}_k + \frac{z^2}{2 n_k}}{1 + \frac{z^2}{n_k}}, \quad \text{Half-Width} = \frac{z}{1 + \frac{z^2}{n_k}} \sqrt{\frac{\hat{y}_k(1 - \hat{y}_k)}{n_k} + \frac{z^2}{4 n_k^2}}$$
where $z \approx 1.95996$.

### Expected Calibration Error (ECE)
$$\text{ECE} = \sum_{k=1}^K \frac{n_k}{N} |\hat{y}_k - \bar{p}_k|$$

---

## 4. How to Run

### Run Full Pipeline End-to-End (Live Kalshi API)
```bash
python3 main.py --all
```

### Run Modular Stages Separately
1. **Discover Markets**:
   ```bash
   python3 main.py --discover
   ```
2. **Harvest Candlesticks & Build Panel**:
   ```bash
   python3 main.py --pipeline
   ```
3. **Statistical Scoring & Calibration Evaluation**:
   ```bash
   python3 main.py --evaluate --horizons 7 30 60 --liquidity-barrier 25000
   ```
4. **Render Figures**:
   ```bash
   python3 main.py --visualize
   ```

### Run Econometric Benchmark Validation (Offline Mode)
```bash
python3 main.py --benchmark-mock
```

### Run Unit Tests
```bash
python3 test_pipeline.py
```

---

## 5. Picking Up Tomorrow

To resume tomorrow:
- All core modules are written, modularized, tested, and validated.
- To execute against live Kalshi endpoints, run `python3 main.py --discover && python3 main.py --pipeline` with network access enabled.
- All figures and panel data are safely persisted in `figures/` and `data/`.
