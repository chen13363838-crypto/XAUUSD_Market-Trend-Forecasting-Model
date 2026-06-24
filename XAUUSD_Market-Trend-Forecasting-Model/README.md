# jud_model — Market Regime Detection Framework

> A Python framework for detecting market regimes (trend / oscillation / transition) from OHLCV data using rule-based scoring and unsupervised machine learning.

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Overview

**jud_model** classifies the current market state into:

| Regime | Description |
|--------|-------------|
| `TREND` | Strong directional movement (breakout) |
| `OSCILLATION` | Range-bound, sideways movement |
| `FUZZY` | Transition state between trend and oscillation |
| `UNKNOWN` | Insufficient confidence to classify |

Supports **four analysis modes**:

1. **Rule mode** — Multi-factor scoring with fixed thresholds (no training needed)
2. **Learn mode** — Unsupervised ML (GMM/HMM) learns regimes from data
3. **Hybrid mode** — ML first, falls back to rules when confidence is low
4. **Multi-timeframe mode** — Fuses M5/M15/M30 features (21 dimensions)

## Features

- **Zero ta-lib dependency** — All indicators (ATR, ADX, BB, MACD, CMF) in pure NumPy
- **Multiple ML backends** — GaussianMixture (default) and GaussianHMM (optional)
- **Multi-timeframe fusion** — Aligns and merges M5/M15/M30 features
- **Flexible data input** — CSV, DataFrame, list, MT5 terminal, or real-time
- **Incremental learning** — Update models with new data without full retraining
- **History decoding** — Label every historical bar for backtesting
- **Model persistence** — Save/load trained models with joblib

## Quick Start

```bash
git clone https://github.com/yourusername/jud_model.git
cd jud_model
pip install -r requirements.txt
python main.py
```

## Project Structure

```
jud_model/
├── config/settings.py          # Global configuration
├── core/
│   ├── indicators.py           # Technical indicators (pure NumPy)
│   ├── features.py             # 9-dim feature engineering
│   ├── analyzer.py             # Rule-based scoring engine
│   ├── agent.py                # Single-TF Agent (rule/learn/hybrid)
│   ├── multi_tf_features.py    # 21-dim multi-TF feature fusion
│   └── multi_tf_agent.py       # Multi-TF Agent
├── data/feed.py                # Data feeder (CSV/DF/MT5/real-time)
├── models/
│   ├── base_model.py           # Abstract base class
│   ├── gmm_model.py            # GaussianMixture model
│   ├── hmm_model.py            # GaussianHMM model (optional)
│   └── multi_tf_gmm.py         # Multi-TF GMM model
├── utils/logger.py             # Unified logging
├── main.py                     # Single-TF demo
├── main_multi_tf.py            # Multi-TF demo
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
pip install hmmlearn        # For HMM model (needs C++ compiler, Python<=3.12)
pip install MetaTrader5     # For MT5 data feed (Windows only)
```

## Usage Examples

### 1. Rule Mode (No Training)

```python
from data.feed import DataFeeder
from core.agent import JudgeAgent

df = DataFeeder.make_synthetic(n_bars=500, mode="mixed")
agent = JudgeAgent(mode="rule")
result = agent.run(df)
print(f"Regime: {result.regime}, Confidence: {result.confidence:.2f}")
```

### 2. Learn Mode (GMM)

```python
train_df = DataFeeder.make_synthetic(n_bars=2000, mode="mixed")
agent = JudgeAgent(mode="learn")
agent.train(train_df, model_type="gmm", n_states=3)
agent.save_model("saved_models/my_model.pkl")

test_df = DataFeeder.make_synthetic(n_bars=300, mode="trend")
result = agent.run(test_df)
```

### 3. Load Existing Model

```python
agent = JudgeAgent.from_pretrained("saved_models/my_model.pkl")
result = agent.run(df)
```

### 4. Multi-Timeframe

```python
from core.multi_tf_agent import MultiTFAgent

tf_data = {
    "M30": df_m30.rename(columns={"time": "datetime"}),
    "M15": df_m15.rename(columns={"time": "datetime"}),
    "M5":  df_m5.rename(columns={"time": "datetime"}),
}
agent = MultiTFAgent(mode="learn")
agent.train(tf_data, n_states=3)
result = agent.run(tf_data)
```

### 5. CSV Data

```python
feeder = DataFeeder()
df = feeder.feed_from_csv("data/XAUUSD_M15.csv", sep="\t")  # MT5: tab-separated
split = int(len(df) * 0.8)
agent = JudgeAgent(mode="hybrid")
agent.train(df.iloc[:split], model_type="gmm", n_states=2)
result = agent.run(df.iloc[split:])
```

### 6. MT5 Real-time

```python
df = feeder.feed_from_mt5(symbol="XAUUSD", bars=2000)
agent = JudgeAgent(mode="hybrid")
agent.train(df, model_type="gmm", n_states=3)
result = agent.run(df)
```

### 7. History Decoding (Backtest)

```python
labels = agent.decode_history(df)
print(labels.value_counts())
```

### 8. Incremental Learning

```python
new_data = feeder.feed_from_csv("data/latest.csv")
agent.partial_fit(new_data)
agent.save_model("saved_models/my_model.pkl")
```

## Architecture

```
         DataFeeder (CSV/DF/MT5/Real-time)
              │
         JudgeAgent (rule/learn/hybrid)
         │           │              │
  RegimeAnalyzer  ML Model     MultiTFAgent
  (rule scoring)  (GMM/HMM)    (M5+M15+M30)
         │           │              │
         └───────────┴──────────────┘
                     │
              AnalysisResult
   (regime / confidence / score / indicators)
```

### Feature Engineering

**Single TF (9 dims):** log_return, volatility, bb_width, atr_ratio, adx_norm, di_diff, range_position, cmf_norm, macd_hist_norm

**Multi-TF (21 dims):** M30 (9) + M15 aggregated (6) + M5 aggregated (6)

### Rule Engine

Each factor → score ∈ [-1, +1]:
- **ADX**: High → trend, Low → oscillation
- **BB Width**: Wide → trend, Narrow → oscillation
- **ATR Ratio**: Expanding → trend, Contracting → oscillation
- **Range Break**: Breakout → trend, Range-bound → oscillation

Weighted composite + factor agreement → confidence → classification.

### ML State Labeling

States auto-labeled by trend strength:

```
score = 0.35*vol + 0.35*ADX + 0.20*ATR + 0.10*BB
```

Highest → TREND, lowest → OSCILLATION, middle → UNKNOWN/FUZZY.

## Configuration

```python
from config.settings import AgentConfig, IndicatorConfig

config = AgentConfig(
    min_bars=120,
    indicator=IndicatorConfig(
        adx_period=14,
        adx_trend_threshold=25.0,
        adx_osc_threshold=20.0,
    ),
)
agent = JudgeAgent(config=config, mode="hybrid")
```

## Data Input

| Method | Description |
|--------|-------------|
| `feed_from_csv()` | CSV (MT5 tab-separated or standard) |
| `feed_from_dataframe()` | pandas DataFrame |
| `feed_from_list()` | List of dicts |
| `feed_from_mt5()` | MT5 terminal (Windows) |
| `feed_single_bar()` | Real-time bar-by-bar |

Required columns: `open`, `high`, `low`, `close`. Optional: `time`, `volume`.
Column names case-insensitive with alias support.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/new-indicator`)
3. Commit changes (`git commit -m 'Add new indicator'`)
4. Push to branch (`git push origin feature/new-indicator`)
5. Open a Pull Request

Ideas: new indicators, new ML models, improve scoring, add data sources.

## License

[MIT](LICENSE)

## Disclaimer

For **educational and research purposes only**. NOT financial advice. Trading involves substantial risk. Always backtest thoroughly before live trading.
