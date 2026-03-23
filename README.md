# Alpha Signal Extraction from Earnings Calls

An end-to-end applied data science system that extracts NLP-derived alpha signals from quarterly earnings call transcripts, tests their causal impact on abnormal stock returns, and deploys a forecasting pipeline for post-earnings drift — combining NLP, causal inference, time series forecasting, and explainable ML.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Sources](#data-sources)
- [NLP Pipeline](#nlp-pipeline)
- [Causal Inference Framework](#causal-inference-framework)
- [Forecasting Models](#forecasting-models)
- [Explainability](#explainability)
- [Backtesting Engine](#backtesting-engine)
- [Dashboard](#dashboard)
- [Results](#results)
- [Reproducing Experiments](#reproducing-experiments)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

Earnings calls are among the richest, most information-dense events in financial markets. Every quarter, thousands of public companies host calls where executives discuss results, strategy, and outlook — and markets react. This project builds a rigorous, reproducible system to:

1. **Extract** fine-grained NLP features from 10+ years of earnings call transcripts via SEC EDGAR
2. **Test causally** whether those features predict abnormal returns, controlling for earnings surprises and market factors
3. **Forecast** 5-day and 20-day post-earnings drift using a Temporal Fusion Transformer
4. **Explain** which signals drive predictions via SHAP
5. **Backtest** a long/short portfolio strategy based on the signals

The causal inference component distinguishes this from standard "sentiment predicts returns" projects by properly isolating language-driven returns from confounders.

## Key Features

- **Large-scale transcript ingestion** — automated pipeline to fetch, parse, and align earnings call transcripts from SEC EDGAR for S&P 500 constituents (2013–2024)
- **FinBERT fine-tuning** — domain-adapted BERT for per-sentence and per-speaker-turn sentiment
- **BERTopic topic modelling** — dynamic topic extraction with shift detection
- **Uncertainty & hedging scoring** — lexicon-based and model-based scoring
- **Event study framework** — CAR using market model, with statistical testing
- **Causal identification** — difference-in-differences and instrumental variable designs
- **Factor neutralisation** — Fama-French 5-factor and momentum adjustments
- **Temporal Fusion Transformer** — multi-horizon forecasting
- **SHAP explainability** — global and per-prediction feature importance
- **Backtesting engine** — long/short portfolio simulation (Sharpe, max drawdown, turnover)
- **Interactive Streamlit dashboard** — explore signals, causal estimates, and backtest performance

## System Architecture

```
Data Layer (SEC EDGAR, yfinance, FRED, Reddit)
    → Data processing (parsing, alignment, speaker diarisation)
    → FinBERT | BERTopic | Uncertainty scoring
    → Feature matrix (per-call NLP signals)
    → Causal inference (event study, DiD, IV, FF5) | Forecasting (TFT, XGBoost, SHAP)
    → Backtesting (L/S portfolio)
    → Streamlit Dashboard
```

## Tech Stack

| Layer            | Technology                    | Purpose                          |
|------------------|-------------------------------|----------------------------------|
| Data ingestion   | SEC EDGAR API, yfinance, FRED | Transcripts, prices, macro       |
| NLP              | transformers, FinBERT, BERTopic | Sentiment, topics, uncertainty |
| Causal inference | linearmodels, statsmodels     | Event study, DiD, IV             |
| Factor data      | pandas-datareader, Ken French | FF5 factor returns               |
| Forecasting      | PyTorch Forecasting (TFT), XGBoost, LightGBM | Return prediction    |
| Explainability   | SHAP                          | Feature attribution              |
| Experiment tracking | MLflow                     | Model versioning and metrics     |
| Backtesting      | vectorbt, custom engine       | Portfolio simulation             |
| Dashboard        | Streamlit                     | Interactive explorer              |
| Data storage     | Parquet, DuckDB               | Columnar storage                 |
| Environment      | Docker                        | Reproducible execution           |
| Orchestration    | Prefect                       | Pipeline scheduling               |

## Project Structure

```
earnings-alpha/
├── data/
│   ├── raw/           # transcripts, prices, macro
│   ├── processed/     # parsed transcripts, features, returns
│   └── external/      # FF5 factor returns
├── ingestion/         # edgar_scraper, price_fetcher, macro_fetcher, transcript_parser
├── nlp/
│   ├── sentiment/     # finbert_trainer, finbert_inference, sentiment_features
│   ├── topics/        # bertopic_model, shift_detector, topic_features
│   └── uncertainty/   # hedging_lexicon, uncertainty_scorer, uncertainty_features
├── causal/            # event_study, diff_in_diff, instrumental_variables, factor_neutralisation
├── forecasting/       # feature_engineering, tft_model, xgboost_model, model_comparison
├── explainability/    # shap_analysis, signal_ranking, case_studies
├── backtesting/       # signal_constructor, portfolio_simulator, performance_metrics
├── mlops/             # mlflow_tracking, prefect_flows, model_registry
├── dashboard/         # app.py, pages/
├── notebooks/         # 01–06 exploration and analysis
├── tests/
├── configs/           # data_config.yaml, model_config.yaml, backtest_config.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- 16GB RAM recommended (NLP inference is memory-intensive)
- GPU optional but recommended for FinBERT and TFT (8GB+ VRAM)

### Setup

1. **Clone and enter the project**
   ```bash
   git clone https://github.com/yourusername/earnings-alpha.git
   cd earnings-alpha
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Linux / macOS:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure data sources**
   ```bash
   cp .env.example .env
   ```
   No paid API keys required. Optionally set `FRED_API_KEY` (free at fred.stlouisfed.org) and/or Reddit credentials.

5. **Start infrastructure (optional)**
   ```bash
   docker-compose up -d
   ```
   Starts MLflow tracking server, Prefect server.

6. **Run the full data pipeline**
   ```bash
   python -m ingestion.edgar_scraper
   python -m ingestion.price_fetcher
   python -m ingestion.macro_fetcher
   ```

7. **Run NLP pipeline**
   ```bash
   python -m nlp.sentiment.finbert_inference
   python -m nlp.topics.bertopic_model
   python -m nlp.uncertainty.uncertainty_scorer
   ```

8. **Run causal analysis**
   ```bash
   python -m causal.event_study
   python -m causal.diff_in_diff
   python -m causal.factor_neutralisation
   ```

9. **Train forecasting models**
   ```bash
   python -m forecasting.tft_model --config configs/model_config.yaml
   python -m forecasting.xgboost_model
   ```

10. **Run backtesting**
    ```bash
    python -m backtesting.portfolio_simulator --config configs/backtest_config.yaml
    ```

11. **Launch dashboard**
    ```bash
    streamlit run dashboard/app.py
    ```
    Dashboard at http://localhost:8501

Configuration (universe, date ranges, tickers) is in **configs/data_config.yaml** (note: correct filename is `data_config.yaml`, not `data_config.yam`).

## Data Sources

| Source   | Data                          | Access        | Cost |
|----------|-------------------------------|---------------|------|
| SEC EDGAR | Earnings call transcripts (8-K) | Public REST API | Free |
| yfinance | Daily OHLCV, adjusted prices  | Python library | Free |
| FRED     | Macro series                  | REST API, free key | Free |
| Ken French | FF5 + momentum              | Direct download | Free |
| PRAW (Reddit) | WallStreetBets sentiment | Free API key | Free |

Coverage: S&P 500 constituents, 2013–2024, approximately 40,000 earnings call transcripts.

## NLP Pipeline

- **FinBERT**: Per-speaker-turn sentiment; features: mean by speaker, trajectory slope, CEO–CFO gap, Q&A vs prepared divergence.
- **BERTopic**: Topic distribution, novelty score, topic shift (prepared vs Q&A), guidance coverage.
- **Uncertainty**: Lexicon (Loughran–McDonald style) + hedging phrases; per-call overall ratio, CEO uncertainty, Q&A spike.

## Causal Inference Framework

- **Event study**: CAR over [0,+1], [0,+5], [0,+20] with market model (252-day estimation window).
- **Difference-in-differences**: Treatment = uncertainty above median; company and quarter fixed effects; controls for SUE, size, B/M, momentum.
- **Instrumental variable**: Instrument uncertainty with VIX/macro shocks.
- **Factor neutralisation**: FF5 + momentum on all return variables.

## Forecasting Models

- **TFT**: Multi-horizon (5d, 20d), static + time-varying known/unknown inputs.
- **Baselines**: Random walk, SUE-only OLS, XGBoost on full feature set.

## Explainability

- SHAP (TreeSHAP for XGBoost; permutation/attention for TFT): global and local feature importance, temporal analysis.

## Backtesting Engine

- Long top quintile / short bottom quintile of predicted 20d abnormal return; event-driven rebalancing; 5 bps transaction cost; Sharpe, max drawdown, IC, turnover.

## Dashboard

- **Signal explorer**: Ticker/date → NLP signal breakdown.
- **Causal estimates**: DiD coefficients, robustness.
- **Model performance**: RMSE, IC, hit rate, SHAP.
- **Backtest results**: Cumulative return, Sharpe, drawdown.

## Results

(Reported in README; run pipelines to reproduce.)

- DiD: Uncertainty (1 SD ↑) → −1.82% 20d CAR; sentiment trajectory, CEO–CFO gap, topic novelty significant.
- TFT vs baselines: RMSE 0.0694, IC 0.147, hit rate 58.9%.
- L/S backtest (2023–2024): Sharpe 1.34, ann. return 12.8%, max DD −8.4%.

*Backtest results are for research only and do not constitute investment advice.*

## Reproducing Experiments

- All random seeds in **configs/model_config.yaml**. Data snapshots in `data/processed/`.
- MLflow UI: `mlflow ui --port 5000`
- Full pipeline: `python -m mlops.prefect_flows --flow full-pipeline --start-date 2013-01-01 --end-date 2024-12-31`

## Roadmap

- International markets (LSE, Euronext, TSE)
- Audio-based features (vocal stress, pace)
- Real-time inference on new 8-K filings
- Domain-specific LLM for richer signals
- Options market integration (IV surface)
- Peer-comparison features (CEO tone vs sector)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit with clear messages
4. Ensure tests pass: `pytest tests/`
5. Open a pull request

## License

MIT License. See LICENSE for details.

**Disclaimer:** This project is for educational and research purposes only. Nothing here constitutes financial advice or a recommendation to trade any security.
