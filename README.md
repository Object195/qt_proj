# Stock Direction Prediction with Machine Learning

A research-oriented machine learning pipeline for predicting short-horizon stock price direction from historical market data and engineered technical features.

The project implements an end-to-end workflow including data collection, feature engineering, feature selection, model training, rolling-window evaluation, and backtesting. The current experiments primarily focus on **TSLA**, with **SPY** available as an additional market reference.

> **Note:** This project is intended for research and educational purposes. It is not a production trading system and should not be interpreted as financial advice.

## Overview

The goal of this project is to investigate whether information contained in historical price, volume, volatility, momentum, and trend indicators can provide predictive signal for short-term market direction.

The current pipeline formulates the problem as a **three-class classification task**:

- **Down**
- **Neutral**
- **Up**

The prediction horizon and target construction are configurable. The default configuration currently uses a **3-day forecast horizon**.

Two model families are implemented:

- **XGBoost** — the primary model used in the current pipeline and sliding-window experiments.
- **PatchTST** — a Transformer-based time-series model implemented as an alternative sequence-modeling approach.

The project places particular emphasis on **time-aware evaluation** and **feature robustness**, rather than relying only on a single random train/test split.

---

## Pipeline

The main workflow is

```text
Market Data
    ↓
Feature Engineering
    ↓
Feature Processing
    ↓
Feature Selection
    ↓
Train / Test Dataset Construction
    ↓
Model Training
    ↓
Backtesting & Diagnostics
```

### 1. Data Collection

Daily OHLCV data are downloaded using `yfinance`.

Multiple tickers can be included in the data pipeline. The current configuration uses:

```python
'tickers': ['TSLA', 'SPY']
'target_tickers': ['TSLA']
```

The separation between `tickers` and `target_tickers` allows market or benchmark information to be incorporated without necessarily generating prediction targets for every asset.

### 2. Feature Engineering

Features are defined centrally in `config.py` and processed through a configurable feature-engineering framework.

Current feature families include:

- Log returns
- Candlestick body and wick ratios
- EMA bias at multiple time scales
- MACD
- RSI
- Bollinger %B and bandwidth
- Relative volume
- Money Flow Index
- Normalized ATR
- Custom forward-return targets
- Experimental support/resistance features

The feature-processing stage also supports transformations and organization of related features into groups for later analysis.

### 3. Target Construction

The primary target is a volatility-adjusted ternary classification target:

```python
TARGET_COL = 'Target_VATC'
```

Rather than predicting the exact future stock price, the model attempts to classify the subsequent price movement into down, neutral, or up regimes.

A continuous version of the target is also generated for analysis.

### 4. Feature Selection

The repository contains multiple approaches for evaluating and filtering features.

#### Rolling IC / IR analysis

Individual features can be evaluated over rolling historical windows using quantities such as:

- Information Coefficient (IC)
- Information Ratio (IR)
- Stability across different market periods

This is intended to identify features whose predictive relationships persist across time rather than appearing only in one fixed sample.

#### SHAP-based analysis

For trained XGBoost models, SHAP values can be aggregated across sliding-window experiments to examine the contribution and robustness of individual features.

This provides a model-dependent complement to the statistical feature-selection procedure.

### 5. Model Training

#### XGBoost

XGBoost is currently the main model used by the automated pipeline.

Its hyperparameters are configured in `config.py`, including:

```python
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'gamma': 0.1,
    ...
}
```

The training code supports class weighting, early stopping, and multiclass evaluation.

#### PatchTST

An experimental PatchTST classifier is also included using the Hugging Face `transformers` implementation.

Unlike XGBoost, which operates on processed feature vectors, PatchTST receives windows of time-series features and uses patch-based Transformer representations for classification.

PatchTST parameters such as context length, patch size, number of attention heads, number of layers, and dropout are configurable in `config.py`.

---

## Sliding-Window Evaluation

Financial data are non-stationary, so performance on a single historical split can provide a misleading estimate of model quality.

`run_sliding_window.py` implements repeated chronological training and testing over different market periods:

```text
|------ Train ------| Gap |-- Test --|
              |------ Train ------| Gap |-- Test --|
                            ...
```

This allows the project to study:

- Performance stability across market regimes
- Generalization to unseen future periods
- Changes in feature importance over time
- Sensitivity of the model to the selected training period

A separate sliding-window pipeline can incorporate additional feature filtering.

---

## Backtesting and Diagnostics

The backtesting module evaluates model predictions together with the underlying market returns.

In addition to classification performance, the repository contains tools for examining:

- Prediction probabilities
- Performance over time
- Feature importance
- Grouped feature importance
- Return-related statistics
- Model behavior across different historical windows

Interactive visualizations are produced with Plotly where appropriate.

---

## Repository Structure

```text
qt_proj/
│
├── backtest/
│   └── Backtesting and model-evaluation utilities
│
├── feature_extraction/
│   ├── generate_data.py
│   ├── generate_features.py
│   ├── create_model_data.py
│   └── feature_gen_config.py
│
├── feature_selection/
│   ├── feature_filter.py
│   └── feature_filter_shap.py
│
├── features/
│   ├── data_fetcher.py
│   ├── feature_engineer.py
│   └── Technical and custom feature implementations
│
├── training/
│   ├── train_xgboost.py
│   ├── train_patchtst.py
│   ├── xgboost_converter.py
│   └── patchtst_converter.py
│
├── visualization/
│   └── Visualization utilities
│
├── config.py
├── run_pipeline.py
├── run_sliding_window.py
└── run_sliding_window_filter.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Object195/qt_proj.git
cd qt_proj
```

The project uses Python and packages including:

```text
numpy
pandas
yfinance
pandas-ta
scikit-learn
xgboost
shap
plotly
tqdm
torch
transformers
databento
```

A dedicated virtual or Conda environment is recommended.

For example:

```bash
conda create -n qt_proj python=3.11
conda activate qt_proj
```

Then install the required packages for the components you intend to run.

---

## Running the Main Pipeline

The complete XGBoost workflow can be launched with:

```bash
python run_pipeline.py
```

The default script performs:

```text
Data download
→ Feature generation
→ Feature filtering
→ Train/test dataset construction
→ XGBoost training
→ Backtesting
```

Training, testing, and data-fetching date ranges are currently defined near the bottom of `run_pipeline.py`.

Experiment-level parameters, features, targets, and model hyperparameters are mainly controlled through:

```text
config.py
```

---

## Running Sliding-Window Experiments

For time-dependent model evaluation:

```bash
python run_sliding_window.py
```

The script repeatedly constructs chronological training and testing windows and saves results from each iteration.

The corresponding filtering workflow is available through:

```bash
python run_sliding_window_filter.py
```

These experiments are useful for examining whether model performance and feature importance remain stable across different market regimes.

---

## Configuration

Most high-level experimental settings are centralized in `config.py`.

For example:

```python
PIPELINE = {
    'tickers': ['TSLA', 'SPY'],
    'target_tickers': ['TSLA'],
    'interval': '1d',
    'input_window': 25,
    'forecast_horizon': 3,
}
```

`FEATURES` controls which raw and technical indicators are generated, while `XGBOOST_PARAMS`, `PATCHTST_PARAMS`, and `SR_PARAMS` contain model- and feature-specific hyperparameters.

This makes it possible to modify the experimental setup without rewriting the main pipeline.

---

## Experimental Components

Some parts of the repository are exploratory and are not enabled in the default pipeline.

In particular, the project contains experimental code for support/resistance detection using intraday Databento data. The current `generate_data.py` contains a local path for this dataset and disables support/resistance detection by default.

Users wishing to enable this component should replace the local data path with the location of their own intraday dataset.

The codebase should therefore be viewed as an **active research project** rather than a packaged production library.

---

## Motivation

Short-term market prediction is a difficult problem because financial time series are noisy, non-stationary, and highly sensitive to changing market conditions.

This project is therefore not intended to demonstrate that a particular model can reliably "predict the stock market." Instead, it provides an experimental framework for studying questions such as:

- Which engineered market features contain measurable predictive information?
- How stable are those signals across time?
- Does feature filtering improve out-of-sample performance?
- How does a tree-based model such as XGBoost compare with a temporal Transformer model?
- How strongly does model performance depend on the selected training regime?

The broader goal is to combine **time-series analysis, feature engineering, machine learning, and careful out-of-sample evaluation** in a reproducible research workflow.

---

## Disclaimer

This repository is an experimental machine-learning project developed for research and educational purposes only.

Historical backtesting does not guarantee future performance. Transaction costs, slippage, liquidity constraints, market impact, and many other real-world effects may not be fully represented.

Nothing in this repository constitutes investment or financial advice.