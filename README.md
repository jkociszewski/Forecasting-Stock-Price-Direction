<div align="center">

# Forecasting Stock Price Direction

**Can machine learning predict stock direction better than simple baselines?**

Reproducible experiments on Apple, Microsoft, and Alphabet, with chronological validation and an exploratory news sentiment pipeline.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Validation](https://img.shields.io/badge/Validation-Walk--forward-2563EB)
![Status](https://img.shields.io/badge/Status-Research%20prototype-64748B)

[Quick start](#quick-start) · [Results](#results-at-a-glance) · [Methodology](#methodology) · [News pilot](#news-sentiment-pilot) · [Project layout](#project-layout)

</div>

## Overview

This project investigates short-term stock direction for **AAPL**, **MSFT**, and **GOOGL**. It compares Logistic Regression, Random Forest, SVM, XGBoost, and a soft-voting ensemble against majority-class, always-up, and random baselines.

The research focuses on three questions:

- Do historical price and volume features contain useful predictive information?
- Does combining classifiers improve on individual models and simple baselines?
- Does adding headline activity and sentiment improve predictions on the same test sessions?

Two labeling schemes address different prediction tasks:

| Task | Prediction target | Labels |
| :--- | :--- | :--- |
| **Binary direction** | Whether the next session closes above the current close | `0` down or unchanged · `1` up |
| **Triple barrier (TBM)** | Which volatility-based barrier is hit first within up to five sessions, using closing prices | `-1` loss · `0` timeout · `1` profit |

> **Research status:** Current experiments do not establish a reliable advantage over simple baselines. These are classification experiments; trading performance requires a separate execution and transaction-cost model.

## Results at a glance

### Price-only experiments · 2016–2023

Pooled binary accuracy across the annual walk-forward test windows:

| Stock | Soft-voting ensemble | AlwaysUp baseline |
| :--- | ---: | ---: |
| AAPL | 51.69% | **53.43%** |
| MSFT | 53.08% | **53.93%** |
| GOOGL | 52.14% | **53.13%** |

The ensemble did not outperform AlwaysUp by accuracy in these experiments. Per-year results, additional metrics, and uncertainty estimates are described in the [validation report](docs/VALIDATION.md) and saved in [validation_summary.csv](reports/validation_summary.csv).

### News sentiment pilot · GOOGL, 2020

The pilot compares the same ensemble with and without news features on **253 test sessions**:

| Task | Prices | Prices + news |
| :--- | ---: | ---: |
| Binary direction | 51.78% | 52.57% |
| Triple barrier | 49.01% | 47.43% |

Paired confidence intervals do not confirm an improvement from adding news. The pilot covers one stock and one test year; binary and TBM accuracy describe different tasks and should be assessed separately.

See the [news data audit](docs/NEWS.md) and [complete pilot results](reports/news_validation_summary.csv).

## Quick start

**Requirements:** Python 3.11+ and a terminal opened in the repository root. The default experiment runs offline using the included price snapshots.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m stock_direction --ticker AAPL
```

The default run compares Logistic Regression, majority-class, AlwaysUp, and random baselines. Training starts in **2010**; annual tests run from **2016 through 2023**. Reports are written to a new directory under `reports/runs/`.

<details>
<summary><strong>More experiments and installation options</strong></summary>

Install XGBoost and optional hyperparameter tuning:

```bash
python -m pip install -e '.[boosting,tuning]'
```

Compare classifiers, the ensemble, and baselines:

```bash
python -m stock_direction --ticker MSFT \
  --models lr rf svm xgb ensemble majority always_up random
```

Evaluate triple-barrier labels:

```bash
python -m stock_direction --ticker GOOGL --target tbm --horizon 5 \
  --models lr rf majority always_up
```

Select features and tune Logistic Regression within the training windows:

```bash
python -m stock_direction --ticker AAPL --models lr \
  --select-k 10 --tuning-trials 10
```

Inspect the available arguments:

```bash
python -m stock_direction --help
```

For an installation using the resolved dependency versions in `uv.lock`:

```bash
uv sync --locked --extra dev --extra boosting --extra tuning
```

Add the `notebooks` extra to use JupyterLab and the interactive examples:

```bash
python -m pip install -e '.[notebooks]'
jupyter lab
```

</details>

## Methodology

```text
Price snapshots → Causal features → Labels with event end dates
                                             ↓
                        Expanding training window → Next-year test
                                             ↓
                         Predictions, metrics, and uncertainty reports
```

**Features.** Historical returns, volatility, OHLCV relationships, RSI, MACD, distances from moving averages and recent extremes, and causal CUSUM/z-score statistics. Optional news features describe headline activity and VADER sentiment.

**Temporal validation.** Every label records its end date in `Label_End`. Training observations whose labels reach the test boundary are purged. An optional `--gap` removes additional training rows. Scaling, optional feature selection, and hyperparameter tuning are fitted within training windows. SVM probability calibration also uses chronological splits.

**Evaluation.** Models are compared on matching dates and labels using accuracy, balanced accuracy, precision, recall, macro F1, and probability-based ROC-AUC. Binary precision and recall refer to the up class; TBM uses support-weighted precision and recall. AUC is left undefined when a required test class is absent.

**Uncertainty.** Moving-block bootstrap intervals use blocks of 20 observations. Runs including the majority baseline report paired accuracy differences; news experiments also report paired differences between feature sets. These intervals do not correct for selecting among many experiments.

<details>
<summary><strong>Target definitions and important limitations</strong></summary>

- Binary labels retain small and flat moves by default. `--barrier-std 0.5` filters observations using the magnitude of their **future** return. This produces a conditional evaluation, rather than a selection rule available before a trade.
- TBM barriers use trailing volatility from 20 sessions. Barrier hits are evaluated on closes; intraday High/Low hits and overlapping trading positions are not simulated.
- Adjusted historical prices are not a fully reconstructed point-in-time database.
- The archived macroeconomic CSV files lack publication and revision histories, so macro features are excluded from current experiments.
- VADER is the implemented sentiment baseline. FinBERT is not currently used.
- Signals computed after the close cannot automatically be executed at that same closing price. Returns after costs and slippage require a separate backtest.

</details>

## News sentiment pilot

The news pipeline is optional. The inspected public datasets did **not** provide a verified news history for all three stocks over 2010–2023.

For the current pilot, a pinned FNSPID Nasdaq-100 subset yields **4,105 unique Alphabet headlines from 2019–2021** after retaining only Nasdaq article URLs with an encoded date. This date is an availability proxy; the archive does not establish first-seen timestamps or article revision history. No AAPL or MSFT records in that copy meet the same URL-date requirement.

```bash
python -m pip install -e '.[news,news-download]'

python -m stock_direction.news_cli fetch-fnspid
python -m stock_direction.news_cli prepare \
  data/raw/news/fnspid_nasdaq100.csv --require-url-date

python -m stock_direction --ticker GOOGL \
  --models lr rf majority always_up \
  --news-csv data/processed/news/vader_news.csv \
  --first-train-end-year 2019 --last-test-year 2020
```

Passing `--news-csv` automatically evaluates **prices** and **prices_news** on the same sessions. Each article is assigned to its first eligible close after the end of its URL date in New York, plus **24 hours** and **5 minutes** of processing latency. The calendar accounts for daylight saving and early closes.

The importer removes invalid records and duplicate headlines, preserves source metadata and hashes, and rejects test years without any news coverage. Locally downloaded and scored headlines are excluded from Git. Better source data are required to extend the pilot to AAPL and MSFT.

The [news notebook](notebooks/02_news_experiments.ipynb) demonstrates acquisition, coverage inspection, and model comparison. The [source audit](docs/NEWS.md) documents rejected datasets and timing assumptions.

## Reports and reproducibility

Each run produces the following artifacts:

| Artifact | Contents |
| :--- | :--- |
| `comparison.csv` | Pooled metrics and bootstrap intervals for each model and feature set |
| `<model>/folds.csv` | Actual sample counts, boundary dates, and metrics for each test year |
| `<model>/predictions.csv` | Dates, observed labels, predictions, and probabilities named by class |
| `<model>/selected_features.json` | Features used in each training window |
| `<model>/model_parameters.json` | Pipeline parameters, including tuned values, for each window |
| `metadata.json` | Arguments, seeds, package versions, Git state, and input/source hashes |
| `news_coverage.csv` | Observed headline counts per session, when news is enabled |

News runs use directories such as `lr_prices/` and `lr_prices_news/`. An explicit `--output-dir` must point to a new directory. Data acquisition and preparation also refuse to overwrite existing outputs.

Generated runs are ignored by Git. Compact validation summaries and manifests are retained in `reports/`.

## Project layout

```text
Forecasting-Stock-Price-Direction/
├── src/stock_direction/          # Data, features, targets, models, validation, and CLI
├── notebooks/
│   ├── 01_experiments.ipynb     # Price-only experiments
│   ├── 02_news_experiments.ipynb # News acquisition and feature comparison
│   └── legacy/                 # Original notebooks and helpers
├── data/
│   ├── raw/                    # Price snapshots and source datasets
│   ├── processed/news/         # Scored headlines and provenance (Git-ignored)
│   ├── interim/technical/      # Historical indicator tables
│   └── legacy/                 # Original merged feature and label tables
├── reports/
│   ├── runs/                   # Generated experiments (Git-ignored)
│   ├── legacy/                 # Historical results and feature lists
│   └── *.csv / *.json          # Validation summaries and manifests
├── tests/                      # Regression and integration tests
├── docs/                       # Audits, validation notes, and presentation
├── pyproject.toml              # Package configuration and optional dependencies
├── uv.lock                     # Resolved dependency versions
└── .github/workflows/          # Continuous integration
```

Original notebooks and results are preserved in `legacy` directories. They retain the issues described in the audit and their original relative paths. Use the current package and the two top-level notebooks for new experiments.

## Development

Install the dependencies used by the full test suite:

```bash
python -m pip install -e '.[dev,boosting,tuning,news,news-download]'
pytest -q
ruff check .
```

Tests cover causal feature construction, event-date purging, target boundaries, probability alignment, training-only feature selection, news timing and deduplication, bootstrap behavior, and CLI integration. Tests requiring optional dependencies are skipped when those extras are absent. CI installs them and targets Python 3.11 and 3.13.

## Documentation and next steps

| Resource | Description |
| :--- | :--- |
| [Methodology audit](docs/AUDIT.md) | Original issues, corrections, and research limitations *(Polish)* |
| [Price-only validation](docs/VALIDATION.md) | Validation procedure and saved experimental results *(Polish)* |
| [News audit and pilot](docs/NEWS.md) | Source inspection, availability assumptions, and pilot results *(Polish)* |
| [Data notes](data/README.md) | Snapshot provenance and dataset organization *(Polish)* |

The next research priorities are a fixed evaluation protocol, better news coverage with verified availability, a separate comparison of headline counts and sentiment, and an executable backtest with costs and position rules.

## License and data usage

No repository license has been established. External datasets and model resources retain their own usage terms; they are not covered by a license for this project's code.
