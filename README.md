# GROUP56_CENG313 - DRW Crypto Market Prediction

CENG313 Introduction to Data Science, Fall 2026 term project.

## Team
| Name | Student ID | GitHub |
| --- | --- | --- |
| Mustafa Karçak | 23118080054 | @Mustafa-karcak |

## Competition
[DRW - Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction):
predict short-term crypto price movements (`label`) from minute-level market data.
Evaluation metric: Pearson correlation between predictions and the true label.

## Dataset
- `train.parquet`: 525,886 rows (one per minute, 2023-03-01 to 2024-02-29) and 786 columns:
  5 market features (bid_qty, ask_qty, buy_qty, sell_qty, volume), 780 anonymous features (X1-X780) and the target `label`.
- `test.parquet`: 538,150 rows, same features.
- The data is not stored in this repository (about 6 GB); the notebook downloads it with the Kaggle API.

## Approach (Assignment 1)
**EDA**
- Data quality: no missing values, no infinite values, no constant columns, no duplicate timestamps.
- Sniff tests: no negative quantities; volume = buy_qty + sell_qty in every row.
- Distributions: label is standardized (mean ≈ 0, std ≈ 1) but very heavy-tailed (kurtosis 16);
  market features are strongly right-skewed (log transform makes them bell-shaped).
- Extremes: the highest-volume minute (2023-10-16 13:31) also has the lowest label (-24.4), a real market event.
- Correlation: all feature-label correlations are small (max 0.09, X752); market features ≈ 0;
  many anonymous features form highly correlated groups (redundant).
- Time patterns: clear daily volume cycle (peak around 14-15h); no long-term trend in label.

**Validation:** time-based split, the last 20% of the training period (no shuffling, to avoid using future information).

**Baselines:** constant mean, linear regression on the 5 market features,
ridge on the 50 features most correlated with label, ridge on all features.

## Results (validation, last 20%)
| Model | Pearson | RMSE |
| --- | --- | --- |
| Mean of train (constant) | undefined | 1.0405 |
| Linear regression, 5 market features | 0.0037 | 1.0406 |
| **Ridge, top-50 features** | **0.0980** | 1.0657 |
| Ridge, all features (1/4 of train rows, RAM limit) | 0.0943 | 1.2287 |

Best baseline: Ridge on the top-50 features. This is the baseline for Assignment 2.

## How to run
1. Open `assignment1_eda_baseline.ipynb` in Google Colab.
2. Upload your `kaggle.json` when asked (Kaggle > Settings > API > Create Legacy API Key).
   You must have joined the competition on Kaggle.
3. Run all cells. The notebook downloads the data, clones this repository and imports the helper code in `src/`:
   - `src/data.py`: loading (float32 to save memory) and cleaning helpers
   - `src/utils.py`: EDA helpers (missing / infinite / constant columns, correlations)
   - `src/models.py`: time-based split, Pearson metric, baseline models

## Kaggle submission history
| Date | Model | Private score | Public score |
| --- | --- | --- | --- |
| 2026-11-08 | Ridge, top-50 features | 0.08866 | 0.05181 |

## Progress log
- 2026-11-06: Created the repository and src structure; Kaggle data download; first look; data quality and sniff tests;
  distribution, extremes and correlation analysis.
- 2026-11-08: Time patterns; baseline models with time-based validation; first Kaggle submission; README.

## Next steps (Assignment 2)
- Feature engineering: log transforms, hour of day, order-flow imbalance (buy - sell) / (buy + sell).
- Reduce redundancy among the anonymous features (PCA or one feature per correlated group).
- Non-linear models (e.g. gradient boosting) compared against the Ridge top-50 baseline.
