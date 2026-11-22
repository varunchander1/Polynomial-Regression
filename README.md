# Bike-Sharing Demand: Linear vs Polynomial Regression

**Predicting daily bike-rental demand from weather, comparing simple, multiple and polynomial regression with full residual diagnostics.**

Part of the *AI & ML for Industry* programme, IIT Delhi (Module 2: Regression).

📄 **[Read the full project report (PDF)](https://github.com/varunchander1/bike-sharing-demand-regression/raw/main/Bike_Sharing_Demand_Regression_Report.pdf)**

---

## Overview

Using the UCI Bike Sharing dataset (731 days of rentals in Washington, D.C.), this project asks how well weather explains daily rental counts (`cnt`), and whether a curved relationship beats adding more linear predictors.

Three models are trained on the same 80/20 train-test split (`random_state = 12345`):

1. **Simple linear regression:** `temp` only.
2. **Multiple linear regression:** `temp`, `atemp`, `hum`, `windspeed`.
3. **Polynomial regression:** `temp` and `temp²`, which fits a quadratic curve.

Each model is checked against the linear-regression assumptions: linearity (pairplots), correlation structure, residual distribution and homoscedasticity.

## Results

Test-set performance:

| Model | R² | MSE |
|---|---:|---:|
| Simple linear (`temp`) | 0.432 | 2,091,294 |
| Multiple linear (`temp`, `atemp`, `hum`, `windspeed`) | 0.443 | 2,053,887 |
| **Polynomial (`temp`, `temp²`)** | **0.522** | **1,761,539** |

**Key findings**
- Temperature alone explains about 43% of the variance in daily rentals.
- Adding humidity, windspeed and apparent temperature helps only slightly (+1 point R²), and the residuals show clear heteroscedasticity: errors grow as demand rises.
- A single squared-temperature term gives the biggest gain: +8 points R² and 14% lower MSE than the multiple linear model, with residuals centred much closer to zero (mean −1.6 vs −70.5). Capturing the curve in the temperature-demand relationship matters more than adding linear features.

## Repository contents

| File | Description |
|---|---|
| `Bike_Sharing_Demand_Regression.ipynb` | Data profiling, EDA, the three models and residual diagnostics |
| `Bike_Sharing_Demand_Regression_Report.pdf` | Written report with figures and discussion |
| `day.csv` | Daily Bike Sharing data (UCI) |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook Bike_Sharing_Demand_Regression.ipynb
```

The notebook reads `day.csv`. Update the `read_csv` path in the first cells to point at this folder.

## Data source

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5W894>. Licensed under CC BY 4.0.

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · seaborn · Matplotlib

---

**Author:** Varun Chander · [GitHub](https://github.com/varunchander1)
