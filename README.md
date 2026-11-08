# Mobile Phone Price Estimation from Technical Specifications

ML CLA-3 Case Study (Regression)

## Problem statement
Estimate the price (in Indian rupees) of a mobile phone from its technical specifications such as RAM, storage, battery, screen size and cameras.

## Dataset
- **Name:** Mobile Phone Specifications and Prices (Kaggle, user `pratikgarai`)
  https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices
- **File used:** `dataset/ndtv_data_final.csv` (downloaded from Kaggle)
- 1,359 phones, 21 columns (after removing an extra index column), no missing values. Target column: `price` (Rs.)

## Models used
Linear Regression, Decision Tree, Random Forest (200 trees), Gradient Boosting, plus a baseline that always predicts the mean training price.
Preprocessing: removed the extra index column, renamed columns, dropped `name` and `model`, Yes/No -> 1/0, one-hot encoding for brand and operating system, StandardScaler for numeric columns. 80/20 train-test split (`random_state=42`): 1,087 train, 272 test.

## How to run
```bash
pip install -r requirements.txt
python code/mobile_price_prediction.py
```
Outputs are saved in the `results/` folder.

## Results (test set, 272 phones)
| Model | MAE (Rs.) | RMSE (Rs.) | R2 | 5-fold CV R2 |
|---|---|---|---|---|
| Baseline (always predict mean price) | 8210.39 | 16247.21 | -0.0014 | -0.0095 |
| Linear Regression | 4740.14 | 8760.19 | 0.7089 | 0.6015 |
| Decision Tree | 5187.67 | 11834.25 | 0.4687 | 0.2380 |
| Random Forest | 3595.49 | 9243.91 | 0.6758 | 0.6232 |
| Gradient Boosting | 3751.44 | 9138.64 | 0.6832 | 0.6616 |

Best model by test R2: **Linear Regression**. Random Forest has the lowest MAE. Most important features (Random Forest): resolution_y (~37%), internal_storage, ram, battery_capacity. Graph: `results/actual_vs_predicted.png`.

## Libraries used
pandas, numpy, matplotlib, scikit-learn

## Base paper
P. Bhatnagar et al., "Prediction of Mobile Phone Prices using Machine Learning," ICMLT 2024, doi: 10.1145/3674029.3674031
