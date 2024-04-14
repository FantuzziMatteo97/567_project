# 567_project

## Why VOO
We opted to utilize a dataset containing the daily statistics of the VOO symbol, which is a stock designed by Vanguard to mirror the S&P500. This choice allows us to forecast the S&P500 index's value more effectively without having to individually model and predict each constituent company. Given our constraints in terms of time and resources, this dataset empowers us to focus on developing and evaluating multiple models efficiently.
## Data Preprocessing
- Add derived columns:
    - Relative Stock Index (default 2-week window)
    - Moving Average Convergence Divergence
    - Simple Moving Average
        - 5 days (weekly)
        - 20 days (monthly)
        - 60 days (bi-monthly)
        - 120 days (quarterly)

## Model Training
- Input (d = 10 days of historical data):
    - Open
    - High
    - Low
    - Close
    - Adj Close
    - Volume

    - Relative Stock Index (default 2-week window)
    - Moving Average Convergence Divergence
    - Simple Moving Average
        - 5 days (weekly)
        - 20 days (monthly)
        - 60 days (bi-monthly)
        - 120 days (quarterly)

- Output
    - Regression: closing stock price for day *t*
        - RMSE
        - MSE
        - MAE
        - R^2

    - Classification: bear, bull, neutral for day *t*
        - Precision
        - Recall
        - F1-Score
        - Accuracy
        - Confusion Matrix