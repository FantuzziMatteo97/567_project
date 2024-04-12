# 567_project

## Why VOO

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