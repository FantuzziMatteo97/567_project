import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataset_df(file_path):
    dataset_df = pd.read_csv(file_path)
    # drop rows with NaN
    dataset_df = dataset_df.dropna()
    # drop unused cols
    dataset_df = dataset_df.drop('Unnamed: 0', axis=1)
    dataset_df = dataset_df.drop('Date', axis=1)
    return dataset_df

def train_val_test_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - (n_train + n_val)

    train_df, temp_df = train_test_split(df, train_size=n_train, random_state=42, shuffle=False)
    val_df, test_df = train_test_split(temp_df, train_size=n_val, random_state=42, shuffle=False)

    assert len(train_df) == n_train
    assert len(val_df) == n_val
    assert len(test_df) == n_test

    return train_df, val_df, test_df

def convert_df_to_inputs_targets(df, target_col, timesteps=5):
    X, y_decoder, y = [], [], []
    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y_decoder.append(df.iloc[i+1:i+timesteps+1][target_col].values)
        y.append(df.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y_decoder), np.array(y)

def convert_to_trend_labels(prices):
    """
        Convert stock prices to binary trend labels:
        1 (up) or 0 (down or unchanged).
        """
    # Calculate differences between consecutive days
    trend_diff = np.diff(prices) > 0

    # Insert 0 for the first day (no previous comparison)
    trend_labels = np.insert(trend_diff.astype(int), 0, 0)

    return trend_labels