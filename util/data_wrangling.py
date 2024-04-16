import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_dataset_df(file_path):
    dataset_df = pd.read_csv(file_path)
    # drop rows with NaN
    dataset_df = dataset_df.dropna()
    # drop unused cols
    dataset_df = dataset_df.drop('Unnamed: 0',axis =1)
    return dataset_df

def get_train_val_test_dfs(df, train_ratio=0.8, val_ratio=0.2):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df, temp_df = train_test_split(df, train_size=n_train, random_state=42, shuffle=False)
    val_df, test_df = train_test_split(temp_df, train_size=n_val, random_state=42, shuffle=False)

    assert len(train_df) == n_train
    assert len(val_df) == n_val
    assert len(test_df) == n_test

    return train_df, val_df, test_df

def create_timeseries_dataset(df, target_col, timesteps=5):
    X, y = [], []
    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y.append(df.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y)