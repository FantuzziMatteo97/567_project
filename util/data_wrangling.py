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
    dataset_df = dataset_df.drop('Date', axis=1)
    return dataset_df

def create_timeseries_dataset(df, target_col, timesteps=5):
    X, y = [], []
    for i in range(len(df) - timesteps):
        X.append(df.iloc[i:i+timesteps].values)
        y.append(df.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y)