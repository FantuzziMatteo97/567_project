from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.rnn import RNN
import pandas as pd
from util.data_wrangling import create_timeseries_dataset, get_dataset_df

def main():
    # read and process data
    dataset_df = get_dataset_df('data/VOO_processed.csv')
    train_df, val_df = train_test_split(dataset_df, train_size=0.8, shuffle=False)

    # scale `Close`
    close_scaler = MinMaxScaler()
    train_df['Close'] = close_scaler.fit_transform(train_df['Close'].values.reshape(-1, 1))
    val_df['Close'] = close_scaler.transform(val_df['Close'].values.reshape(-1, 1))

    # scale remaining cols
    scaler = MinMaxScaler()
    train_df[train_df.columns.difference(['Close'])] = scaler.fit_transform(train_df[train_df.columns.difference(['Close'])])
    val_df[val_df.columns.difference(['Close'])] = scaler.transform(val_df[val_df.columns.difference(['Close'])])

    # convert dfs to timeseries data
    X_train, y_train = create_timeseries_dataset(train_df, target_col='Close', timesteps=10)
    X_val, y_val = create_timeseries_dataset(val_df, target_col='Close', timesteps=10)

    # define model architecture
    rnn_model = RNN(timesteps=10, input_d=13)
    rnn_model.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=50)

    # evaluate test loss
    print(f'Final training MSE: {rnn_model.evaluate(X_train, y_train)}')
    print(f'Final validation MSE: {rnn_model.evaluate(X_val, y_val)}')

    # graph predictions on train set
    y_pred_train = rnn_model.predict(X_train, close_scaler)
    y_train = close_scaler.inverse_transform(y_train.reshape(-1, 1))
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_train, color = "b", label = "y_pred_train" )
    plt.plot(y_train, color = "g", label = "y_train")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title("Simple RNN model Train Predictions")
    plt.legend()
    plt.show()

    # graph predictions on val set
    y_pred_val = rnn_model.predict(X_val, close_scaler)
    y_val = close_scaler.inverse_transform(y_val.reshape(-1, 1))
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_val, color = "b", label = "y_pred_val" )
    plt.plot(y_val, color = "g", label = "y_val")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title("Simple RNN model Validation Predictions")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()