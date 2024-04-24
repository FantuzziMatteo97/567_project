from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from models.rnn import RNN
from util.data_wrangling import convert_df_to_sequences, get_dataset_df, train_val_test_split

def main():
    # read and process data
    dataset_df = get_dataset_df('data/VOO_processed.csv')
    train_df, val_df, test_df = train_val_test_split(dataset_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # scale `Close`
    close_scaler = MinMaxScaler()
    train_df['Close'] = close_scaler.fit_transform(train_df['Close'].values.reshape(-1, 1))
    val_df['Close'] = close_scaler.transform(val_df['Close'].values.reshape(-1, 1))
    test_df['Close'] = close_scaler.transform(test_df['Close'].values.reshape(-1, 1))

    # scale remaining cols
    scaler = MinMaxScaler()
    train_df[train_df.columns.difference(['Close'])] = scaler.fit_transform(train_df[train_df.columns.difference(['Close'])])
    val_df[val_df.columns.difference(['Close'])] = scaler.transform(val_df[val_df.columns.difference(['Close'])])
    test_df[test_df.columns.difference(['Close'])] = scaler.transform(test_df[test_df.columns.difference(['Close'])])

    # convert dfs to timeseries data
    timesteps = 50
    X_train, y_train = convert_df_to_sequences(train_df, target_col='Close', timesteps=timesteps)
    X_val, y_val = convert_df_to_sequences(val_df, target_col='Close', timesteps=timesteps)
    X_test, y_test = convert_df_to_sequences(test_df, target_col='Close', timesteps=timesteps)

    # define model architecture
    rnn_model = RNN(scaler=close_scaler, input_shape=X_train.shape[1:])
    history = rnn_model.train(X_train, y_train, X_val, y_val, batch_size=32, epochs=50)

    # evaluate train, val, and test MSE
    y_train = close_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_val = close_scaler.inverse_transform(y_val.reshape(-1, 1))
    y_test = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    print(f'Final training MSE: {rnn_model.evaluate(X_train, y_train):.4f}')
    print(f'Final validation MSE: {rnn_model.evaluate(X_val, y_val):.4f}')
    print(f'Final test MSE: {rnn_model.evaluate(X_test, y_test):.4f}')

    # graph predictions on train set
    y_pred_train = rnn_model.predict(X_train)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_train, color="b", label="y_pred_train" )
    plt.plot(y_train, color="g", label="y_train")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title("RNN | Actual vs Predicted Close Price | Train")
    plt.legend()
    plt.show()

    # graph predictions on val set
    y_pred_val = rnn_model.predict(X_val)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_val, color="b", label="y_pred_val" )
    plt.plot(y_val, color="g", label="y_val")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title("RNN | Actual vs Predicted Close Price | Validation")
    plt.legend()
    plt.show()

    # graph predictions on test set
    y_pred_test = rnn_model.predict(X_test)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_test, color="b", label="y_pred_test" )
    plt.plot(y_test, color="g", label="y_test")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title("RNN | Actual vs Predicted Close Price | Test")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()