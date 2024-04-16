from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.rnn import RNN
import pandas as pd
from util.data_wrangling import create_timeseries_dataset, get_dataset_df

def main():
    # read and process data
    dataset_df = get_dataset_df('data/VOO_processed.csv')
    train_df, test_df = train_test_split(dataset_df, train_size=0.8, test_size=0.2, shuffle=False)

    # scale `Close`
    close_scaler = MinMaxScaler()
    train_df['Close'] = close_scaler.fit_transform(train_df['Close'].values.reshape(-1, 1))
    test_df['Close'] = close_scaler.transform(test_df['Close'].values.reshape(-1, 1))

    # scale remaining cols
    scaler = MinMaxScaler()
    train_df[train_df.columns.difference(['Close'])] = scaler.fit_transform(train_df[train_df.columns.difference(['Close'])])
    test_df[test_df.columns.difference(['Close'])] = scaler.transform(test_df[test_df.columns.difference(['Close'])])

    # convert dfs to timeseries data
    X_train, y_train = create_timeseries_dataset(train_df, target_col='Close', timesteps=100)
    X_test, y_test = create_timeseries_dataset(test_df, target_col='Close', timesteps=100)

    # define model architecture
    rnn_model = RNN(timesteps=100, input_d=13)
    rnn_model.train(X_train, y_train, batch_size=32, epochs=50)

    # check predictions on test set
    print('=====Actual=====')
    print(close_scaler.inverse_transform(y_test[0:5].reshape(-1, 1)))
    print('=====Predicted=====')
    print(rnn_model.predict(X_test, close_scaler)[0:5])

    # evaluate test loss
    print(rnn_model.evaluate(X_test, y_test))
    
if __name__ == '__main__':
    main()