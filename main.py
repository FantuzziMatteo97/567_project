from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.rnn import RNN
import pandas as pd
from util.data_wrangling import create_timeseries_dataset, get_dataset_df

def main():
    # read and process data
    dataset_df = get_dataset_df('data/VOO_processed.csv')
    train_df, test_df = train_test_split(dataset_df, train_size=0.8, test_size=0.2, shuffle=False)

    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(train_df['Open'].values.reshape(-1, 1))
    train_df['Open_scaled'] = scaler.transform(train_df['Open'].values.reshape(-1, 1))
    test_df['Open_scaled'] = scaler.transform(test_df['Open'].values.reshape(-1, 1))

    X_train, y_train = create_timeseries_dataset(train_df['Open_scaled'].squeeze().to_frame(), target_col='Open_scaled')
    X_test, y_test = create_timeseries_dataset(test_df['Open_scaled'].squeeze().to_frame(), target_col='Open_scaled')

    # define model architecture
    rnn_model = RNN()
    rnn_model.train(X_train, y_train)

    print('=====Actual=====')
    print(scaler.inverse_transform(y_test[0:5].reshape(-1, 1)))
    print('=====Predicted=====')
    print(rnn_model.predict(X_test, scaler)[0:5])

    print(rnn_model.evaluate(X_test, y_test))
    
if __name__ == '__main__':
    main()