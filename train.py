import argparse
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from models.naive_bayes import NaiveBayes
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.transformer_test import Transformer
from util.data_wrangling import convert_df_to_sequences, get_dataset_df, train_val_test_split

def main(model_type, timesteps, batch_size, epochs):
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
    X_train, y_train = convert_df_to_sequences(train_df, target_col='Close', timesteps=timesteps)
    X_val, y_val = convert_df_to_sequences(val_df, target_col='Close', timesteps=timesteps)
    X_test, y_test = convert_df_to_sequences(test_df, target_col='Close', timesteps=timesteps)
    print(X_train.shape)

    # define model architecture
    if model_type.lower() == 'rnn':
        model = RNNModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'lstm':
        model = LSTMModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'transformer':
        model = Transformer(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'naive_bayes':
        model = NaiveBayes(scaler=close_scaler, input_shape=X_train.shape[1:])
    else:
        raise ValueError("Invalid model type.")
    
    # train model
    history = model.train(X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs)

    # evaluate train, val, and test MSE
    y_train = close_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_val = close_scaler.inverse_transform(y_val.reshape(-1, 1))
    y_test = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    print(f'Final training MSE: {model.evaluate(X_train, y_train):.4f}')
    print(f'Final validation MSE: {model.evaluate(X_val, y_val):.4f}')
    print(f'Final test MSE: {model.evaluate(X_test, y_test):.4f}')

    # graph predictions on train set
    y_pred_train = model.predict(X_train)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_train, color="b", label="y_pred_train" )
    plt.plot(y_train, color="g", label="y_train")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Train")
    plt.legend()
    plt.show()

    # graph predictions on val set
    y_pred_val = model.predict(X_val)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_val, color="b", label="y_pred_val" )
    plt.plot(y_val, color="g", label="y_val")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Validation")
    plt.legend()
    plt.show()

    # graph predictions on test set
    y_pred_test = model.predict(X_test)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_test, color="b", label="y_pred_test" )
    plt.plot(y_test, color="g", label="y_test")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Test")
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate deep learning model for stock price prediction")
    parser.add_argument("model_type", type=str, choices=['rnn', 'lstm', 'transformer', 'naive_bayes'], help="Type of model to train")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of timesteps for input sequences (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training (default: 50)")
    args = parser.parse_args()
    main(args.model_type, args.timesteps, args.batch_size, args.epochs)