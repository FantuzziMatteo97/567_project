import argparse
from matplotlib import pyplot as plt
from models.decoder_transformer import DecoderTransformer
from models.encoder_transformer import EncoderTransformer
from models.linear import LinearModel
from models.lstm import LSTMModel
from models.rnn import RNNModel
from sklearn.preprocessing import MinMaxScaler
from util.data_wrangling import convert_df_to_inputs_targets, get_dataset_df, train_val_test_split

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

    # convert dfs to input sequences and targets for deep learning sequential models
    X_train, y_decoder_train, y_train = convert_df_to_inputs_targets(train_df, target_col='Close', timesteps=timesteps)
    X_val, y_decoder_val, y_val = convert_df_to_inputs_targets(val_df, target_col='Close', timesteps=timesteps)
    X_test, _, y_test = convert_df_to_inputs_targets(test_df, target_col='Close', timesteps=timesteps)

    # flatten inputs for baseline linear model
    n_train, n_val, n_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]
    d = X_train.shape[2]
    X_train_flattened = X_train.reshape((n_train, timesteps * d))
    X_val_flattened = X_val.reshape((n_val, timesteps * d))
    X_test_flattened = X_test.reshape((n_test, timesteps * d))

    # define model architecture
    if model_type.lower() == 'rnn':
        model = RNNModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'lstm':
        model = LSTMModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'encoder':
        model = EncoderTransformer(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'decoder':
        model = DecoderTransformer(scaler=close_scaler, input_shape=X_train.shape[1:])
    elif model_type.lower() == 'linear':
        model = LinearModel(scaler=close_scaler, input_shape=X_train_flattened.shape[1:])
    else:
        raise ValueError("Invalid model type.")
    
    # train model
    if model_type.lower() == 'decoder':
        history = model.train(X_train, y_decoder_train, X_val, y_decoder_val, batch_size=batch_size, epochs=epochs)
    elif model_type.lower() == 'linear':
        history = model.train(X_train_flattened, y_train, X_val_flattened, y_val, batch_size=batch_size, epochs=epochs)
    else:
        history = model.train(X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs)

    # evaluate train, val, and test MSE
    y_train_unscaled = close_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_val_unscaled = close_scaler.inverse_transform(y_val.reshape(-1, 1))
    y_test_unscaled = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    if model_type.lower() == 'linear':
        print(f'Final training MSE: {model.evaluate(X_train_flattened, y_train_unscaled):.4f}')
        print(f'Final validation MSE: {model.evaluate(X_val_flattened, y_val_unscaled):.4f}')
        print(f'Final test MSE: {model.evaluate(X_test_flattened, y_test_unscaled):.4f}')
    else:
        print(f'Final training MSE: {model.evaluate(X_train, y_train_unscaled):.4f}')
        print(f'Final validation MSE: {model.evaluate(X_val, y_val_unscaled):.4f}')
        print(f'Final test MSE: {model.evaluate(X_test, y_test_unscaled):.4f}')

    # graph predictions on train set
    if model_type.lower() == 'linear':
        y_pred_train = model.predict(X_train_flattened)
    else:
        y_pred_train = model.predict(X_train)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_train, color="b", label="y_pred_train" )
    plt.plot(y_train_unscaled, color="g", label="y_train")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Train")
    plt.legend()
    plt.savefig(f'results/{model_type.lower()}/train_pred_plot')
    plt.show()

    # graph predictions on val set
    if model_type.lower() == 'linear':
        y_pred_val = model.predict(X_val_flattened)
    else:
        y_pred_val = model.predict(X_val)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_val, color="b", label="y_pred_val" )
    plt.plot(y_val_unscaled, color="g", label="y_val")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Validation")
    plt.legend()
    plt.savefig(f'results/{model_type.lower()}/val_pred_plot')
    plt.show()

    # graph predictions on test set
    if model_type.lower() == 'linear':
        y_pred_test = model.predict(X_test_flattened)
    else:
        y_pred_test = model.predict(X_test)
    plt.figure(figsize = (30,10))
    plt.plot(y_pred_test, color="b", label="y_pred_test" )
    plt.plot(y_test_unscaled, color="g", label="y_test")
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"{model_type.upper()} | Actual vs Predicted Close Price | Test")
    plt.legend()
    plt.savefig(f'results/{model_type.lower()}/test_pred_plot')
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate deep learning model for stock price prediction")
    parser.add_argument("model_type", type=str, choices=['rnn', 'lstm', 'encoder', 'decoder', 'linear'], help="Type of model to train")
    parser.add_argument("--timesteps", type=int, default=5, help="Number of timesteps for input sequences (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training (default: 50)")
    args = parser.parse_args()
    main(args.model_type, args.timesteps, args.batch_size, args.epochs)