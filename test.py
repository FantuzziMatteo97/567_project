import os
import matplotlib.pyplot as plt
import argparse
from util.data_wrangling import convert_df_to_inputs_targets, get_dataset_df, train_val_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from models.linear import LinearModel
from models.lstm import LSTMModel
from models.rnn import RNNModel
from sklearn.preprocessing import MinMaxScaler
from models.decoder_transformer import DecoderTransformer
from models.encoder_transformer import EncoderTransformer


def run_tests(timesteps):
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
    train_df[train_df.columns.difference(['Close'])] = scaler.fit_transform(
        train_df[train_df.columns.difference(['Close'])])
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
    rnn = RNNModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    lstm = LSTMModel(scaler=close_scaler, input_shape=X_train.shape[1:])
    encoder = EncoderTransformer(scaler=close_scaler, input_shape=X_train.shape[1:])
    decoder = DecoderTransformer(scaler=close_scaler, input_shape=X_train.shape[1:])
    linear = LinearModel(scaler=close_scaler, input_shape=X_train_flattened.shape[1:])

    rnn.load_checkpoint()
    lstm.load_checkpoint()
    encoder.load_checkpoint()
    decoder.load_checkpoint()
    linear.load_checkpoint()

    test_results = {}
    test_results['rnn'] = rnn.predict(X_test)
    test_results['lstm'] = lstm.predict(X_test)
    test_results['encoder'] = encoder.predict(X_test)
    test_results['decoder'] = decoder.predict(X_test)
    test_results['linear'] = linear.predict(X_test_flattened)
    test_results['true'] = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    plot_all(test_results, 'Test')

    val_results = {}
    val_results['rnn'] = rnn.predict(X_val)
    val_results['lstm'] = lstm.predict(X_val)
    val_results['encoder'] = encoder.predict(X_val)
    val_results['decoder'] = decoder.predict(X_val)
    val_results['linear'] = linear.predict(X_val_flattened)
    val_results['true'] = close_scaler.inverse_transform(y_val.reshape(-1, 1))
    plot_all(val_results, 'Validation')


def plot_all(results, file_name):
    plt.figure(figsize=(30, 10))
    for name, result in results.items():
        plt.plot(result, label=name)
    plt.xlabel("Days")
    plt.ylabel("Close price")
    plt.title(f"Actual vs Predicted Close Price | {file_name}")
    plt.legend()
    plt.savefig(f'results/all/{file_name}_pred_plot_of_best_models')
    plt.show()


def get_best_model(model_name):
    checkpoint_filepath = f'.best_{model_name}.keras'
    model = load_model(checkpoint_filepath)
    return model


def setup():
    if not os.path.isdir(f'results/all'):
        os.mkdir(f'results/all')


if __name__ == '__main__':
    setup()
    parser = argparse.ArgumentParser(description="Test and evaluate deep learning model for stock price prediction")
    parser.add_argument("--timesteps", type=int, default=5,
                        help="Number of timesteps for input sequences (default: 5) for lstm")
    args = parser.parse_args()
    run_tests(timesteps=args.timesteps)
