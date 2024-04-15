from models.rnn import RNN
import pandas as pd

if __name__ == '__main__':
    datasets = pd.read_csv('data/VOO_processed.csv')

    datasets = datasets.drop('Date', axis=1)
    datasets = datasets.drop('Unnamed: 0', axis=1)
    train_y, train_x = datasets['Close'], datasets

    rnn = RNN(timesteps=5, input_d=15)
    rnn.train(datasets, batch_size=1)
