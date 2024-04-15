from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense, Input
import numpy as np

from models.base_567_model import Base567Model


class RNN(Base567Model):
    def __init__(self, timesteps, input_d, optimizer = 'adam', loss='mean_squared_error'):
        super().__init__()
        self.timesteps = timesteps
        self.model = Sequential([
            Input(shape=(timesteps, input_d)),
            SimpleRNN(50, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(100),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer=optimizer, loss=loss)

    def preprocessing(self, x, y):
        x_seq, y_seq = [], []
        for i in range(len(x) - self.timesteps):
            x_seq.append(x[i:i+self.timesteps])
            y_seq.append(y[i:i+self.timesteps])
        return np.array(x_seq), np.array(y_seq)

    def train(self, x_train, y_train, batch_size=64, epochs=50):
        preprocessed_x, preprocessed_y = self.preprocessing(x_train, y_train)
        self.model.fit(preprocessed_x, preprocessed_y, batch_size, epochs)

    def test(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        return self.postprocess(predictions, y_test)
