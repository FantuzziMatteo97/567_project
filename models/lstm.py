from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

from models.base_567_model import Base567Model


class LSTM(Base567Model):
    def __init__(self, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__()
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, data, batch_size=1, epochs=1):
        x_train, y_train = self.preprocess(data)
        self.model.fit(x_train, y_train, batch_size, epochs)

    def test(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        return self.postprocess(predictions, y_test)
