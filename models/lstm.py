from keras.models import Sequential
from keras.layers import Dense, LSTM


class LSTM:
    def __init__(self, input_shape, loss='mean_squared_error'):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss=loss)

    def train(self, x_train, y_train, batch_size=1, epochs=1):
        self.model.fit(x_train, y_train, batch_size, epochs)

    def test(self, x_test):
        return self.model.predict(x_test)
