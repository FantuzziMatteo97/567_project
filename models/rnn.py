from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense

class RNN:
    def __init__(self, timesteps, input_d, loss='mean_squared_error'):
        self.model = Sequential()

        self.model.add(SimpleRNN(50, input_shape=(timesteps, input_d), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(SimpleRNN(100))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss=loss)

    def train(self, x_train, y_train, batch_size=64, epochs=50):
        self.model.fit(x_train, y_train, batch_size, epochs)

    def test(self, x_test):
        return self.model.predict(x_test)