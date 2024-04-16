from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dropout, Dense, Input
import numpy as np

class RNN:
    def __init__(self, timesteps=5, input_d=1, optimizer='adam', loss='mean_squared_error'):
        super().__init__()
        self.model = Sequential([
            Input(shape=(timesteps, input_d)),
            SimpleRNN(50, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(100),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        # Define ModelCheckpoint callback to save the model with the best validation loss
        checkpoint_filepath = '.best_rnn.model.keras'
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        # Fit the model with callbacks
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])

    def predict(self, X, scaler):
        return scaler.inverse_transform(self.model.predict(X))
    
    def evaluate(self, X_test, y_test):
        self.model = load_model('.best_rnn.model.keras')
        test_loss = self.model.evaluate(X_test, y_test)
        return test_loss