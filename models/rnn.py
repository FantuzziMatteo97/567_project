from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dropout, Dense, Input
import numpy as np

class RNN:
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__()
        self.scaler = scaler
        self.model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(50, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(100),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        # Define ModelCheckpoint callback to save the model with the best validation loss
        checkpoint_filepath = '.best_rnn.model.keras'
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        # Fit the model with callbacks
        history = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint])

        # Load and save best model
        self.model = load_model('.best_rnn.model.keras')
        return history

    def predict(self, X):
        return self.scaler.inverse_transform(self.model.predict(X))
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse