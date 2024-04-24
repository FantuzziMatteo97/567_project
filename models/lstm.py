from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

from models.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__(scaler, input_shape, optimizer, loss)

    def build_model(self):
        self.model = Sequential([
            Input(shape=self.input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])