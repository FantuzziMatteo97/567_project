from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense, Input

from models.base_model import BaseModel

class RNN(BaseModel):
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__(scaler, input_shape, optimizer, loss)

    def build_model(self):
        self.model = Sequential([
            Input(shape=self.input_shape),
            SimpleRNN(50, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(100),
            Dropout(0.2),
            Dense(1)
        ])