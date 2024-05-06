from keras.models import Sequential
from keras.layers import Dense, Input
from models.base_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__(scaler, input_shape, optimizer, loss)

    def build_model(self):
        self.model = Sequential([
            Input(shape=self.input_shape),
            Dense(1, activation='linear')
        ])