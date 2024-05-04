from sklearn.naive_bayes import GaussianNB
import numpy as np
from models import base_model


class NaiveBayes(base_model.BaseModel):
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        super().__init__(scaler, input_shape, optimizer, loss)
        self.model = GaussianNB()

    def build_model(self):
        # No need to build anything specific for GaussianNB
        pass


    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
