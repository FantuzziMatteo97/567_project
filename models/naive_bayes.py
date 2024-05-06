from sklearn.naive_bayes import GaussianNB
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
