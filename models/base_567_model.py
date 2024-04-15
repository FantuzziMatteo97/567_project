from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Base561Model:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def preprocess(self, dataset):
        training_data_len = int(np.ceil(len(dataset) * .95))
        scaled_data = self.scaler.fit_transform(dataset)

        # Create the training data set
        # Create the scaled training data set
        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        return x_train, y_train

    def postprocess(self, predictions, y_test):
        predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        return rmse
