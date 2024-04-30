from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np

class BaseModel:
    def __init__(self, scaler, input_shape, optimizer='adam', loss='mean_squared_error'):
        """
        Initialize the base model.

        Args:
            scaler: Scaler object used for inverse scaling of predictions.
            input_shape (tuple): Shape of the input data (excluding batch size).
            optimizer (str): Optimizer to use during model compilation.
            loss (str): Loss function to use during model compilation.
        """
        self.scaler = scaler
        self.model = None
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.loss = loss
        self.build_model()

    def build_model(self):
        """
        Build the model architecture.
        To be implemented by subclasses.
        """
        raise NotImplementedError("build_model method must be implemented by subclass")

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model.

        Args:
            X_train: Training input data.
            y_train: Training target data.
            X_val: Validation input data.
            y_val: Validation target data.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            History: Object containing training history.
        """
        checkpoint_filepath = f'.best_{self.__class__.__name__.lower()}.keras'
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        history = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint])

        self.model = load_model(checkpoint_filepath)
        return history

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        print("X shape: ", X.shape)
        return self.scaler.inverse_transform(self.model.predict(X))
    
    def evaluate(self, X, y):
        """
        Evaluate the trained model on test data.

        Args:
            X: Test input data.
            y: True target values for test data.

        Returns:
            float: Mean squared error (MSE) between true and predicted values.
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse