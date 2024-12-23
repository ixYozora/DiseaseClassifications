import joblib
import numpy as np

"""

@author: Iraj Masoudian
"""


class RandomForestPredictor:
    def __init__(self, model_path, scaler_path=None):
        """
        Initialize the predictor class.

        Parameters:
        - model_path: str, path to the saved RandomForest model file.
        - scaler: StandardScaler or None, optional scaler to normalize data. Default is None.
        """
        # Load the saved RandomForest model
        self.model = joblib.load(model_path)

        # Initialize scaler if provided
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def preprocess_data(self, X):
        """
        Preprocess the input data by scaling.

        Parameters:
        - X: pd.DataFrame or np.ndarray, the input data to be preprocessed.

        Returns:
        - np.ndarray, preprocessed data.
        """
        # Apply scaling if a scaler is defined
        if self.scaler:
            X = self.scaler.transform(X)

        return np.array(X)  # Convert to numpy array if it's a DataFrame

    def predict(self, X):
        """
        Make predictions on new data using the pre-trained model.

        Parameters:
        - X: pd.DataFrame or np.ndarray, new data for prediction (must contain only the 18 specific features).

        Returns:
        - np.ndarray, predicted class labels.
        """
        # Preprocess the input data
        X_preprocessed = self.preprocess_data(X)

        # Make predictions using the pre-trained model
        return self.model.predict(X_preprocessed)

    def predict_proba(self, X):
        """
        Predict the probabilities of the classes on new data.

        Parameters:
        - X: pd.DataFrame or np.ndarray, new data for prediction (must contain only the 18 specific features).

        Returns:
        - np.ndarray, predicted probabilities of class labels.
        """
        # Preprocess the input data
        X_preprocessed = self.preprocess_data(X)

        # Predict probabilities using the pre-trained model
        return self.model.predict_proba(X_preprocessed)
