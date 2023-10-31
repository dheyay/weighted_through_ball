from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import joblib


class rf_xp_model():
    def __init__(self, max_depth=20, max_features='auto', min_samples_leaf=2, min_samples_split=2, n_estimators=500):
        self.model = RandomForestRegressor(max_depth=max_depth, max_features=max_features, 
                                           min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, 
                                           n_estimators = n_estimators)

    def train_model(self, X, y):
        """
        Trains the RF model.
        
        Parameters:
        - X: Feature DataFrame
        - y: Target DataFrame
        """
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train.drop(columns=['name', 'kickoff_time']), y_train)

        predictions = self.model.predict(X_valid.drop(columns=['name', 'kickoff_time']))
        mse = mean_squared_error(y_valid, predictions)
        mae = mean_absolute_error(y_valid, predictions)

        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')

    def predict(self, X):
        """
        Get model's predictions given input data.
        
        Parameters:
        - X: Input data DataFrame
        
        Returns:
        Numpy array of predictions.
        """
        return self.model.predict(X)

    def save_model(self, file_name):
        """
        Saves the model to a file.
        
        Parameters:
        - file_name: str, path to save the model
        """
        joblib.dump(self.model, file_name)
        print(f'Model saved as {file_name}')
        
    def load_model(self, file_name):
        """
        Loads the model from a file.
        
        Parameters:
        - file_name: str, path to load the model from
        """
        self.model = joblib.load(file_name)
        print(f'Model loaded from {file_name}')

    def add_predictions_to_daata(self, X):
        """
        Generates the predictions and adds it to the Expected points column
            ['xP'] as the predictions
        """
        preds = self.predict(X)
        X['xP'] = preds
        return X 
