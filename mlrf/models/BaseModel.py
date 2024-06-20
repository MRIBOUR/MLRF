import pickle
import logging
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, model: BaseEstimator = None):
        """
        Base class for models.
        
        Parameters:
        - model (BaseEstimator): An existing fitted model instance.
        """
        self._model = model

    @staticmethod
    @abstractmethod
    def load_model(path: str):
        """
        Loads a model from a file.
        
        Parameters:
        - path (str): The file path from which to load the model.
        
        Returns:
        - BaseModel: The loaded model instance.
        """
        pass

    def save_model(self, path: str):
        """
        Saves the model to a file.
        
        Parameters:
        - path (str): The file path to save the model to.
        """
        try:
            with open(path, 'wb') as file:
                pickle.dump(self._model, file)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not save to file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while saving the model: {e}')

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fits the model to the data.
        
        Parameters:
        - X (array-like): Training data.
        - y (array-like, optional): Target values (required for supervised models).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts using the model.
        
        Parameters:
        - X (array-like): Data to predict on.
        
        Returns:
        - array-like: Predicted values.
        """
        pass

    @abstractmethod
    def test_accuracy(self, X_test, y_test):
        """
        Tests the model's accuracy on test data.
        
        Parameters:
        - X_test (array-like): Test data.
        - y_test (array-like): True values for test data.
        
        Returns:
        - float: The accuracy score of the model.
        """
        pass





