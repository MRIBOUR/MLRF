import numpy as np
import pickle
import logging
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

class SVCModel:
    def __init__(self, kernel='rbf', c=1.0, model=None):
        """
        Initializes the Support Vector Classifier (SVC) model.

        Parameters:
        - kernel (str): Specifies the kernel type to be used in the algorithm (e.g., 'linear', 'poly', 'rbf').
        - c (float): Regularization parameter; the strength of the regularization is inversely proportional to C.
        - model (SVC): An existing fitted SVC model instance. If provided, the model will use this instance instead of creating a new one.
        """
        self._SVC = model if model else SVC(C=c, kernel=kernel)
        self._fitted = model is not None  # Tracks if the model has been fitted

    @staticmethod
    def load_model(path='models/SVC_model.pkl'):
        """
        Loads an SVC model from a file.

        Parameters:
        - path (str): The file path from which to load the model.

        Returns:
        - SVCModel: The loaded SVC model.
        """
        try:
            with open(path, 'rb') as file:
                loaded_model = pickle.load(file)
            return SVCModel(model=loaded_model)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not load file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while loading the model: {e}')

    def save_model(self, path='models/SVC_model.pkl'):
        """
        Saves the SVC model to a file.

        Parameters:
        - path (str): The file path to save the model to.
        """
        try:
            with open(path, 'wb') as file:
                pickle.dump(self._SVC, file)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not save to file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while saving the model: {e}')

    def fit(self, X, y):
        """
        Fits the SVC model to the data.

        Parameters:
        - X (array-like): Training data.
        - y (array-like): Target values.
        """
        if self._fitted:
            logger.warning('Model is already fitted')
            return
        self._SVC.fit(X, y)
        self._fitted = True  # Update the fitted status

    def predict(self, X):
        """
        Predicts using the SVC model.

        Parameters:
        - X (array-like): Data to predict on.

        Returns:
        - array-like: Predicted values.
        """
        if not self._fitted:
            raise Exception('The model has not been fitted and can not predict yet.')
        return self._SVC.predict(X)

    def test_accuracy(self, X_test, y_test):
        """
        Tests the accuracy of the SVC model on test data.

        Parameters:
        - X_test (array-like): Test data.
        - y_test (array-like): True values for test data.

        Returns:
        - float: The accuracy score of the model.
        """
        if not self._fitted:
            raise Exception('The model has not been fitted and can not predict yet.')
        y_pred = self._SVC.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f'Tested model SVC on test set, accuracy = {round(acc * 100, 2)}%')
        return acc