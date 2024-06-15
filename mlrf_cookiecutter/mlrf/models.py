import numpy as np
import pickle
import logging
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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
        self._fitted = model is not None  # Tracks if the model has been fitted

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

class SVCModel(BaseModel):
    def __init__(self, kernel='rbf', C=1.0, model=None):
        """
        Initializes the SVC model.
        
        Parameters:
        - kernel (str): Specifies the kernel type to be used.
        - C (float): Regularization parameter.
        - model (SVC): An existing fitted SVC model instance.
        """
        super().__init__(model if model else SVC(kernel=kernel, C=C))

    @staticmethod
    def load_model(path='models/SVC_model.pkl'):
        try:
            with open(path, 'rb') as file:
                loaded_model = pickle.load(file)
            return SVCModel(model=loaded_model)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not load file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while loading the model: {e}')

    def fit(self, X, y):
        if self._fitted:
            logger.warning('Model is already fitted')
            return
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X):
        return self._model.predict(X)

    def test_accuracy(self, X_test, y_test):
        y_pred = self._model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f'Tested SVC model on test set, accuracy = {round(acc * 100, 2)}%')
        return acc

class LogisticRegressionModel(BaseModel):
    def __init__(self, solver='lbfgs', C=1.0, model=None):
        """
        Initializes the Logistic Regression model.
        
        Parameters:
        - solver (str): The algorithm to use in the optimization problem.
        - C (float): Inverse of regularization strength.
        - model (LogisticRegression): An existing fitted Logistic Regression model instance.
        """
        super().__init__(model if model else LogisticRegression(solver=solver, C=C))

    @staticmethod
    def load_model(path='models/LogisticRegression_model.pkl'):
        try:
            with open(path, 'rb') as file:
                loaded_model = pickle.load(file)
            return LogisticRegressionModel(model=loaded_model)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not load file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while loading the model: {e}')

    def fit(self, X, y):
        if self._fitted:
            logger.warning('Model is already fitted')
            return
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X):
        return self._model.predict(X)

    def test_accuracy(self, X_test, y_test):
        y_pred = self._model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f'Tested Logistic Regression model on test set, accuracy = {round(acc * 100, 2)}%')
        return acc

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=8, init='k-means++', model=None):
        """
        Initializes the KMeans clustering model.
        
        Parameters:
        - n_clusters (int): The number of clusters to form.
        - init (str): Method for initialization ('k-means++', 'random').
        - model (KMeans): An existing fitted KMeans model instance.
        """
        super().__init__(model if model else KMeans(n_clusters=n_clusters, init=init))

    @staticmethod
    def load_model(path='models/KMeans_model.pkl'):
        try:
            with open(path, 'rb') as file:
                loaded_model = pickle.load(file)
            return KMeansModel(model=loaded_model)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not load file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while loading the model: {e}')

    def fit(self, X, y=None):
        if self._fitted:
            logger.warning('Model is already fitted')
            return
        self._model.fit(X)
        self._fitted = True

    def predict(self, X):
        return self._model.predict(X)

    def test_accuracy(self, X_test, y_test):
        y_pred = self._model.predict(X_test)

        # We need to map cluster labels to true labels for evaluation
        labels = np.zeros_like(y_pred)
        for i in range(self._model.n_clusters):
            mask = (y_pred == i)
            if np.any(mask):
                true_labels = y_test[mask]
                labels[mask] = np.bincount(true_labels).argmax()

        acc = accuracy_score(y_test, labels)
        logger.info(f'Tested KMeans model on test set, accuracy = {round(acc * 100, 2)}%')
        return acc