import numpy as np
import pickle
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from .BaseModel import BaseModel

logger = logging.getLogger(__name__)

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=10, init='k-means++', algorithm='elkan', model=None):
        """
        Initializes the KMeans clustering model.
        
        Parameters:
        - n_clusters (int): The number of clusters to form.
        - init (str): Method for initialization ('k-means++', 'random').
        - model (KMeans): An existing fitted KMeans model instance.
        """
        super().__init__(model if model else KMeans(n_clusters=n_clusters, init=init, algorithm=algorithm))
        self._fitted = model is not None

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
    
    def grid_search_CV(self, param_grid, X, y):
        """
        Performs a grid search with cross-validation to find the optimal hyperparameters
        for the Logistic Regression model based on the provided parameter grid.

        Parameters:
        - param_grid (dict or list of dicts): Dictionary with parameter names (`str`) as keys 
          and lists of parameter settings to try as values.
        - X (array-like or pandas DataFrame): The input data for training the model.
        - y (array-like or pandas Series, optional): The target values corresponding to the 
          input data X.

        Returns:
        - The best parameters obtained after performing grid search and cross-validation.
        """
        if not isinstance(param_grid, (dict, list)):
            raise ValueError("The 'param_grid' parameter must be a dictionary or a list of dictionaries.")
        
        grid_search = GridSearchCV(self._model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)

        self._model = grid_search.best_estimator_
        self._fitted = True

        logger.info(f'Best parameters found: {grid_search.best_params_}')
        return grid_search.best_params_

    def test_accuracy(self, X_test, y_test):
        y_pred = self._model.predict(X_test)

        labels = np.zeros_like(y_pred)
        for i in range(self._model.n_clusters):
            mask = (y_pred == i)
            if np.any(mask):
                true_labels = y_test[mask]
                labels[mask] = np.bincount(true_labels).argmax()

        acc = accuracy_score(y_test, labels)
        logger.info(f'Tested KMeans model on test set, accuracy = {round(acc * 100, 2)}%')
        return acc