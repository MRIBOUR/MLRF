import pickle
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from .BaseModel import BaseModel

logger = logging.getLogger(__name__)

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
        self._fitted = model is not None

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

    def grid_search_CV(self, param_grid, X, y=None):
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

    def predict(self, X):
        return self._model.predict(X)

    def test_accuracy(self, X_test, y_test):
        y_pred = self._model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f'Tested SVC model on test set, accuracy = {round(acc * 100, 2)}%')
        return acc