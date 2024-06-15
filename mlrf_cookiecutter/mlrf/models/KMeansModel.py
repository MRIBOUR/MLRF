import numpy as np
import pickle
import logging
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class KMeansModel:
    def __init__(self, n_clusters=8, init='k-means++', model=None):
        """
        Initializes the KMeans clustering model.

        Parameters:
        - n_clusters (int): The number of clusters to form and the number of centroids to generate.
        - init (str): Method for initialization ('k-means++', 'random').
        - model (KMeans): An existing fitted KMeans model instance. If provided, the model will use this instance instead of creating a new one.
        """
        self._kmeans = model if model else KMeans(n_clusters=n_clusters, init=init)
        self._fitted = model is not None  # Tracks if the model has been fitted

    @staticmethod
    def load_model(path='models/KMeans_model.pkl'):
        """
        Loads a KMeans model from a file.

        Parameters:
        - path (str): The file path from which to load the model.

        Returns:
        - KMeansModel: The loaded KMeans model.
        """
        try:
            with open(path, 'rb') as file:  # Open the file in 'rb' mode to read binary
                loaded_model = pickle.load(file)
            return KMeansModel(model=loaded_model)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not load file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while loading the model: {e}')

    def save_model(self, path='models/KMeans_model.pkl'):
        """
        Saves the KMeans model to a file.

        Parameters:
        - path (str): The file path to save the model to.
        """
        try:
            with open(path, 'wb') as file:  # Open the file in 'wb' mode to write binary
                pickle.dump(self._kmeans, file)
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not save to file {path}')
        except Exception as e:
            raise Exception(f'An error occurred while saving the model: {e}')

    def fit(self, X):
        """
        Fits the KMeans model to the data.

        Parameters:
        - X (array-like): Training data.
        """
        if self._fitted:
            logger.warning('Model is already fitted')
            return
        self._kmeans.fit(X)
        self._fitted = True  # Update the fitted status

    def predict(self, X):
        """
        Predicts the closest cluster each sample in X belongs to using the KMeans model.

        Parameters:
        - X (array-like): Data to predict on.

        Returns:
        - array-like: Index of the cluster each sample belongs to.
        """
        return self._kmeans.predict(X)

    def test_accuracy(self, X_test, y_test):
        """
        Tests the clustering accuracy of the KMeans model on test data.
        Note: Clustering models do not have labels for accuracy in the traditional sense.
              This method matches clusters to provided labels for comparison.

        Parameters:
        - X_test (array-like): Test data.
        - y_test (array-like): True labels for test data.

        Returns:
        - float: The accuracy score of the model.
        """
        y_pred = self._kmeans.predict(X_test)

        # Clustering does not directly provide labels, so we use a mapping heuristic for evaluation
        # We need to map cluster labels to true labels for evaluation
        from scipy.stats import mode
        labels = np.zeros_like(y_pred)
        for i in range(self._kmeans.n_clusters):
            mask = (y_pred == i)
            if np.any(mask):
                labels[mask] = mode(y_test[mask])[0]

        acc = accuracy_score(y_test, labels)
        logger.info(f'Tested model KMeans on test set, accuracy = {round(acc * 100, 2)}%')
        return acc