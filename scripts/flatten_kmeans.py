from mlrf.feature_extraction import FlattenFE
from mlrf.models import KMeansModel
import mlrf

import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.NullHandler())

X_train, y_train, X_test, y_test = mlrf.cifar_utils.load_cifar()

flatten = FlattenFE(X_train)
X_train_flatten = flatten.get_features()

KMeans = KMeansModel(algorithm='lloyd', init='k-means++', n_clusters=10)
KMeans.fit(X_train_flatten)

flatten_test = FlattenFE(X_test)
X_test_flatten = flatten_test.get_features()

acc = KMeans.test_accuracy(X_test_flatten, y_test)

print('Best Parameters for Kmeans model with flattened images:')
print("{'algorithm': 'lloyd', 'init': 'k-means++', 'n_clusters': 10}")
print(f'Model accuracy: {round(acc * 100, 2)}%')