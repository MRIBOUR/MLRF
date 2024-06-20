from mlrf.feature_extraction import FlattenFE
from mlrf.models import SVCModel
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

SVC = SVCModel(kernel='rbf', C=1.0)
best_params = SVC.fit(X_train_flatten, y_train)

flatten_test = FlattenFE(X_test)
X_test_flatten = flatten_test.get_features()

acc = SVC.test_accuracy(X_test_flatten, y_test)

print('Parameters for SVC model with Histogram of Gradients feature extraction:')
print("{'kernel'='rbf', C=1.0}")
print(f'Model accuracy: {round(acc * 100, 2)}%')