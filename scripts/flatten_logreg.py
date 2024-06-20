from mlrf.feature_extraction import FlattenFE
from mlrf.models import LogisticRegressionModel
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

LogReg = LogisticRegressionModel()
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'sag', 'saga']
}
best_params = LogReg.grid_search_CV(param_grid, X_train_flatten, y_train)

flatten_test = FlattenFE(X_test)
X_test_flatten = flatten_test.get_features()

acc = LogReg.test_accuracy(X_test_flatten, y_test)

print('Best Parameters for Logistic Regression model with BoVW feature extraction:')
print(best_params)
print(f'Model accuracy: {round(acc * 100, 2)}%')