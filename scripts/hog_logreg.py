from mlrf.feature_extraction import HoGFE
from mlrf.models import LogisticRegressionModel
import mlrf

import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.NullHandler())

X_train, y_train, X_test, y_test = mlrf.cifar_utils.load_cifar()

hog = HoGFE(X_train)
X_train_hog = hog.get_features()

LogReg = LogisticRegressionModel(C=10, penalty='l2', solver='saga')
best_params = LogReg.fit(X_train_hog, y_train)

hog_test = HoGFE(X_test)
X_test_hog = hog_test.get_features()

acc = LogReg.test_accuracy(X_test_hog, y_test)

print('Best Parameters for Logistic Regression model with Histogram of Gradients feature extraction:')
print("{'C': 10, 'penalty': 'l2', 'solver': 'saga'}")
print(f'Model accuracy: {round(acc * 100, 2)}%')