from mlrf.feature_extraction import HoGFE
from mlrf.models import SVCModel
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

SVC = SVCModel(kernel='rbf', C=1.0)
best_params = SVC.fit(X_train_hog, y_train)

hog_test = HoGFE(X_test)
X_test_hog = hog_test.get_features()

acc = SVC.test_accuracy(X_test_hog, y_test)

print('Parameters for SVC model with Histogram of Gradients feature extraction:')
print("{'kernel'='rbf', C=1.0}")
print(f'Model accuracy: {round(acc * 100, 2)}%')