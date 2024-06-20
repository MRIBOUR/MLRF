from mlrf.feature_extraction import BoVWFE
from mlrf.models import SVCModel
import mlrf

import logging
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.NullHandler())

X_train, y_train, X_test, y_test = mlrf.cifar_utils.load_cifar()

bovw = BoVWFE(X_train, kmeans_clusters=100)
X_train_bovw = bovw.get_features()

SVC = SVCModel(kernel='rbf', C=1.0)
best_params = SVC.fit(X_train_bovw, y_train)

bovw_test = BoVWFE(X_test, kmeans_clusters=100)
X_test_bovw = bovw_test.get_features()

acc = SVC.test_accuracy(X_test_bovw, y_test)

print('Parameters for SVC model with BoVW feature extraction:')
print("{'kernel'='rbf', C=1.0}")
print(f'Model accuracy: {round(acc * 100, 2)}%')