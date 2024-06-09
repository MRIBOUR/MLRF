from dataset import load_cifar10_batch
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from find_cifar import find_cifar

import numpy as np
import cv2

###

def compute_histograms(images, sift, kmeans, num_clusters):
    histograms = []

    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
        else:
            histogram = np.zeros(num_clusters)
        histograms.append(histogram)

    return np.array(histograms)

###

dataset_path = find_cifar()

data_batches = []
labels_batches = []

for i in range(1, 2):
    batch = load_cifar10_batch(dataset_path / f'data_batch_{i}')
    data_batches.append(batch[b'data'])
    labels_batches.append(batch[b'labels'])

X_train = np.concatenate(data_batches)
y_train = np.concatenate(labels_batches)

test_batch = load_cifar10_batch(dataset_path / 'test_batch')
X_test = test_batch[b'data']
y_test = test_batch[b'labels']

X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0, 2, 3, 1)
X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0, 2, 3, 1)

x_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train])
x_test_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_test])


# SIFT
sift = cv2.SIFT_create()

descriptors_list = []

for img in x_train_gray:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        descriptors_list.append(descriptors)

descriptors = np.vstack(descriptors_list)
#

N_CLUSTERS = 100

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans.fit(descriptors)

visual_vocabulary = kmeans.cluster_centers_

train_histograms = compute_histograms(x_train_gray, sift, kmeans, N_CLUSTERS)
test_histograms = compute_histograms(x_test_gray, sift, kmeans, N_CLUSTERS)

svm = SVC(kernel='linear', random_state=42)
svm.fit(train_histograms, y_train)

y_pred = svm.predict(test_histograms)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')