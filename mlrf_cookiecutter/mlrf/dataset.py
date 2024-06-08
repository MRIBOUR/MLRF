import tarfile
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

extracted_path = 'cifar-10-batches-py'


# Function to load a batch
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load all batches
data_batches = []
labels_batches = []
for i in range(1, 6):
    batch = load_cifar10_batch(f"C:\\Users\\User\\OneDrive\\Bureau\\SCIA-G\\MLRF\\MLRF\\mlrf_cookiecutter\\data\\cifar-10-batches-py\\data_batch_{i}")
    data_batches.append(batch[b'data'])
    labels_batches.append(batch[b'labels'])

X_train = np.concatenate(data_batches)
y_train = np.concatenate(labels_batches)

test_batch = load_cifar10_batch("C:\\Users\\User\\OneDrive\\Bureau\\SCIA-G\\MLRF\\MLRF\\mlrf_cookiecutter\\data\\cifar-10-batches-py\\test_batch")
X_test = test_batch[b'data']
y_test = test_batch[b'labels']

# Reshape data to 32x32x3
X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0, 2, 3, 1)
X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0, 2, 3, 1)

X_combined = np.concatenate((X_train, X_test))
y_combined = np.concatenate((y_train, y_test))

cifar10_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Convert to pandas DataFrame
df = pd.DataFrame({
    'image': list(X_combined),
    'label': [cifar10_labels[label] for label in y_combined]
})

# plot images
def plot_images_from_df(df, n=10):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(2, n // 2, i + 1)
        plt.imshow(df['image'].iloc[i])
        plt.title(df['label'].iloc[i])
        plt.axis('off')
    plt.show()

# Display first 10 images from the DataFrame
plot_images_from_df(df)
