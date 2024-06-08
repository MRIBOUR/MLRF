from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
# CIFAR-10
CIFAR_transform_train = transforms.Compose([transforms.ToTensor()])
CIFAR_transform_test =transforms.Compose([transforms.ToTensor()])
trainset_CIFAR = datasets.CIFAR10(root='./data', train=True, download=True, transform=
CIFAR_transform_train)
testset_CIFAR = datasets.CIFAR10(root='./data', train=False, download=True,
transform=CIFAR_transform_test)
CIFAR_train = DataLoader(trainset_CIFAR, batch_size=32, shuffle=True, num_workers=2)
CIFAR_test = DataLoader(testset_CIFAR, batch_size=32, shuffle=False, num_workers=2)
CIFAR_train_images = []
CIFAR_train_labels = []
for batch in CIFAR_train:
    images, labels = batch
    images_flat = images.view(images.shape[0], -1)
    CIFAR_train_images.append(images_flat.numpy())
    CIFAR_train_labels.append(labels.numpy())
CIFAR_train_images = np.vstack(CIFAR_train_images)
CIFAR_train_labels = np.concatenate(CIFAR_train_labels)
CIFAR_test_images = []
CIFAR_test_labels = []
for batch in CIFAR_test:
    images, labels = batch
    images_flat = images.view(images.shape[0], -1)
    CIFAR_test_images.append(images_flat.numpy())
    CIFAR_test_labels.append(labels.numpy())
CIFAR_test_images = np.vstack(CIFAR_test_images)
CIFAR_test_labels = np.concatenate(CIFAR_test_labels)