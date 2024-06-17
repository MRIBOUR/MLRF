from pathlib import Path
import pickle
import numpy as np

def find_cifar():
    """
    Searches for the CIFAR-10 dataset folder within the project directory structure.

    The function starts from the current working directory and traverses upwards until it finds 
    a directory named 'mlrf_cookiecutter'. Within this root directory, it searches recursively 
    for a folder named 'cifar-10-batches-py'.

    Returns:
    - Path: The path to the 'cifar-10-batches-py' dataset folder.

    Raises:
    - FileNotFoundError: If the CIFAR-10 dataset folder is not found in the project.
    """
    cwd = Path.cwd()
    while 'mlrf_cookiecutter' in cwd.parts and cwd.parts[-1] != 'mlrf_cookiecutter':
        cwd = cwd.parent
        
    target_folder_name = 'cifar-10-batches-py'
    found_folders = [p for p in cwd.rglob(target_folder_name) if p.is_dir()]

    if not found_folders:
        raise FileNotFoundError(f"'{target_folder_name}' folder not found. Please ensure the CIFAR-10 dataset is present in the project directory.")
    
    return found_folders[0]

def load_cifar(path=None):
    """
    Loads the CIFAR-10 dataset from the specified directory.

    This function loads the CIFAR-10 dataset, which consists of five training batches and one test batch,
    from the specified directory. If no directory is provided, it searches for the dataset using `find_cifar`.

    Parameters:
    - path (Path or None): The path to the CIFAR-10 dataset folder. If None, the function searches for the dataset.

    Returns:
    - tuple: Four numpy arrays containing the training data, training labels, test data, and test labels:
      - X_train (np.ndarray): The training images as a 2D array where each row is a flattened image.
      - y_train (np.ndarray): The training labels.
      - X_test (np.ndarray): The test images as a 2D array where each row is a flattened image.
      - y_test (np.ndarray): The test labels.
    """
    def load_cifar10_batch(file):
        """
        Loads a single CIFAR-10 data batch file.

        Parameters:
        - file (Path): The path to the batch file.

        Returns:
        - dict: A dictionary containing the data and labels from the batch.
        """
        with open(file, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
        return batch_dict
    
    if not path:
        path = find_cifar()

    data_batches = []
    labels_batches = []
    for i in range(1, 6):
        batch = load_cifar10_batch(path / f'data_batch_{i}')
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])

    X_train = np.concatenate(data_batches)
    y_train = np.concatenate(labels_batches)
    
    test_batch = load_cifar10_batch(path / 'test_batch')
    X_test = np.array(test_batch[b'data'])
    y_test = np.array(test_batch[b'labels'])

    X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = y_train.reshape(len(y_train), 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = y_test.reshape(len(y_test), 3, 32, 32).transpose(0, 2, 3, 1)

    return X_train, y_train, X_test, y_test

