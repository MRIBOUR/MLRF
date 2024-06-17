from skimage.feature import hog
from skimage import color
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

class FlattenFE:
    def __init__(self, data, grayscale = False):
        """
        Initializes the HoG class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        """
        self._data = data
        self._grayscale = grayscale
        logger.info('Created Flatten Class')

    def get_features(self):
        """
        Flattens data images.
        
        Returns:
        - np.ndarray: Array of flattened images.
        """
        flattened_images = []
        for img in self._data:
            if self._grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flattened_images.append(img.reshape(-1))
        return np.array(self._features)

    def get_images(self):
        """
        Returns the images.
        
        Returns:
        - np.ndarray: Array of images.
        """
        return np.array(self._data)