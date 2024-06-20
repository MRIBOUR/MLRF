import numpy as np
import cv2
import logging

from .FeatureExtractor import FeatureExtractor

logger = logging.getLogger(__name__)

class FlattenFE(FeatureExtractor):
    def __init__(self, data, grayscale=False):
        """
        Initializes the FlattenFE class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        - grayscale (bool): Whether to convert images to grayscale before flattening.
        """
        super().__init__(data)
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
        
        logger.info('Flattened images into feature vectors')
        return np.array(flattened_images)

    def get_images(self):
        """
        Returns the images.
        
        Returns:
        - np.ndarray: Array of images.
        """
        return np.array(self._data)