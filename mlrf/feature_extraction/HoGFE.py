import numpy as np
import logging
from skimage.feature import hog
from skimage import color

from .FeatureExtractor import FeatureExtractor

logger = logging.getLogger(__name__)

class HoGFE(FeatureExtractor):
    def __init__(self, data):
        """
        Initializes the HoG class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        """
        super().__init__(data)
        self._tdata = []
        self._features = []
        logger.info('Created HoG Class')

    def get_features(self):
        """
        Extracts HoG features from the input images.
        
        Returns:
        - np.ndarray: Array of HoG features for each image.
        """
        self._tdata = []
        self._features = []

        for img in self._data:
            try:
                features, timage = hog(
                    img,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3),
                    block_norm='L2-Hys',
                    visualize=True,
                    transform_sqrt=False,
                    feature_vector=True,
                    channel_axis=2
                )
                
                self._features.append(features)
                self._tdata.append(timage)
                
            except Exception as e:
                logger.error(f'Error processing image: {e}')
                continue

        logger.info('Calculated gradients successfully')
        return np.array(self._features)

    def get_images(self):
        """
        Returns the transformed images showing HoG visualizations.
        
        Returns:
        - np.ndarray: Array of HoG visualized images.
        """
        if not self._tdata:
            logger.info('No gradients found, executing gradient detection')
            self.get_features()

        return np.array(self._tdata)