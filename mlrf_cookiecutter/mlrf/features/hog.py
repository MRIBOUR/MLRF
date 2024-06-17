from skimage.feature import hog
from skimage import color
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HoGFE:
    def __init__(self, data):
        """
        Initializes the HoG class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        """
        self._data = data
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
                if len(img.shape) == 3:
                    image = color.rgb2gray(img)
                else:
                    image = img

                features, timage = hog(
                    image,
                    orientations=9,
                    pixels_per_cell=(32, 32),
                    cells_per_block=(2, 2),
                    visualize=True,
                    multichannel=False
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