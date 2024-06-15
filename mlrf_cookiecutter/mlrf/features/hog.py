from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HoG():
    def __init__(self, data):
        self._data = data
        self._tdata = []
        self._features = []
        logger.info('Created Hog Class')
    
    def get_features(self):
        self._tdata = []
        self._features = []
        for img in self._data:
            if len(img.shape) == 3:
                image = color.rgb2gray(image)

            features, timage = hog(image, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=True, multichannel=False)

            self._features.append(features)
            self._tdata.append(timage)
        logger.info('Calculated gradients successfully')
        return np.array(self._features)
    
    def get_images(self):
        if self._tdata == []:
            logger.info('No gradients found, executing gradient detection')
            self.get_features()
        return self._tdata