from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    def __init__(self, data):
        self._data = data

    @abstractmethod
    def get_features(self):
        pass

    @abstractmethod
    def get_images(self):
        pass