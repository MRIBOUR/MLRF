from sklearn.cluster import KMeans
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class BoVW:
    def __init__(self, data, kmeans_clusters=10):
        """
        Initializes the Bag of Visual Words (BoVW) class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        - kmeans_clusters (int): The number of clusters for KMeans (visual words).
        """
        self._data = data
        self._sift = cv2.SIFT_create()
        logger.info('Created SIFT')

        self._descriptors = []
        self._get_img_descriptors()
        logger.info('Computed Descriptors')

        self._kmeans_clusters = kmeans_clusters
        self._kmeans = self._train_kmeans(self._kmeans_clusters)
        logger.info('Trained KMeans')

    def _get_img_descriptors(self):
        """
        Computes SIFT descriptors for each image in the dataset and stores them.
        
        This method processes each image to extract its SIFT keypoints and descriptors,
        appending the descriptors to the `_descriptors` list.
        """
        for img in self._data:
            _, descriptors = self._sift.detectAndCompute(img, None)
            
            if descriptors is not None:
                self._descriptors.append(descriptors)
            else:
                self._descriptors.append([])

    def _train_kmeans(self, kmeans_clusters):
        """
        Trains a KMeans model on the collected descriptors to create visual words.
        
        Parameters:
        - kmeans_clusters (int): The number of clusters (visual words) for KMeans.
        
        Returns:
        - KMeans: The trained KMeans model.
        """
        all_descriptors = np.vstack(self._descriptors) if self._descriptors else np.empty((0, 128))
        
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
        kmeans.fit(all_descriptors)
        
        return kmeans

    def get_features(self):
        """
        Converts image descriptors into feature histograms using the trained KMeans model.
        
        This method predicts the closest visual words for each descriptor and 
        constructs a histogram of visual word occurrences for each image.
        
        Returns:
        - np.ndarray: Array of histograms representing features for each image.
        """
        histograms = []
        
        for descriptors in self._descriptors:
            if len(descriptors) == 0:
                hist = np.zeros(self._kmeans_clusters)
            else:
                words = self._kmeans.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(self._kmeans_clusters + 1))
            
            histograms.append(hist)
        
        logger.info('Transformed Features into Histograms')
        return np.array(histograms)