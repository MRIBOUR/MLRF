from sklearn.cluster import KMeans
import numpy as np
import cv2
import logging

from .FeatureExtractor import FeatureExtractor

logger = logging.getLogger(__name__)

class BoVWFE(FeatureExtractor):
    def __init__(self, data, kmeans_clusters=10):
        """
        Initializes the Bag of Visual Words (BoVW) class with input data.
        
        Parameters:
        - data (array-like): A list or array of images.
        - kmeans_clusters (int): The number of clusters for KMeans (visual words).
        """
        super().__init__(data)
        self._sift = cv2.SIFT_create()
        logger.info('Created SIFT')

        self._descriptors = []
        self._keypoints = []
        self._get_img_descriptors()
        logger.info('Computed Descriptors')

        self._kmeans_clusters = kmeans_clusters
        self._kmeans = self._train_kmeans(self._kmeans_clusters)
        logger.info('Trained KMeans')

    def _get_img_descriptors(self):
        """
        Computes SIFT descriptors for each image in the dataset and stores them.
        """
        for img in self._data:
            keypoints, descriptors = self._sift.detectAndCompute(img, None)
        
            if descriptors is not None and len(descriptors) > 0:
                self._descriptors.append(descriptors)
                self._keypoints.append(keypoints)
            else:
                self._descriptors.append(np.empty((0, 128)))
                self._keypoints.append([])
                logger.warning(f"No descriptors found for image: {img.shape if img is not None else 'None'}")

    def _train_kmeans(self, kmeans_clusters):
        """
        Trains a KMeans model on the collected descriptors to create visual words.
        
        Parameters:
        - kmeans_clusters (int): The number of clusters (visual words) for KMeans.
        
        Returns:
        - KMeans: The trained KMeans model.
        """
        filtered_descriptors = [desc for desc in self._descriptors if desc.size > 0]

        if not filtered_descriptors:
            raise ValueError("No valid descriptors to train KMeans.")

        all_descriptors = np.vstack(filtered_descriptors)
                
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
        kmeans.fit(all_descriptors)
        
        return kmeans

    def get_features(self):
        """
        Converts image descriptors into feature histograms using the trained KMeans model.
        
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

    def get_images(self):
        """
        Draws the SIFT keypoints on each image and returns the images with keypoints.
        
        Returns:
        - List of images with SIFT keypoints drawn on them.
        """
        images_with_keypoints = []
        
        for img, keypoints in zip(self._data, self._keypoints):
            if keypoints:
                img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                img_with_keypoints = img.copy()
                
            images_with_keypoints.append(img_with_keypoints)
        
        logger.info('Keypoints drawn on images')
        return np.array(images_with_keypoints)