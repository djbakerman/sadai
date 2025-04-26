# preprocessor.py
import re
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class EventPreprocessor:
    """Preprocesses syslog events and converts them to embedding vectors"""
    
    def __init__(self, embedding_dim=64, max_features=1000):
        """
        Initialize the event preprocessor
        
        Args:
            embedding_dim: Dimension of the output embedding vectors
            max_features: Maximum number of features for TF-IDF vectorization
        """
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='word',
            token_pattern=r'\b[a-zA-Z0-9_:\.]+\b',  # Include special chars common in logs
            stop_words='english',
            lowercase=True,
            use_idf=True
        )
        
        # Initialize dimensionality reducer (SVD)
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        
        # Track if the vectorizer and SVD have been fitted
        self.is_fitted = False
    
    def _extract_features(self, event):
        """
        Extract textual features from a syslog event
        
        Args:
            event: Syslog event dictionary
            
        Returns:
            String of extracted features
        """
        # Extract the main message
        message = event.get('message', '')
        
        # Extract source
        source = event.get('source', '')
        
        # Extract level
        level = event.get('level', 'INFO')
        
        # Combine features
        features = f"{source} {level} {message}"
        
        # Normalize whitespace
        features = re.sub(r'\s+', ' ', features).strip()
        
        return features
    
    def _normalize_vector(self, vector):
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def preprocess(self, event):
        """
        Preprocess a syslog event into an embedding vector
        
        Args:
            event: Syslog event dictionary
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        # Extract features
        features = self._extract_features(event)
        
        if not features:
            logger.warning("Empty features extracted from event")
            # Return zero vector of correct dimension
            return np.zeros(self.embedding_dim)
        
        # If this is the first event, fit the vectorizer and SVD
        if not self.is_fitted:
            logger.info("First-time fitting of vectorizer and SVD")
            self.vectorizer.fit([features])
            # Generate a dummy vector to fit SVD
            tfidf_vector = self.vectorizer.transform([features])
            # If dimensionality is too low for requested embedding_dim, adjust SVD
            if tfidf_vector.shape[1] < self.embedding_dim:
                adjusted_dim = min(tfidf_vector.shape[1], 32)  # Fallback to smaller dimension
                logger.warning(f"Adjusting embedding dim from {self.embedding_dim} to {adjusted_dim}")
                self.svd = TruncatedSVD(n_components=adjusted_dim, random_state=42)
                self.embedding_dim = adjusted_dim
            self.svd.fit(tfidf_vector)
            self.is_fitted = True
        
        try:
            # Transform features to TF-IDF vector
            tfidf_vector = self.vectorizer.transform([features])
            
            # Reduce dimensionality with SVD
            embedding = self.svd.transform(tfidf_vector)[0]
            
            # Normalize the embedding
            normalized_embedding = self._normalize_vector(embedding)
            
            return normalized_embedding
            
        except ValueError as e:
            logger.error(f"Error preprocessing event: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def partial_fit(self, features_list):
        """
        Update the vectorizer and SVD with new data
        
        Args:
            features_list: List of feature strings extracted from events
        """
        if not features_list:
            return
            
        # Update vectorizer
        self.vectorizer.fit(features_list)
        
        # Generate vectors for SVD update
        tfidf_vectors = self.vectorizer.transform(features_list)
        
        # Update SVD
        self.svd.fit(tfidf_vectors)
        
        self.is_fitted = True
        logger.info("Updated vectorizer and SVD with new data")
