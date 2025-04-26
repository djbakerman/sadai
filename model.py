# model.py
import logging
import numpy as np
import pickle
import os
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime

logger = logging.getLogger(__name__)

class InfoNCEModel:
    """
    Implementation of InfoNCE contrastive learning for anomaly detection
    """
    
    def __init__(self, embedding_dim=64, queue_size=1000, temperature=0.1):
        """
        Initialize the InfoNCE model
        
        Args:
            embedding_dim: Dimension of event embeddings
            queue_size: Size of the memory queue for contrastive learning
            temperature: Temperature parameter for softmax calculation
        """
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Initialize memory queue for storing embeddings
        self.memory_queue = np.zeros((queue_size, embedding_dim))
        self.queue_ptr = 0
        self.queue_full = False
    
    def _compute_similarity(self, query, key):
        """Compute cosine similarity between query and key"""
        return np.dot(query, key) / (np.linalg.norm(query) * np.linalg.norm(key) + 1e-8)
    
    def train(self, embedding):
        """
        Update the model with a new embedding
        
        Args:
            embedding: New event embedding vector
        """
        # Add the embedding to the memory queue
        self.memory_queue[self.queue_ptr] = embedding
        
        # Update queue pointer
        self.queue_ptr = (self.queue_ptr + 1) % self.queue_size
        
        # Mark queue as full if we've gone through the entire queue once
        if self.queue_ptr == 0:
            self.queue_full = True
    
    def anomaly_score(self, embedding):
        """
        Calculate anomaly score using contrastive learning
        
        Args:
            embedding: Event embedding vector
            
        Returns:
            Anomaly score between 0 and 1 (higher means more anomalous)
        """
        # If the queue is empty, return zero (not anomalous)
        if not self.queue_full and self.queue_ptr == 0:
            return 0.0
        
        # Number of embeddings in the queue
        num_embeddings = self.queue_size if self.queue_full else self.queue_ptr
        
        # Only use the filled portion of the queue
        queue = self.memory_queue[:num_embeddings]
        
        # Calculate similarities between the embedding and all items in the queue
        similarities = np.array([self._compute_similarity(embedding, q) for q in queue])
        
        # Adjust by temperature
        exp_sim = np.exp(similarities / self.temperature)
        
        # Calculate InfoNCE loss (negative of the contrastive loss)
        # Higher loss means the event is more dissimilar to existing patterns
        if len(exp_sim) > 0:
            mean_sim = np.mean(exp_sim)
            
            # Convert to anomaly score (0 to 1)
            # Lower similarity means higher anomaly score
            anomaly_score = 1.0 - (mean_sim / (1.0 + mean_sim))
            
            return anomaly_score
        
        return 0.0  # Default: not anomalous if no comparison available


class AnomalyDetectionModel:
    """Combined anomaly detection model using K-Means clustering and InfoNCE"""
    
    def __init__(self, num_clusters=5, embedding_dim=64, database=None, model_path="./model"):
        """
        Initialize the anomaly detection model
        
        Args:
            num_clusters: Number of clusters for K-Means
            embedding_dim: Dimension of event embeddings
            database: EmbeddingDatabase instance for storing events
            model_path: Path to save/load model files
        """
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.database = database
        self.model_path = model_path
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize K-Means model
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=100,
            random_state=42
        )
        
        # Initialize InfoNCE model
        self.infoNCE = InfoNCEModel(embedding_dim=embedding_dim)
        
        # Track if the model has been trained
        self.is_initialized = False
    
    def _distance_to_nearest_cluster(self, embedding):
        """
        Calculate distance to the nearest cluster center
        
        Args:
            embedding: Event embedding vector
            
        Returns:
            Distance to nearest cluster (normalized between 0 and 1)
        """
        if not self.is_initialized:
            return 0.0  # Return zero (not anomalous) if model is not initialized
        
        # Get cluster centers
        centers = self.kmeans.cluster_centers_
        
        # Calculate distances to all cluster centers
        distances = np.array([np.linalg.norm(embedding - center) for center in centers])
        
        # Get minimum distance
        min_distance = np.min(distances)
        
        # Normalize to [0, 1] using a reasonable scale factor
        max_possible_distance = np.sqrt(self.embedding_dim)  # Max possible distance in embedding space
        normalized_distance = min_distance / max_possible_distance
        
        # Clip to ensure it's between 0 and 1
        normalized_distance = np.clip(normalized_distance, 0.0, 1.0)
        
        return normalized_distance
    
    def is_anomalous(self, embedding, threshold):
        """
        Determine if an event is anomalous based on combined model scores
        
        Args:
            embedding: Event embedding vector
            threshold: Anomaly detection threshold
            
        Returns:
            Boolean indicating if the event is anomalous
        """
        # If model is not initialized, nothing is anomalous
        if not self.is_initialized:
            return False
        
        # Calculate K-Means distance score
        cluster_distance = self._distance_to_nearest_cluster(embedding)
        
        # Calculate InfoNCE anomaly score
        contrastive_score = self.infoNCE.anomaly_score(embedding)
        
        # Combine scores (simple average)
        combined_score = (cluster_distance + contrastive_score) / 2.0
        
        logger.debug(f"Anomaly scores - Cluster: {cluster_distance:.4f}, "
                    f"Contrastive: {contrastive_score:.4f}, "
                    f"Combined: {combined_score:.4f}, "
                    f"Threshold: {threshold:.4f}")
        
        # Return True if combined score exceeds threshold
        return combined_score > threshold
    
    def update(self, embedding, event=None):
        """
        Update the model with a new event embedding
        
        Args:
            embedding: Event embedding vector
            event: Original event data (for database storage)
        """
        # Skip update if embedding is empty or all zeros
        if embedding is None or (isinstance(embedding, np.ndarray) and np.all(embedding == 0)):
            return
        
        # Reshape for sklearn API
        reshaped_embedding = embedding.reshape(1, -1)
        
        # If this is the first event, initialize the model
        if not self.is_initialized:
            # For the first event, we'll initialize the K-Means centers
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                batch_size=100,
                random_state=42,
                init=np.vstack([embedding] + [
                    embedding + 0.1 * np.random.randn(self.embedding_dim) 
                    for _ in range(self.num_clusters - 1)
                ])
            )
            self.kmeans.partial_fit(reshaped_embedding)
            self.is_initialized = True
            logger.info("Initialized K-Means model with first event")
        else:
            # Update K-Means with partial_fit
            self.kmeans.partial_fit(reshaped_embedding)
        
        # Update InfoNCE model
        self.infoNCE.train(embedding)
        
        # Store in database if provided
        if self.database is not None and event is not None:
            timestamp = event.get('timestamp', datetime.now().isoformat())
            event_id = event.get('id', str(datetime.now().timestamp()))
            self.database.store(event_id, embedding, timestamp, event)
    
    def save_model(self):
        """Save model to disk"""
        try:
            # Save K-Means model
            with open(f"{self.model_path}/kmeans.pkl", 'wb') as f:
                pickle.dump(self.kmeans, f)
            
            # Save InfoNCE model
            with open(f"{self.model_path}/infonce.pkl", 'wb') as f:
                pickle.dump(self.infoNCE, f)
            
            # Save initialization state
            with open(f"{self.model_path}/state.pkl", 'wb') as f:
                pickle.dump({'is_initialized': self.is_initialized}, f)
                
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model from disk"""
        # Check if model files exist
        kmeans_path = f"{self.model_path}/kmeans.pkl"
        infonce_path = f"{self.model_path}/infonce.pkl"
        state_path = f"{self.model_path}/state.pkl"
        
        if not all(os.path.exists(p) for p in [kmeans_path, infonce_path, state_path]):
            raise FileNotFoundError("Model files not found")
        
        try:
            # Load K-Means model
            with open(kmeans_path, 'rb') as f:
                self.kmeans = pickle.load(f)
            
            # Load InfoNCE model
            with open(infonce_path, 'rb') as f:
                self.infoNCE = pickle.load(f)
            
            # Load initialization state
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.is_initialized = state.get('is_initialized', False)
                
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
