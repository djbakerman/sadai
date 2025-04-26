# reinforcement_learning.py
import logging
import numpy as np
import json
import os
from datetime import datetime
import random
from collections import deque
import pickle

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    Reinforcement learning class for optimizing the anomaly detection threshold.
    Uses a Q-learning approach to learn the optimal threshold based on feedback.
    """
    
    def __init__(self, 
                 threshold_min=0.3, 
                 threshold_max=0.9, 
                 num_thresholds=13,
                 alpha=0.1,  # Learning rate
                 gamma=0.9,  # Discount factor
                 epsilon=1.0,  # Exploration rate
                 epsilon_min=0.1,
                 epsilon_decay=0.995,
                 model_path="./rl_model"):
        """
        Initialize the threshold optimizer
        
        Args:
            threshold_min: Minimum threshold value
            threshold_max: Maximum threshold value
            num_thresholds: Number of discrete threshold values to consider
            alpha: Learning rate for Q-learning updates
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            model_path: Path to save/load model files
        """
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.num_thresholds = num_thresholds
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model_path = model_path
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        # Generate discrete threshold values
        self.thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)
        
        # Initialize Q-values for each threshold
        self.q_values = np.zeros(num_thresholds)
        
        # Memory of recent experiences for batch learning
        self.memory = deque(maxlen=2000)
        
        # Track current state
        self.current_threshold_idx = None
        self.cumulative_reward = 0.0
        self.episode_count = 0
        
        # Performance metrics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.performance_history = []
    
    def get_next_threshold(self):
        """
        Choose the next threshold to try using epsilon-greedy policy
        
        Returns:
            Selected threshold value
        """
        # Start a new episode if needed
        if self.current_threshold_idx is None:
            self.episode_count += 1
            
            # Epsilon-greedy policy
            if np.random.random() < self.epsilon:
                # Exploration: choose random threshold
                self.current_threshold_idx = np.random.randint(0, self.num_thresholds)
            else:
                # Exploitation: choose best threshold
                self.current_threshold_idx = np.argmax(self.q_values)
                
            logger.info(f"Starting episode {self.episode_count} with threshold index {self.current_threshold_idx} "
                      f"(value: {self.thresholds[self.current_threshold_idx]:.3f})")
        
        # Return the selected threshold
        return self.thresholds[self.current_threshold_idx]
    
    def give_feedback(self, is_anomalous, is_actually_anomalous):
        """
        Provide feedback on a detection event
        
        Args:
            is_anomalous: Boolean indicating if the event was flagged as anomalous
            is_actually_anomalous: Boolean indicating if the event was actually anomalous
        """
        # Update performance metrics
        if is_anomalous and is_actually_anomalous:
            self.true_positives += 1
            reward = 1.0  # Reward for correct detection
        elif is_anomalous and not is_actually_anomalous:
            self.false_positives += 1
            reward = -1.0  # Penalty for false alarm
        elif not is_anomalous and is_actually_anomalous:
            self.false_negatives += 1
            reward = -2.0  # Higher penalty for missing a real anomaly
        else:  # True negative
            reward = 0.1  # Small reward for correctly ignoring normal events
        
        # Update cumulative reward
        self.cumulative_reward += reward
    
    def end_episode(self):
        """
        End the current episode and learn from the experience
        
        Returns:
            Dictionary with episode metrics
        """
        if self.current_threshold_idx is None:
            logger.warning("No episode in progress")
            return None
        
        # Calculate F1 score as a final reward
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Calculate final reward (weighted combination of F1 score and cumulative reward)
        episode_reward = 0.7 * f1_score + 0.3 * (self.cumulative_reward / max(1, self.true_positives + self.false_positives + self.false_negatives))
        
        # Update Q-value for the used threshold
        self.q_values[self.current_threshold_idx] += self.alpha * (episode_reward - self.q_values[self.current_threshold_idx])
        
        # Store episode metrics
        metrics = {
            'episode': self.episode_count,
            'threshold': self.thresholds[self.current_threshold_idx],
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'reward': episode_reward,
            'q_value': self.q_values[self.current_threshold_idx],
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_history.append(metrics)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Log episode results
        logger.info(f"Episode {self.episode_count} completed - Threshold: {self.thresholds[self.current_threshold_idx]:.3f}, "
                  f"F1: {f1_score:.3f}, Reward: {episode_reward:.3f}")
        
        # Reset for next episode
        self.current_threshold_idx = None
        self.cumulative_reward = 0.0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        return metrics
    
    def get_best_threshold(self):
        """
        Get the current best threshold based on learned Q-values
        
        Returns:
            Best threshold value
        """
        best_idx = np.argmax(self.q_values)
        return self.thresholds[best_idx]
    
    def save_model(self):
        """Save model state to disk"""
        try:
            model_data = {
                'q_values': self.q_values,
                'thresholds': self.thresholds,
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
                'performance_history': self.performance_history,
                'hyperparams': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay
                }
            }
            
            with open(f"{self.model_path}/rl_model.pkl", 'wb') as f:
                pickle.dump(model_data, f)
                
            # Save performance history as JSON for easier analysis
            with open(f"{self.model_path}/performance_history.json", 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                
            logger.info(f"RL model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def load_model(self):
        """Load model state from disk"""
        model_path = f"{self.model_path}/rl_model.pkl"
        
        if not os.path.exists(model_path):
            logger.warning(f"No existing RL model found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_values = model_data['q_values']
            self.thresholds = model_data['thresholds']
            self.epsilon = model_data['epsilon']
            self.episode_count = model_data['episode_count']
            self.performance_history = model_data['performance_history']
            
            # Update hyperparameters if available
            if 'hyperparams' in model_data:
                hyperparams = model_data['hyperparams']
                self.alpha = hyperparams.get('alpha', self.alpha)
                self.gamma = hyperparams.get('gamma', self.gamma)
                self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
            
            logger.info(f"RL model loaded from {model_path}")
            logger.info(f"Current best threshold: {self.get_best_threshold():.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            return False


class ThresholdEnvironment:
    """
    Simulated environment for training the threshold optimizer using synthetic data.
    This allows faster learning without waiting for real-world feedback.
    """
    
    def __init__(self, 
                 true_threshold=0.65,
                 noise_level=0.1,
                 num_events=1000,
                 anomaly_ratio=0.05):
        """
        Initialize the simulated environment
        
        Args:
            true_threshold: The "true" optimal threshold value to learn
            noise_level: Noise level in the simulated scores
            num_events: Number of events to generate per episode
            anomaly_ratio: Proportion of events that are anomalous
        """
        self.true_threshold = true_threshold
        self.noise_level = noise_level
        self.num_events = num_events
        self.anomaly_ratio = anomaly_ratio
    
    def generate_events(self):
        """
        Generate synthetic events with anomaly scores
        
        Returns:
            List of (anomaly_score, is_actually_anomalous) tuples
        """
        events = []
        
        # Generate normal events
        normal_count = int(self.num_events * (1 - self.anomaly_ratio))
        for _ in range(normal_count):
            # Normal events have scores below true threshold with some noise
            base_score = np.random.normal(self.true_threshold - 0.2, self.noise_level)
            # Clip to valid range
            score = np.clip(base_score, 0.0, 0.99)
            events.append((score, False))
        
        # Generate anomalous events
        anomaly_count = self.num_events - normal_count
        for _ in range(anomaly_count):
            # Anomalous events have scores above true threshold with some noise
            base_score = np.random.normal(self.true_threshold + 0.2, self.noise_level)
            # Clip to valid range
            score = np.clip(base_score, 0.0, 0.99)
            events.append((score, True))
        
        # Shuffle events
        np.random.shuffle(events)
        
        return events
    
    def simulate_episode(self, optimizer):
        """
        Simulate a full episode with the given optimizer
        
        Args:
            optimizer: ThresholdOptimizer instance
            
        Returns:
            Dictionary with episode metrics
        """
        # Generate synthetic events
        events = self.generate_events()
        
        # Get threshold from optimizer
        threshold = optimizer.get_next_threshold()
        
        # Process events with the threshold
        for score, is_actually_anomalous in events:
            is_anomalous = score > threshold
            optimizer.give_feedback(is_anomalous, is_actually_anomalous)
        
        # End episode and return metrics
        return optimizer.end_episode()
