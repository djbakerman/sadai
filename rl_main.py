# rl_main.py
import logging
import time
from datetime import datetime, timedelta
import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from config import load_configuration
from graylog_client import GraylogClient
from preprocessor import EventPreprocessor
from model import AnomalyDetectionModel
from database import EmbeddingDatabase
from alert_manager import EmailAlertManager
from reinforcement_learning import ThresholdOptimizer, ThresholdEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLAnomalyDetector:
    """
    Anomaly detection system that uses reinforcement learning to 
    adaptively tune the anomaly detection threshold
    """
    
    def __init__(self, 
                 graylog_api_url,
                 graylog_api_key,
                 embedding_dim=64,
                 num_clusters=5,
                 database_path="./embeddings_db",
                 alert_recipients=None,
                 model_path="./model",
                 rl_model_path="./rl_model",
                 feedback_mode="auto"):
        """
        Initialize the RL-based anomaly detector
        
        Args:
            graylog_api_url: URL of the Graylog API
            graylog_api_key: API key for Graylog authentication
            embedding_dim: Dimension of event embeddings
            num_clusters: Number of clusters for K-Means
            database_path: Path to store the embedding database
            alert_recipients: List of email addresses to receive alerts
            model_path: Path to save/load model files
            rl_model_path: Path to save/load RL model files
            feedback_mode: How to collect feedback (auto, manual, or simulated)
        """
        # Initialize components
        self.graylog_client = GraylogClient(graylog_api_url, graylog_api_key)
        self.preprocessor = EventPreprocessor(embedding_dim)
        self.database = EmbeddingDatabase(database_path)
        self.model = AnomalyDetectionModel(
            num_clusters=num_clusters,
            embedding_dim=embedding_dim,
            database=self.database,
            model_path=model_path
        )
        
        # Initialize alert manager with recipients
        if alert_recipients is None:
            alert_recipients = ["admin@datacenter.com"]
        self.alert_manager = EmailAlertManager(recipients=alert_recipients)
        
        # Initialize threshold optimizer
        self.optimizer = ThresholdOptimizer(model_path=rl_model_path)
        
        # Set feedback mode
        self.feedback_mode = feedback_mode
        self.pending_feedback = []
        
        # Load existing models if available
        self._load_or_initialize_models()
    
    def _load_or_initialize_models(self):
        """Load existing ML and RL models or initialize new ones"""
        try:
            # Load anomaly detection model
            self.model.load_model()
            logger.info("Loaded existing anomaly detection model")
        except FileNotFoundError:
            logger.info("No existing anomaly detection model found, will initialize new model")
        
        # Load RL model
        rl_loaded = self.optimizer.load_model()
        if not rl_loaded:
            logger.info("No existing RL model found, will initialize new model")
    
    def fetch_recent_events(self, minutes=10):
        """Fetch events from the last N minutes from Graylog"""
        now = datetime.now()
        from_time = now - timedelta(minutes=minutes)
        logger.info(f"Fetching events from {from_time} to {now}")
        
        events = self.graylog_client.query_syslog(from_time, now)
        logger.info(f"Fetched {len(events)} events")
        return events
    
    def process_events(self, events):
        """
        Process a batch of events, detect anomalies using learned threshold,
        and prepare for feedback collection
        """
        # Get the current best threshold from the optimizer
        threshold = self.optimizer.get_next_threshold()
        logger.info(f"Using anomaly threshold: {threshold:.3f}")
        
        anomalies_detected = []
        self.pending_feedback = []
        
        for event in events:
            # Preprocess and embed the event
            event_vector = self.preprocessor.preprocess(event)
            
            # Check if the event is anomalous
            is_anomalous = self.model.is_anomalous(event_vector, threshold)
            
            # Store for feedback collection
            self.pending_feedback.append({
                'event': event,
                'embedding': event_vector,
                'is_anomalous': is_anomalous,
                'score': self._get_anomaly_score(event_vector)
            })
            
            if is_anomalous:
                logger.warning(f"Anomalous event detected: {event}")
                anomalies_detected.append(event)
            
            # Update the anomaly detection model with the new event
            self.model.update(event_vector, event)
        
        # Return any anomalies found
        return anomalies_detected
    
    def _get_anomaly_score(self, embedding):
        """Get the raw anomaly score for an embedding (for debugging)"""
        if not self.model.is_initialized:
            return 0.0
            
        # Calculate K-Means distance score
        cluster_distance = self.model._distance_to_nearest_cluster(embedding)
        
        # Calculate InfoNCE anomaly score
        contrastive_score = self.model.infoNCE.anomaly_score(embedding)
        
        # Combine scores (simple average)
        combined_score = (cluster_distance + contrastive_score) / 2.0
        
        return combined_score
    
    def handle_anomalies(self, anomalies):
        """Handle detected anomalies by sending alerts"""
        if anomalies:
            logger.warning(f"Sending alert for {len(anomalies)} anomalies")
            self.alert_manager.send_alert(
                subject=f"Anomalous Syslog Events Detected (Last 10 mins)",
                body=self._format_alert_message(anomalies)
            )
    
    def _format_alert_message(self, anomalies):
        """Format anomalies into a readable alert message"""
        message = "The following unusual syslog entries were detected and require immediate attention:\n\n"
        
        for i, event in enumerate(anomalies, 1):
            timestamp = event.get('timestamp', datetime.now().isoformat())
            message += f"{i}. [{timestamp}] {event.get('message', 'No message')}\n"
        
        message += "\nPlease investigate promptly."
        return message
    
    def collect_feedback(self):
        """
        Collect feedback on detected anomalies to train the reinforcement learning model.
        The feedback mechanism depends on the configured feedback mode.
        """
        if not self.pending_feedback:
            logger.info("No pending events for feedback")
            return
            
        logger.info(f"Collecting feedback on {len(self.pending_feedback)} events")
        
        if self.feedback_mode == "auto":
            # In auto mode, assume high anomaly scores are true anomalies
            for item in self.pending_feedback:
                score = item['score']
                # Simplistic heuristic: assume events with very high scores are true anomalies
                is_actually_anomalous = score > 0.8
                
                self.optimizer.give_feedback(item['is_anomalous'], is_actually_anomalous)
                
        elif self.feedback_mode == "manual":
            # In manual mode, ask user for confirmation of anomalies
            anomaly_count = sum(1 for item in self.pending_feedback if item['is_anomalous'])
            
            if anomaly_count > 0:
                logger.info(f"Detected {anomaly_count} potential anomalies")
                
                for i, item in enumerate(self.pending_feedback):
                    if item['is_anomalous']:
                        event = item['event']
                        print(f"\nPotential anomaly {i+1}/{anomaly_count}:")
                        print(f"Message: {event.get('message', 'No message')}")
                        print(f"Source: {event.get('source', 'Unknown')}")
                        print(f"Timestamp: {event.get('timestamp', 'Unknown')}")
                        print(f"Anomaly score: {item['score']:.3f}")
                        
                        # Ask for feedback
                        while True:
                            response = input("Is this actually anomalous? (y/n): ").lower().strip()
                            if response in ('y', 'yes'):
                                is_actually_anomalous = True
                                break
                            elif response in ('n', 'no'):
                                is_actually_anomalous = False
                                break
                            else:
                                print("Please answer 'y' or 'n'")
                        
                        self.optimizer.give_feedback(True, is_actually_anomalous)
                
                # For non-anomalous events, assume they are correctly classified
                for item in self.pending_feedback:
                    if not item['is_anomalous']:
                        self.optimizer.give_feedback(False, False)
                
            else:
                logger.info("No anomalies detected in this batch")
                # Assume all non-anomalies are correctly classified
                for item in self.pending_feedback:
                    self.optimizer.give_feedback(False, False)
                    
        elif self.feedback_mode == "simulated":
            # In simulated mode, use a threshold environment to provide feedback
            # We'll simulate based on raw scores rather than true/false values
            
            # Simplistic simulation: assume events with scores > 0.75 are true anomalies
            simulated_threshold = 0.75
            
            for item in self.pending_feedback:
                score = item['score']
                is_actually_anomalous = score > simulated_threshold
                
                # Add some noise to make it more realistic
                if np.random.random() < 0.05:  # 5% chance of flipping the label
                    is_actually_anomalous = not is_actually_anomalous
                
                self.optimizer.give_feedback(item['is_anomalous'], is_actually_anomalous)
        
        # Clear pending feedback
        self.pending_feedback = []
    
    def run_detection_cycle(self):
        """Run a single detection cycle with RL threshold adaptation"""
        events = self.fetch_recent_events()
        anomalies = self.process_events(events)
        self.handle_anomalies(anomalies)
        self.collect_feedback()
        
        # End episode and update RL model
        metrics = self.optimizer.end_episode()
        if metrics:
            logger.info(f"RL metrics - F1: {metrics['f1_score']:.3f}, "
                      f"Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}")
        
        # Persist models after processing
        self.model.save_model()
        self.optimizer.save_model()
    
    def run(self, interval_seconds=600):
        """Run the anomaly detection in a continuous loop"""
        logger.info(f"Starting RL-based anomaly detection with {interval_seconds}s interval")
        
        try:
            while True:
                self.run_detection_cycle()
                logger.info(f"Sleeping for {interval_seconds} seconds")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Anomaly detection stopped by user")
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}", exc_info=True)
            raise
        finally:
            # Ensure models are saved on exit
            self.model.save_model()
            self.optimizer.save_model()
            logger.info("Models saved before exit")


def train_with_simulation(episodes=100, plot=True):
    """
    Train the threshold optimizer using a simulated environment
    
    Args:
        episodes: Number of episodes to train for
        plot: Whether to plot training progress
    """
    logger.info(f"Training threshold optimizer with {episodes} simulated episodes")
    
    # Create environment and optimizer
    env = ThresholdEnvironment(
        true_threshold=0.65,  # Hidden "true" threshold to discover
        noise_level=0.1,
        num_events=1000,
        anomaly_ratio=0.05
    )
    
    optimizer = ThresholdOptimizer(
        threshold_min=0.3,
        threshold_max=0.9,
        num_thresholds=13,
        epsilon=1.0,  # Start with high exploration
        epsilon_decay=0.98  # Slower decay for thorough exploration
    )
    
    # Load existing model if available
    optimizer.load_model()
    
    # Track metrics
    f1_scores = []
    thresholds = []
    rewards = []
    
    # Run training episodes
    for i in range(episodes):
        metrics = env.simulate_episode(optimizer)
        
        f1_scores.append(metrics['f1_score'])
        thresholds.append(metrics['threshold'])
        rewards.append(metrics['reward'])
        
        # Save model periodically
        if (i + 1) % 10 == 0:
            optimizer.save_model()
            logger.info(f"Episode {i+1}/{episodes} - "
                      f"Current best threshold: {optimizer.get_best_threshold():.3f}, "
                      f"Avg F1 (last 10): {np.mean(f1_scores[-10:]):.3f}")
    
    # Save final model
    optimizer.save_model()
    
    # Print final results
    logger.info(f"Training complete - Best threshold: {optimizer.get_best_threshold():.3f}")
    
    if plot:
        try:
            # Create results directory
            os.makedirs("./results", exist_ok=True)
            
            # Plot F1 scores
            plt.figure(figsize=(10, 6))
            plt.plot(f1_scores)
            plt.title('F1 Score over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.savefig("./results/f1_scores.png")
            
            # Plot thresholds
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds)
            plt.axhline(y=env.true_threshold, color='r', linestyle='--', label='True Threshold')
            plt.title('Selected Threshold over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Threshold')
            plt.legend()
            plt.grid(True)
            plt.savefig("./results/thresholds.png")
            
            # Plot rewards
            plt.figure(figsize=(10, 6))
            plt.plot(rewards)
            plt.title('Rewards over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig("./results/rewards.png")
            
            # Plot Q-values
            plt.figure(figsize=(10, 6))
            plt.bar(optimizer.thresholds, optimizer.q_values)
            plt.axvline(x=env.true_threshold, color='r', linestyle='--', label='True Threshold')
            plt.title('Q-values for Different Thresholds')
            plt.xlabel('Threshold')
            plt.ylabel('Q-value')
            plt.legend()
            plt.grid(True)
            plt.savefig("./results/q_values.png")
            
            logger.info("Training plots saved to ./results/")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-based Syslog Anomaly Detection")
    parser.add_argument("--run", action="store_true", help="Run the RL detector")
    parser.add_argument("--simulate", action="store_true", help="Train with simulation")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for simulation")
    parser.add_argument("--feedback", choices=["auto", "manual", "simulated"], 
                        default="auto", help="Feedback collection mode")
    
    args = parser.parse_args()
    
    if args.simulate:
        train_with_simulation(episodes=args.episodes)
    elif args.run:
        # Load configuration
        config = load_configuration()
        
        # Create and run detector
        detector = RLAnomalyDetector(
            graylog_api_url=config.get('graylog.api_url'),
            graylog_api_key=config.get('graylog.api_key'),
            embedding_dim=config.get('model.embedding_dim'),
            num_clusters=config.get('model.num_clusters'),
            database_path=config.get('database.path'),
            alert_recipients=config.get('alerts.email.recipients'),
            feedback_mode=args.feedback
        )
        
        # Run detector
        interval_minutes = config.get('graylog.query_interval_minutes', 10)
        detector.run(interval_seconds=interval_minutes * 60)
    else:
        parser.print_help()
        sys.exit(1)
