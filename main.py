# main.py
import time
from datetime import datetime, timedelta
import logging
from graylog_client import GraylogClient
from preprocessor import EventPreprocessor
from model import AnomalyDetectionModel
from database import EmbeddingDatabase
from alert_manager import EmailAlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyslogAnomalyDetector:
    """Main application class that orchestrates the anomaly detection process"""
    
    def __init__(self, 
                 graylog_api_url, 
                 graylog_api_key, 
                 embedding_dim=64, 
                 num_clusters=5,
                 anomaly_threshold=0.7, 
                 database_path="./embeddings_db",
                 alert_recipients=None):
        """
        Initialize the anomaly detector with configuration parameters
        
        Args:
            graylog_api_url: URL of the Graylog API
            graylog_api_key: API key for Graylog authentication
            embedding_dim: Dimension of event embeddings
            num_clusters: Number of clusters for K-Means
            anomaly_threshold: Threshold for anomaly detection
            database_path: Path to store the embedding database
            alert_recipients: List of email addresses to receive alerts
        """
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize components
        self.graylog_client = GraylogClient(graylog_api_url, graylog_api_key)
        self.preprocessor = EventPreprocessor(embedding_dim)
        self.database = EmbeddingDatabase(database_path)
        self.model = AnomalyDetectionModel(
            num_clusters=num_clusters,
            embedding_dim=embedding_dim,
            database=self.database
        )
        
        # Initialize alert manager with recipients
        if alert_recipients is None:
            alert_recipients = ["admin@datacenter.com"]
        self.alert_manager = EmailAlertManager(recipients=alert_recipients)
        
        # Load existing model if available
        self._load_or_initialize_model()
        
    def _load_or_initialize_model(self):
        """Load existing model or initialize a new one"""
        try:
            self.model.load_model()
            logger.info("Loaded existing model")
        except FileNotFoundError:
            logger.info("No existing model found, initializing new model")
            # Model will be initialized on first run
    
    def fetch_recent_events(self, minutes=10):
        """Fetch events from the last N minutes from Graylog"""
        now = datetime.now()
        from_time = now - timedelta(minutes=minutes)
        logger.info(f"Fetching events from {from_time} to {now}")
        
        events = self.graylog_client.query_syslog(from_time, now)
        logger.info(f"Fetched {len(events)} events")
        return events
    
    def process_events(self, events):
        """Process a batch of events, detect anomalies, and update the model"""
        anomalies_detected = []
        
        for event in events:
            # Preprocess and embed the event
            event_vector = self.preprocessor.preprocess(event)
            
            # Check if the event is anomalous
            if self.model.is_anomalous(event_vector, self.anomaly_threshold):
                logger.warning(f"Anomalous event detected: {event}")
                anomalies_detected.append(event)
            
            # Update the model with the new event
            self.model.update(event_vector, event)
        
        # Return any anomalies found
        return anomalies_detected
    
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
    
    def run_detection_cycle(self):
        """Run a single detection cycle"""
        events = self.fetch_recent_events()
        anomalies = self.process_events(events)
        self.handle_anomalies(anomalies)
        
        # Persist model after processing
        self.model.save_model()
    
    def run(self, interval_seconds=600):
        """Run the anomaly detection in a continuous loop"""
        logger.info(f"Starting anomaly detection with {interval_seconds}s interval")
        
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
            # Ensure model is saved on exit
            self.model.save_model()
            logger.info("Model saved before exit")


if __name__ == "__main__":
    # Example configuration
    detector = SyslogAnomalyDetector(
        graylog_api_url="https://graylog.example.com/api",
        graylog_api_key="your_graylog_api_key",
        embedding_dim=64,
        num_clusters=5,
        anomaly_threshold=0.7,
        alert_recipients=["admin@datacenter.com"]
    )
    
    # Run the detector
    detector.run()
