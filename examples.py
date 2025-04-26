# examples.py
"""
Examples demonstrating how to use the syslog anomaly detection system
"""

import logging
from datetime import datetime, timedelta
import json
import numpy as np
import argparse
import sys

# Local imports
from config import load_configuration
from graylog_client import GraylogClient
from preprocessor import EventPreprocessor
from model import AnomalyDetectionModel
from database import EmbeddingDatabase
from alert_manager import EmailAlertManager, WebhookAlertManager
from main import SyslogAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_detector():
    """Run the detector with default configuration"""
    config = load_configuration()
    
    detector = SyslogAnomalyDetector(
        graylog_api_url=config.get('graylog.api_url'),
        graylog_api_key=config.get('graylog.api_key'),
        embedding_dim=config.get('model.embedding_dim'),
        num_clusters=config.get('model.num_clusters'),
        anomaly_threshold=config.get('model.anomaly_threshold'),
        database_path=config.get('database.path'),
        alert_recipients=config.get('alerts.email.recipients')
    )
    
    # Run the detector for one cycle or continuously
    run_once = config.get('run_once', False)
    
    if run_once:
        logger.info("Running detector for one cycle")
        detector.run_detection_cycle()
    else:
        interval_minutes = config.get('graylog.query_interval_minutes', 10)
        logger.info(f"Running detector continuously with {interval_minutes} minute interval")
        detector.run(interval_seconds=interval_minutes * 60)


def simulate_learning_process():
    """
    Simulate the learning process from scratch with synthetic data
    to demonstrate how the model learns over time
    """
    logger.info("Simulating learning process with synthetic data")
    
    # Create temporary database and model
    database = EmbeddingDatabase(db_path="./tmp_embeddings_db")
    model = AnomalyDetectionModel(
        num_clusters=5,
        embedding_dim=64,
        database=database
    )
    preprocessor = EventPreprocessor(embedding_dim=64)
    
    # Generate synthetic routine events
    routine_event_templates = [
        "Router1 interface Gi0/1 link state changed to up",
        "Temperature Sensor Rack12: Temperature stable at {}°C",
        "Successful login from admin IP 10.10.20.5",
        "Server backup completed successfully, {} files processed",
        "System health check passed, all systems normal"
    ]
    
    # Generate synthetic anomalous events
    anomalous_event_templates = [
        "Unauthorized SSH login attempt from IP 203.0.113.99",
        "Temperature Sensor Rack3: Critical temperature alert {}°C",
        "Multiple failed login attempts detected from IP 198.51.100.23",
        "Unexpected system shutdown on Server12",
        "Unusual network traffic pattern detected on VLAN 5"
    ]
    
    # Function to generate a random event
    def generate_event(templates, anomalous=False):
        template = np.random.choice(templates)
        
        if "{}" in template:
            if "Temperature" in template:
                if anomalous:
                    value = np.random.randint(85, 95)  # Critical temperatures
                else:
                    value = np.random.randint(20, 26)  # Normal temperatures
                message = template.format(value)
            elif "files" in template:
                value = np.random.randint(1000, 5000)
                message = template.format(value)
            else:
                message = template
        else:
            message = template
        
        return {
            'id': f"event_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'source': 'simulation',
            'level': 'CRITICAL' if anomalous else 'INFO'
        }
    
    # Setup simulation parameters
    total_events = 1000
    anomaly_probability = 0.05  # 5% chance of anomaly
    anomalies_detected = []
    
    logger.info(f"Starting simulation with {total_events} events")
    logger.info(f"Anomaly probability: {anomaly_probability * 100}%")
    
    # Simulate event stream
    for i in range(total_events):
        # Determine if this will be an anomalous event
        is_anomalous = np.random.random() < anomaly_probability
        
        # Generate event
        if is_anomalous:
            event = generate_event(anomalous_event_templates, anomalous=True)
        else:
            event = generate_event(routine_event_templates, anomalous=False)
        
        # Process event
        event_vector = preprocessor.preprocess(event)
        
        # Check if model detects it as anomalous (after first 50 events to let model stabilize)
        if i >= 50:
            detected_anomaly = model.is_anomalous(event_vector, threshold=0.7)
            
            if detected_anomaly:
                logger.info(f"Event {i}: ANOMALY DETECTED: {event['message']}")
                anomalies_detected.append((i, event))
            
            # Check if detection matches ground truth
            if detected_anomaly and is_anomalous:
                logger.info("  ✓ True positive")
            elif detected_anomaly and not is_anomalous:
                logger.info("  ✗ False positive")
            elif not detected_anomaly and is_anomalous:
                logger.info("  ✗ False negative")
        
        # Update model with event
        model.update(event_vector, event)
        
        # Print progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{total_events} events")
    
    # Print results
    logger.info(f"Simulation complete. {len(anomalies_detected)} anomalies detected.")
    
    # Calculate statistics (if we have ground truth)
    true_anomalies = int(total_events * anomaly_probability)
    logger.info(f"Expected anomalies: {true_anomalies}")
    logger.info(f"Detected anomalies: {len(anomalies_detected)}")
    
    # Clean up temporary database
    import shutil
    try:
        shutil.rmtree("./tmp_embeddings_db")
    except:
        pass


def analyze_existing_data():
    """Analyze existing data in the database"""
    config = load_configuration()
    
    # Load database
    db_path = config.get('database.path')
    database = EmbeddingDatabase(db_path=db_path)
    
    # Get event count
    event_count = database.get_event_count()
    logger.info(f"Database contains {event_count} events")
    
    # If no events, exit
    if event_count == 0:
        logger.info("No events to analyze")
        return
    
    # Get all embeddings
    embeddings = database.get_all_embeddings()
    
    # Create model and load if exists
    model = AnomalyDetectionModel(
        num_clusters=config.get('model.num_clusters'),
        embedding_dim=config.get('model.embedding_dim'),
        database=database,
        model_path=config.get('model.path')
    )
    
    try:
        model.load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("No existing model found")
        return
    
    # Get recent events for analysis
    now = datetime.now()
    start_time = (now - timedelta(days=1)).isoformat()
    events = database.get_events_in_timeframe(start_time, now.isoformat())
    
    logger.info(f"Analyzing {len(events)} events from the last 24 hours")
    
    # Count anomalies
    anomaly_threshold = config.get('model.anomaly_threshold')
    anomalies = []
    
    for event_id, timestamp, embedding, event_data in events:
        if model.is_anomalous(embedding, anomaly_threshold):
            anomalies.append((timestamp, event_data))
    
    logger.info(f"Found {len(anomalies)} anomalies in recent data")
    
    # Print sample anomalies
    if anomalies:
        logger.info("Sample anomalies:")
        for i, (timestamp, event_data) in enumerate(anomalies[:5], 1):
            logger.info(f"  {i}. [{timestamp}] {event_data.get('message', 'No message')}")


def test_with_sample_data():
    """Test the system with sample syslog data"""
    logger.info("Testing with sample syslog data")
    
    # Create components
    config = load_configuration()
    database = EmbeddingDatabase(db_path="./tmp_test_db")
    model = AnomalyDetectionModel(
        num_clusters=config.get('model.num_clusters'),
        embedding_dim=config.get('model.embedding_dim'),
        database=database
    )
    preprocessor = EventPreprocessor(embedding_dim=config.get('model.embedding_dim'))
    
    # Sample syslog entries (mixture of normal and anomalous)
    sample_events = [
        # Normal events
        {
            'id': '1',
            'timestamp': datetime.now().isoformat(),
            'message': 'Router1 interface Gi0/1 link state changed to up',
            'source': 'Router1',
            'level': 'INFO'
        },
        {
            'id': '2',
            'timestamp': datetime.now().isoformat(),
            'message': 'Temperature Sensor Rack12: Temperature stable at 24°C',
            'source': 'Sensor',
            'level': 'INFO'
        },
        {
            'id': '3',
            'timestamp': datetime.now().isoformat(),
            'message': 'User admin logged in from 10.10.20.5',
            'source': 'AuthService',
            'level': 'INFO'
        },
        {
            'id': '4',
            'timestamp': datetime.now().isoformat(),
            'message': 'Daily backup completed successfully',
            'source': 'BackupService',
            'level': 'INFO'
        },
        {
            'id': '5',
            'timestamp': datetime.now().isoformat(),
            'message': 'Scheduled maintenance completed on Server03',
            'source': 'Maintenance',
            'level': 'INFO'
        },
        
        # Repeat similar normal events many times to establish baseline
        *[{
            'id': f'{i+6}',
            'timestamp': datetime.now().isoformat(),
            'message': f'Normal operation log #{i}',
            'source': 'System',
            'level': 'INFO'
        } for i in range(50)],
        
        # Anomalous events
        {
            'id': '56',
            'timestamp': datetime.now().isoformat(),
            'message': 'Failed login attempt from unknown IP 203.0.113.99',
            'source': 'AuthService',
            'level': 'WARNING'
        },
        {
            'id': '57',
            'timestamp': datetime.now().isoformat(),
            'message': 'Temperature Sensor Rack03: Critical temperature alert 87°C',
            'source': 'Sensor',
            'level': 'CRITICAL'
        },
        {
            'id': '58',
            'timestamp': datetime.now().isoformat(),
            'message': 'Unusual network traffic pattern detected on VLAN 5',
            'source': 'Network',
            'level': 'WARNING'
        }
    ]
    
    # Process events
    logger.info(f"Processing {len(sample_events)} sample events")
    
    for i, event in enumerate(sample_events):
        # Preprocess event
        event_vector = preprocessor.preprocess(event)
        
        # After learning from first 55 events, check for anomalies
        if i >= 55:
            is_anomalous = model.is_anomalous(event_vector, threshold=0.7)
            if is_anomalous:
                logger.info(f"Anomaly detected: {event['message']}")
            else:
                logger.info(f"No anomaly detected for: {event['message']}")
        
        # Update model with event
        model.update(event_vector, event)
    
    # Clean up
    import shutil
    try:
        shutil.rmtree("./tmp_test_db")
    except:
        pass
    logger.info("Test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syslog Anomaly Detection Examples")
    parser.add_argument("--run", action="store_true", help="Run the detector")
    parser.add_argument("--simulate", action="store_true", help="Simulate learning process")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing data")
    parser.add_argument("--test", action="store_true", help="Test with sample data")
    
    args = parser.parse_args()
    
    if args.run:
        run_detector()
    elif args.simulate:
        simulate_learning_process()
    elif args.analyze:
        analyze_existing_data()
    elif args.test:
        test_with_sample_data()
    else:
        parser.print_help()
        sys.exit(1)
