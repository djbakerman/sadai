# SAD AI
# Syslog Anomaly Detection AI

A machine learning-based system for detecting anomalies in syslog data using a combination of K-Means clustering and InfoNCE contrastive learning.

## Overview

This system continuously monitors syslog events from a Graylog server, learns patterns of normal behavior, and alerts administrators when anomalous events are detected. The system uses:

- **K-Means Clustering**: Groups similar events into clusters to establish baseline patterns
- **InfoNCE Contrastive Learning**: Learns distinctive embeddings to identify outliers
- **Incremental Learning**: Continuously updates the model as new events arrive
- **Embedding Database**: Stores historical event data for model training and analysis

## Features

- Integration with Graylog's API to fetch recent syslog events
- Automatic preprocessing of log events into numerical embeddings
- Anomaly detection using combined clustering and contrastive learning algorithms
- Email and webhook alerts when anomalies are detected
- Long-term storage of event embeddings in an SQLite database
- Configurable parameters for threshold tuning and model optimization
- Reinforcement Learning extension to optimally set the anomaly threshold value

## Components

- `main.py`: Main application class for the anomaly detection system
- `graylog_client.py`: Client for interacting with the Graylog API
- `preprocessor.py`: Converts syslog events into numerical embedding vectors
- `model.py`: Implementation of the anomaly detection model (K-Means + InfoNCE)
- `database.py`: Database for storing event embeddings and metadata
- `alert_manager.py`: Manages sending email and webhook alerts
- `config.py`: Configuration management for the system
- `examples.py`: Example usage and simulation tools

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a configuration file (or use the default that will be created)

## Configuration

The system is configured using a YAML file (`config.yaml`). If the file doesn't exist, a default configuration will be created automatically.

Key configuration options:

- Graylog API URL and API key
- Model parameters (embedding dimension, number of clusters, anomaly threshold)
- Database settings
- Alert configuration (email recipients, SMTP settings)
- Logging configuration

## Usage

### Running the Detector

```bash
python -m main
```

This will start the anomaly detection process, which will:
1. Fetch recent syslog events from Graylog
2. Process events and check for anomalies
3. Send alerts if anomalies are detected
4. Update the model with new events
5. Sleep for the configured interval before repeating

### Example Usage

The `examples.py` script provides several demonstrations of the system:

```bash
# Run the detector with default configuration
python -m examples --run

# Simulate the learning process with synthetic data
python -m examples --simulate

# Analyze existing data in the database
python -m examples --analyze

# Test with sample syslog data
python -m examples --test
```

## Learning Process

The system's learning process occurs in several stages:

1. **Initialization Phase**: When the database is empty, the first events establish the baseline
2. **Early Learning Phase**: As more events arrive, clusters begin to form
3. **Mature Learning Phase**: After processing hundreds of events, the model develops a robust understanding of normal patterns

Anomaly detection becomes more accurate as the system processes more events.

## Customization

- Adjust the `anomaly_threshold` to control sensitivity
- Modify the `num_clusters` parameter based on the complexity of your log patterns
- Configure the `embedding_dim` to balance between detail and noise
- Set up different alert mechanisms based on your operational requirements

## License

MIT
