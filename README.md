# SAD AI (Syslog Anomaly Detection) AI System

A machine learning-based system for detecting anomalies in syslog data using a combination of K-Means clustering, InfoNCE contrastive learning, and reinforcement learning optimization.

## Overview

This system continuously monitors syslog events from a Graylog server, learns patterns of normal behavior, and alerts administrators when anomalous events are detected. The advanced reinforcement learning component automatically optimizes the anomaly detection threshold for maximum accuracy.

### Core Anomaly Detection

The system uses multiple machine learning approaches to identify unusual patterns:

- **K-Means Clustering**: Groups similar events into clusters to establish baseline patterns
- **InfoNCE Contrastive Learning**: Learns distinctive embeddings to identify outliers
- **Incremental Learning**: Continuously updates the model as new events arrive
- **Embedding Database**: Stores historical event data for model training and analysis

### Reinforcement Learning Enhancement

The traditional anomaly detection approach uses a fixed threshold to determine whether an event is anomalous. Our RL extension learns the optimal threshold through experience, allowing the system to adapt to changing environments and maximize detection accuracy:

- **Adaptive Thresholds**: Learns to adjust the anomaly threshold based on feedback
- **Q-Learning Algorithm**: Optimizes threshold selection to maximize detection performance
- **Persistent Learning**: Saves and loads the RL model to continue learning across restarts

## Components

### Base System
- `main.py`: Main application class for the standard anomaly detection system
- `graylog_client.py`: Client for interacting with the Graylog API
- `preprocessor.py`: Converts syslog events into numerical embedding vectors
- `model.py`: Implementation of the anomaly detection model (K-Means + InfoNCE)
- `database.py`: Database for storing event embeddings and metadata
- `alert_manager.py`: Manages sending email and webhook alerts
- `config.py`: Configuration management for the system
- `examples.py`: Example usage and simulation tools

### RL Extension
- `reinforcement_learning.py`: RL models for optimizing detection thresholds
- `rl_main.py`: Main application that integrates RL with anomaly detection
- `rl_analysis.py`: Tools for analyzing and visualizing RL performance

## Features

- Integration with Graylog's API to fetch recent syslog events
- Automatic preprocessing of log events into numerical embeddings
- Anomaly detection using combined clustering and contrastive learning
- Reinforcement learning to automatically optimize detection thresholds
- Multiple feedback modes for RL training:
  - `auto`: Automatically generates feedback based on heuristics
  - `manual`: Allows operators to provide real-time feedback on detected anomalies
  - `simulated`: Uses a simulated environment to provide feedback
- Email and webhook alerts when anomalies are detected
- Long-term storage of event embeddings in an SQLite database
- Configurable parameters for model optimization
- Performance visualization and analysis tools

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
- Model parameters (embedding dimension, number of clusters)
- Database settings
- Alert configuration (email recipients, SMTP settings)
- Logging configuration

## Usage

### Standard Anomaly Detection

```bash
# Run the detector with default configuration
python -m main
```

This will start the anomaly detection process with a fixed threshold.

### RL-Enhanced Anomaly Detection

```bash
# Train the RL model with simulation
python -m rl_main --simulate --episodes 100

# Run with automatic feedback
python -m rl_main --run --feedback auto

# Run with interactive feedback
python -m rl_main --run --feedback manual
```

### Example Tools and Analysis

```bash
# Run various examples and simulations
python -m examples --test

# Analyze RL model performance
python -m rl_analysis
```

## How the RL Model Works

The reinforcement learning model uses Q-learning to find the optimal threshold:

1. **State Space**: The different possible threshold values (discretized)
2. **Action Space**: Selecting a specific threshold for anomaly detection
3. **Reward Function**: A combination of:
   - Positive rewards for true positives (correctly identified anomalies)
   - Negative rewards for false positives (normal events incorrectly flagged)
   - Large negative rewards for false negatives (missed anomalies)
   - Small positive rewards for true negatives (correctly ignored normal events)
   - An episode-level reward based on the F1 score

## Monitoring Learning Progress

After training, the system generates plots in the `./results/` directory:

- `metrics_over_time.png`: F1, precision, and recall scores over episodes
- `threshold_selection.png`: Selected thresholds over episodes
- `reward_progression.png`: Rewards over episodes
- `confusion_metrics.png`: True/false positive/negative counts
- `f1_vs_threshold.png`: F1 scores for different thresholds
- `performance_summary.txt`: Text summary of model performance

## Learning Process

The system's learning process occurs in several stages:

### Stage 1: Initial ML Model Learning
1. When the database is empty, the first events establish the baseline
2. As more events arrive, clusters begin to form
3. After processing hundreds of events, the model develops a robust understanding of normal patterns

### Stage 2: RL Threshold Optimization
1. RL model begins with high exploration (trying different thresholds)
2. Feedback updates Q-values for different thresholds
3. System gradually favors thresholds with higher rewards

### Stage 3: Mature Operation
1. ML models have stable clusters representing normal patterns
2. RL model has converged to an optimal threshold
3. System accurately detects anomalies with minimal false positives/negatives

## Performance Metrics

The system tracks several performance metrics:

- **True Positives**: Correctly identified anomalies
- **False Positives**: Normal events incorrectly flagged as anomalies
- **False Negatives**: Missed anomalies
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual anomalies
- **F1 Score**: Harmonic mean of precision and recall

These metrics are used to assess the quality of the selected threshold and guide the learning process.

## Customization

- Adjust the learning hyperparameters in the RL model for different exploration/exploitation tradeoffs
- Modify the `num_clusters` parameter based on the complexity of your log patterns
- Configure the `embedding_dim` to balance between detail and noise
- Set up different alert mechanisms based on your operational requirements

## License

MIT