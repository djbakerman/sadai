# Reinforcement Learning for Syslog Anomaly Detection

This extension enhances the Syslog Anomaly Detection System with reinforcement learning capabilities to automatically optimize the anomaly threshold parameter.

## Overview

The traditional anomaly detection system uses a fixed threshold to determine whether an event is anomalous. This RL extension learns the optimal threshold through experience, allowing the system to adapt to changing environments and maximize detection accuracy.

## Components

- **ThresholdOptimizer**: A reinforcement learning agent that learns to select the optimal anomaly threshold
- **ThresholdEnvironment**: A simulated environment for faster training of the RL agent
- **RLAnomalyDetector**: Main application that integrates RL with the anomaly detection system

## Key Features

- **Adaptive Thresholds**: Learns to adjust the anomaly threshold based on feedback
- **Multiple Feedback Modes**:
  - `auto`: Automatically generates feedback based on heuristics
  - `manual`: Allows operators to provide real-time feedback on detected anomalies
  - `simulated`: Uses a simulated environment to provide feedback
- **Training Visualization**: Generates plots to monitor the learning progress
- **Persistent Learning**: Saves and loads the RL model to continue learning across restarts

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

## Usage

### Training with Simulation

```bash
python rl_main.py --simulate --episodes 100
```

This starts a simulated training process with 100 episodes. The simulator creates synthetic events with known anomaly status to train the model quickly.

### Running with Real-Time Learning

```bash
python rl_main.py --run --feedback auto
```

This runs the anomaly detection system with the RL-based threshold optimization. The `--feedback` flag determines how the system collects feedback:

- `auto`: Automatic feedback based on anomaly scores
- `manual`: Interactive prompts for operator feedback
- `simulated`: Simulated feedback for testing

## Monitoring Learning Progress

After training, the system generates plots in the `./results/` directory:

- `f1_scores.png`: F1 scores over episodes
- `thresholds.png`: Selected thresholds over episodes
- `rewards.png`: Rewards over episodes
- `q_values.png`: Final Q-values for different thresholds

## Integration with Base System

The RL extension seamlessly integrates with the existing anomaly detection system:

1. It uses the same underlying models (K-Means clustering and InfoNCE)
2. The anomaly scores from these models are passed to the RL module
3. The RL module determines the optimal threshold for those scores
4. Feedback is collected to improve the threshold selection over time

## Performance Metrics

The system tracks several performance metrics:

- **True Positives**: Correctly identified anomalies
- **False Positives**: Normal events incorrectly flagged as anomalies
- **False Negatives**: Missed anomalies
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual anomalies
- **F1 Score**: Harmonic mean of precision and recall

These metrics are used to assess the quality of the selected threshold and guide the learning process.
