# rl_analysis.py
import logging
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from matplotlib.pyplot import figure

# Local imports
from reinforcement_learning import ThresholdOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_performance_history(model_path="./rl_model"):
    """
    Load performance history from the RL model
    
    Args:
        model_path: Path to the RL model directory
        
    Returns:
        List of performance metrics per episode
    """
    history_path = f"{model_path}/performance_history.json"
    
    if not os.path.exists(history_path):
        logger.error(f"Performance history file not found at {history_path}")
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        logger.info(f"Loaded performance history with {len(history)} episodes")
        return history
        
    except Exception as e:
        logger.error(f"Error loading performance history: {e}")
        return None

def analyze_learning_convergence(history):
    """
    Analyze the convergence of the learning process
    
    Args:
        history: List of performance metrics per episode
        
    Returns:
        Dictionary with convergence metrics
    """
    if not history:
        return None
        
    # Extract metrics
    f1_scores = [episode['f1_score'] for episode in history]
    thresholds = [episode['threshold'] for episode in history]
    rewards = [episode['reward'] for episode in history]
    
    # Calculate convergence metrics
    window_size = min(20, len(f1_scores) // 5)  # Use 20% of episodes or 20, whichever is smaller
    
    f1_mean = np.mean(f1_scores[-window_size:])
    f1_std = np.std(f1_scores[-window_size:])
    
    threshold_mean = np.mean(thresholds[-window_size:])
    threshold_std = np.std(thresholds[-window_size:])
    
    # Check if the threshold has converged
    has_converged = threshold_std < 0.05  # Consider converged if standard deviation is low
    
    # Calculate the best threshold
    best_episode = max(range(len(history)), key=lambda i: history[i]['f1_score'])
    best_threshold = history[best_episode]['threshold']
    best_f1 = history[best_episode]['f1_score']
    
    return {
        'episodes': len(history),
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'threshold_mean': threshold_mean,
        'threshold_std': threshold_std,
        'has_converged': has_converged,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_episode': best_episode + 1  # Convert to 1-indexed
    }

def visualize_performance_history(history, output_dir="./results"):
    """
    Create visualizations of the performance history
    
    Args:
        history: List of performance metrics per episode
        output_dir: Directory to save visualizations
    """
    if not history:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(history)
    
    # Create smoothed versions using rolling average
    window_size = max(5, len(df) // 20)  # 5% of data points or at least 5
    df['f1_score_smooth'] = df['f1_score'].rolling(window=window_size, center=True).mean()
    df['precision_smooth'] = df['precision'].rolling(window=window_size, center=True).mean()
    df['recall_smooth'] = df['recall'].rolling(window=window_size, center=True).mean()
    df['reward_smooth'] = df['reward'].rolling(window=window_size, center=True).mean()
    
    # 1. F1, Precision, and Recall over time
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['f1_score'], 'b-', alpha=0.3, label='F1 Score (raw)')
    plt.plot(df['episode'], df['precision'], 'g-', alpha=0.3, label='Precision (raw)')
    plt.plot(df['episode'], df['recall'], 'r-', alpha=0.3, label='Recall (raw)')
    
    plt.plot(df['episode'], df['f1_score_smooth'], 'b-', linewidth=2, label='F1 Score (smoothed)')
    plt.plot(df['episode'], df['precision_smooth'], 'g-', linewidth=2, label='Precision (smoothed)')
    plt.plot(df['episode'], df['recall_smooth'], 'r-', linewidth=2, label='Recall (smoothed)')
    
    plt.title('Classification Metrics Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_over_time.png")
    
    # 2. Threshold selection over time
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['threshold'], 'b-', label='Selected Threshold')
    
    # Add best performing threshold as horizontal line
    best_episode = df['f1_score'].idxmax()
    best_threshold = df.loc[best_episode, 'threshold']
    plt.axhline(y=best_threshold, color='r', linestyle='--', 
                label=f'Best Threshold ({best_threshold:.3f})')
    
    plt.title('Threshold Selection Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Threshold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/threshold_selection.png")
    
    # 3. Reward progression
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['reward'], 'g-', alpha=0.3, label='Reward (raw)')
    plt.plot(df['episode'], df['reward_smooth'], 'g-', linewidth=2, label='Reward (smoothed)')
    
    plt.title('Reward Progression Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_progression.png")
    
    # 4. Classification confusion metrics
    plt.figure(figsize=(12, 7))
    plt.plot(df['episode'], df['true_positives'], 'g-', label='True Positives')
    plt.plot(df['episode'], df['false_positives'], 'r-', label='False Positives')
    plt.plot(df['episode'], df['false_negatives'], 'b-', label='False Negatives')
    
    plt.title('Classification Confusion Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_metrics.png")
    
    # 5. Create a threshold vs. performance scatter plot
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(df['threshold'], df['f1_score'], 
               c=df['episode'], cmap='viridis', 
               alpha=0.7, s=50, edgecolors='w')
    
    plt.colorbar(scatter, label='Episode')
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Best Threshold ({best_threshold:.3f})')
    
    plt.title('F1 Score vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_vs_threshold.png")
    
    logger.info(f"Performance visualizations saved to {output_dir}")

def generate_performance_report(history, output_dir="./results"):
    """
    Generate a performance report from the RL model history
    
    Args:
        history: List of performance metrics per episode
        output_dir: Directory to save the report
    """
    if not history:
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze convergence
    convergence = analyze_learning_convergence(history)
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_episodes': len(history),
        'convergence': convergence,
        'last_episode': history[-1],
        'best_episode': history[convergence['best_episode'] - 1],
        'final_state': {
            'threshold': history[-1]['threshold'],
            'f1_score': history[-1]['f1_score'],
            'precision': history[-1]['precision'],
            'recall': history[-1]['recall']
        }
    }
    
    # Save full report as JSON
    with open(f"{output_dir}/performance_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a human-readable summary
    with open(f"{output_dir}/performance_summary.txt", 'w') as f:
        f.write("==== RL MODEL PERFORMANCE SUMMARY ====\n\n")
        
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes: {len(history)}\n\n")
        
        f.write("=== Convergence Analysis ===\n")
        f.write(f"Has Converged: {'Yes' if convergence['has_converged'] else 'No'}\n")
        f.write(f"Best Threshold: {convergence['best_threshold']:.4f} (Episode {convergence['best_episode']})\n")
        f.write(f"Best F1 Score: {convergence['best_f1']:.4f}\n")
        f.write(f"Recent Threshold Mean: {convergence['threshold_mean']:.4f} ± {convergence['threshold_std']:.4f}\n")
        f.write(f"Recent F1 Score Mean: {convergence['f1_mean']:.4f} ± {convergence['f1_std']:.4f}\n\n")
        
        f.write("=== Most Recent Episode (#{}) ===\n".format(history[-1]['episode']))
        f.write(f"Threshold: {history[-1]['threshold']:.4f}\n")
        f.write(f"F1 Score: {history[-1]['f1_score']:.4f}\n")
        f.write(f"Precision: {history[-1]['precision']:.4f}\n")
        f.write(f"Recall: {history[-1]['recall']:.4f}\n")
        f.write(f"True Positives: {history[-1]['true_positives']}\n")
        f.write(f"False Positives: {history[-1]['false_positives']}\n")
        f.write(f"False Negatives: {history[-1]['false_negatives']}\n\n")
        
        f.write("=== Best Performing Episode (#{}) ===\n".format(convergence['best_episode']))
        best_ep = history[convergence['best_episode'] - 1]
        f.write(f"Threshold: {best_ep['threshold']:.4f}\n")
        f.write(f"F1 Score: {best_ep['f1_score']:.4f}\n")
        f.write(f"Precision: {best_ep['precision']:.4f}\n")
        f.write(f"Recall: {best_ep['recall']:.4f}\n")
        f.write(f"True Positives: {best_ep['true_positives']}\n")
        f.write(f"False Positives: {best_ep['false_positives']}\n")
        f.write(f"False Negatives: {best_ep['false_negatives']}\n\n")
        
        f.write("=== Recommendation ===\n")
        if convergence['has_converged']:
            f.write(f"The model has converged to a stable threshold around {convergence['threshold_mean']:.4f}.\n")
            f.write(f"Recommended Production Threshold: {convergence['best_threshold']:.4f}\n")
        else:
            f.write("The model has not yet converged to a stable threshold.\n")
            f.write("Recommendation: Continue training for more episodes.\n")
            f.write(f"Current Best Threshold: {convergence['best_threshold']:.4f}\n")
    
    logger.info(f"Performance report saved to {output_dir}")
    
    return report

def interactive_analysis(model_path="./rl_model", output_dir="./results"):
    """
    Perform interactive analysis of the RL model
    
    Args:
        model_path: Path to the RL model directory
        output_dir: Directory to save analysis results
    """
    # Load optimizer to get current state
    optimizer = ThresholdOptimizer(model_path=model_path)
    
    try:
        optimizer.load_model()
    except Exception as e:
        logger.error(f"Failed to load RL model: {e}")
        return
    
    # Load performance history
    history = load_performance_history(model_path)
    
    if not history:
        logger.error("No performance history available")
        return
    
    # Generate visualizations
    visualize_performance_history(history, output_dir)
    
    # Generate performance report
    report = generate_performance_report(history, output_dir)
    
    # Print summary to console
    print("\n==== RL MODEL ANALYSIS ====\n")
    print(f"Model Path: {model_path}")
    print(f"Total Episodes: {len(history)}")
    
    # Convergence info
    if report['convergence']['has_converged']:
        print("\nModel has CONVERGED to a stable threshold")
    else:
        print("\nModel has NOT YET CONVERGED to a stable threshold")
    
    print(f"Best Threshold: {report['convergence']['best_threshold']:.4f} (Episode {report['convergence']['best_episode']})")
    print(f"Best F1 Score: {report['convergence']['best_f1']:.4f}")
    print(f"Recent Threshold Mean: {report['convergence']['threshold_mean']:.4f} ± {report['convergence']['threshold_std']:.4f}")
    
    # Current Q-values
    print("\nCurrent Q-values:")
    for threshold, q_value in zip(optimizer.thresholds, optimizer.q_values):
        print(f"  Threshold {threshold:.3f}: Q-value {q_value:.4f}")
    
    print("\nAnalysis results saved to:", output_dir)
    print("  - Visualizations (PNG files)")
    print("  - Performance report (JSON)")
    print("  - Performance summary (TXT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Model Analysis")
    parser.add_argument("--model-path", type=str, default="./rl_model", 
                        help="Path to RL model directory")
    parser.add_argument("--output-dir", type=str, default="./results", 
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    interactive_analysis(args.model_path, args.output_dir)
