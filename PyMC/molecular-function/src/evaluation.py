import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def calculate_evaluation_metrics(true_labels, predicted_labels):
    """
    Calculate evaluation metrics for model performance.
    
    Parameters:
    true_labels (np.array or pd.Series): True protein function labels.
    predicted_labels (np.array or pd.Series): Predicted protein function labels.
    
    Returns:
    dict: Dictionary of evaluation metrics (e.g., accuracy, F1-score).
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')  # You can choose an appropriate averaging method
    
    metrics = {
        'Accuracy': accuracy,
        'F1 Score': f1,
    }
    
    return metrics

def plot_results(results, labels):
    """
    Visualize model results using bar charts or other appropriate plots.
    
    Parameters:
    results (dict): Dictionary of evaluation metrics.
    labels (list): Labels for the evaluation metrics.
    
    Returns:
    None
    """
    # Create bar chart for evaluation metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, [results[label] for label in labels])
    ax.set_ylabel('Score')
    ax.set_title('Model Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # You can save the plot if needed
    plt.savefig('evaluation_plot.png')
    plt.show()

def main():
    # Load true protein function labels (ground truth)
    true_labels = np.array([1, 0, 1, 0, 1])  # Replace with your actual labels
    
    # Load predicted protein function labels from your Bayesian model
    predicted_labels = np.array([1, 0, 1, 1, 0])  # Replace with your model's predictions
    
    # Calculate evaluation metrics
    evaluation_metrics = calculate_evaluation_metrics(true_labels, predicted_labels)
    
    # Define labels for the evaluation metrics
    metric_labels = list(evaluation_metrics.keys())
    
    # Print evaluation metrics
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot and visualize the evaluation metrics
    plot_results(evaluation_metrics, metric_labels)

if __name__ == "__main__":
    main()
