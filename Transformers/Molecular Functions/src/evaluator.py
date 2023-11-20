import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_score, recall_score
from itertools import cycle
from dataset import CustomDataset
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, model, data_path):
        self.model = model
        self.data_path = data_path

    def evaluate(self):
        # Load your test dataset and make predictions
        test_data = CustomDataset(
            self.data_path
        )  # Assuming you have a CustomDataset for testing
        test_loader = DataLoader(
            test_data, batch_size=32, shuffle=False
        )  # Adjust batch size as needed

        self.model.eval()
        all_true_labels = []
        all_predicted_probs = []

        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                outputs = self.model(batch_sequences)
                all_true_labels.extend(batch_labels.cpu().numpy())
                all_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy())

        all_true_labels = np.array(all_true_labels)
        all_predicted_probs = np.array(all_predicted_probs)

        # Create a list of thresholds to evaluate
        thresholds = np.arange(0.0, 1.1, 0.1)

        # Initialize dictionaries to store evaluation metrics
        evaluation_results = {
            "Threshold": [],
            "Precision": [],
            "Recall": [],
            "F1-Score": [],
            "ROC-AUC": [],
        }

        # Generate the ROC-AUC plot
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(all_true_labels.shape[1]):
            true_labels = all_true_labels[:, i]
            predicted_probs = all_predicted_probs[:, i]

            for threshold in thresholds:
                predicted_labels = (predicted_probs >= threshold).astype(int)
                precision = precision_score(true_labels, predicted_labels)
                recall = recall_score(true_labels, predicted_labels)
                f1_score = 2 * (precision * recall) / (precision + recall)
                roc_auc_i = roc_auc_score(true_labels, predicted_probs)

                evaluation_results["Threshold"].append(threshold)
                evaluation_results["Precision"].append(precision)
                evaluation_results["Recall"].append(recall)
                evaluation_results["F1-Score"].append(f1_score)
                evaluation_results["ROC-AUC"].append(roc_auc_i)

                # Calculate ROC curve and AUC for each label
                fpr[i], tpr[i], _ = roc_curve(true_labels, predicted_probs)
                roc_auc[i] = auc(fpr[i], tpr[i])

        # Save the evaluation metrics to a CSV file
        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_df.to_csv("evaluation_metrics.csv", index=False)

        # Create and save the ROC-AUC plot
        plt.figure(figsize=(10, 8))
        colors = cycle(
            [
                "aqua",
                "darkorange",
                "cornflowerblue",
                "green",
                "red",
                "blue",
                "purple",
                "pink",
                "brown",
                "gray",
            ]
        )

        for i, color in zip(range(all_true_labels.shape[1]), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve (area = %0.2f)" % roc_auc[i],
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig("roc_auc_plot.png")
