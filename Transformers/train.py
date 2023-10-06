import argparse
import csv

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
from model import TransformerFeedForward  # Import your custom model
from trainer import Trainer  # Import your Trainer class
from evaluator import Evaluator  # Import your Evaluator class
from dataset import CustomDataset


def find_best_threshold(model, test_loader):
    model.eval()
    all_true_labels = []
    all_predicted_probs = []

    with torch.no_grad():
        for batch_sequences, batch_labels in test_loader:
            outputs = model(batch_sequences)
            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy())

    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)

    thresholds = np.arange(0.0, 1.1, 0.1)
    best_threshold = None
    best_f1_score = 0.0

    for threshold in thresholds:
        predicted_labels = (all_predicted_probs >= threshold).astype(int)
        f1 = f1_score(all_true_labels, predicted_labels, average="micro")

        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold

    return best_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Transformer-based classifier."
    )
    parser.add_argument("--dataset", required=True, help="Path to the dataset file")
    parser.add_argument(
        "--model_name", required=True, help="Name of the transformer model"
    )

    args = parser.parse_args()

    # Set number of labels based on columns used
    with open(args.dataset, "r") as f:
        reader = csv.reader(f)
        num_labels = (
            len(next(reader)) - 2
        )  # protein column dropped, sequence used as input

    print("Making model...")
    # Load your custom TransformerFeedForward model
    model = TransformerFeedForward(
        args.model_name, num_labels
    )  # Replace 'num_labels' with the number of labels

    # Load the dataset, create DataLoader, and perform training
    trainer = Trainer(model, args.dataset)
    print("Beginning Training...")
    trainer.train()

    # Load the test dataset and create DataLoader
    test_data = CustomDataset(
        args.dataset
    )
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Find the best threshold based on F1-score
    best_threshold = find_best_threshold(model, test_loader)

    print(f"Best Threshold for F1-Score: {best_threshold}")

    # Evaluate the model using the best threshold
    model.eval()
    evaluator = Evaluator(model, args.dataset)
    evaluator.evaluate(best_threshold)
