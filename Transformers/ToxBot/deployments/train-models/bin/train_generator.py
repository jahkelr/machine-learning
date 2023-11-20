import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import pickle
import os

from transformers import GPT2ForQuestionAnswering
from transformers import GPT2Tokenizer
from transformers import get_scheduler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import mlflow
import evaluate


class Transformed_ds(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer) -> None:
        super().__init__()

        print("Initializing Dataset...")

        # Tokenize input sequences
        self.inputs = tokenizer(
            dataset["context"].to_list(),
            dataset["Text"].to_list(),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,  # Adjust max sequence length as needed
            add_special_tokens=True,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "inputs": self.inputs[index],
        }


class GPT2Trainer:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2ForQuestionAnswering.from_pretrained("gpt2")
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.batch_size = 64

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def load_datasets(self):
        if not os.path.exists("data/processed_data.pkl"):
            dataset = pd.read_pickle("data/toxic_dataset.pkl")
            tds = Transformed_ds(dataset, self.tokenizer)

            with open("data/processed_data.pkl", "wb") as file:
                pickle.dump(tds, file)
        else:
            with open("data/processed_data.pkl", "rb") as file:
                tds = pickle.load(file)

        self.dataloader = DataLoader(tds, shuffle=True, batch_size=self.batch_size)

    def train(self):
        num_epochs = 1
        num_training_steps = num_epochs * len(self.dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Log with MLFLow
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("num_training_steps", num_training_steps)
        mlflow.log_param("batch_size", self.dataloader.batch_size)
        mlflow.log_param("learning_rate", self.optimizer.defaults["lr"])

        self.model.to(self.device)

        # Save tokenizer
        with open("../../models/tokenizer.pt", "wb") as f:
            torch.save(self.tokenizer, f)

        progress_bar = tqdm(range(num_training_steps), desc="Training Progress:")

        self.model.train()
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

        with open("models/model.pth", "wb") as f:
            torch.save(self.model, f)

    def evaluate(self):
        metric = evaluate.load("rouge")

        if os.path.exists("models/model.pth"):
            with open("models/model.pth", "rb") as f:
                model = torch.load(f)
            self.model = model

        self.model.eval()

        progress_bar = tqdm(range(len(self.dataloader)), desc="Evaluation Progress:")

        for batch in self.dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                self.model.forward()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(
                predictions=self.tokenizer.batch_decode(predictions),
                references=self.tokenizer.batch_decode(batch["labels"]),
            )
            progress_bar.update(1)

        metric_value = metric.compute()
        mlflow.log_metric("rouge_score", metric_value)

        return metric_value


def main():
    mlflow.set_tracking_uri("http://0.0.0.0:8080")  # Set the appropriate tracking URI
    with mlflow.start_run():
        trainer = GPT2Trainer()
        trainer.load_datasets()

        trainer.train()
        metric_value = trainer.evaluate()

        # Log the evaluation metric to the run
        mlflow.log_metric("final_metric", metric_value)

    # End the run
    mlflow.end_run()


if __name__ == "__main__":
    main()
