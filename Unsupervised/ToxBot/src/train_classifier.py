import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import mlflow

# Load the labeled dataset from a CSV file
data = pd.read_csv('data/classifier_data.csv')

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text.to_list()
        self.labels = labels.to_list()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = int(self.labels[idx])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define constants
MAX_LEN = 128
BATCH_SIZE = 32

# Initialize the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Split the dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create DataLoader for training and testing
train_dataset = CustomDataset(text=train_data['text'], labels=train_data['label'], tokenizer=tokenizer, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = CustomDataset(text=test_data['text'], labels=test_data['label'], tokenizer=tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
with mlflow.start_run():
    mlflow.log_params({"epochs": 3, "learning_rate": 2e-5, "batch_size": BATCH_SIZE})
    
    for epoch in range(3):  # Adjust the number of epochs as needed
        model.train()
        total_loss = 0
        
        # Use tqdm to track progress
        with tqdm(train_loader, unit="batch") as t:
            for batch in t:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
        
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
                loss.backward()
                optimizer.step()
                
                t.set_postfix({"Loss": total_loss / (t.n + 1)})  # Update progress bar
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{3}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    classification_rep = classification_report(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_rep)
    
    # Log metrics to mlflow
    mlflow.log_metrics({"accuracy": accuracy})
