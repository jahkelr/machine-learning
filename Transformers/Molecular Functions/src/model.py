import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerFeedForward(nn.Module):
    def __init__(self, transformer_model_name, num_labels):
        super(TransformerFeedForward, self).__init__()

        # Load the pre-trained transformer model and tokenizer
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        # Define the feedforward network for classification
        self.ffn = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_text):
        # Tokenize input text and get transformer embeddings
        enc = self.tokenizer(
            input_text,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        transformer_outputs = self.transformer(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )[
            0
        ]  # Getting the hidden states (output)

        # Pool the embeddings (e.g., mean pooling or max pooling)
        pooled_embeddings = torch.mean(
            transformer_outputs, dim=1
        )  # You can use other pooling strategies

        # Pass pooled embeddings through the feedforward network
        logits = self.ffn(pooled_embeddings)
        return logits


# Usage Example:
# Initialize the model for your task with the appropriate transformer model name and number of labels
# transformer_model_name = 'bert-base-uncased'  # Replace with your preferred transformer model
# num_labels = 100  # Number of labels to predict
# model = TransformerFeedForward(transformer_model_name, num_labels)

# Define your training loop and Trainer class as previously shown
# trainer = Trainer(model, data_path)
# trainer()
