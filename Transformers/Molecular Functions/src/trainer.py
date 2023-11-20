import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from tqdm import tqdm


class Trainer:
    def __init__(
        self, model, data_path, batch_size=32, learning_rate=0.001, num_epochs=10
    ):
        self.model = model
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self):
        # Load your dataset using the CustomDataset class
        dataset = CustomDataset(self.data_path)

        # Create a DataLoader for batching and shuffling
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define a loss function (binary cross-entropy for multi-label classification)
        criterion = nn.BCEWithLogitsLoss()

        # Define an optimizer (e.g., Adam)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Set the model in training mode
        self.model.train()

        for epoch in tqdm(range(self.num_epochs)):
            total_loss = 0.0
            for batch_inputs, batch_labels in tqdm(dataloader):
                # Forward pass
                outputs = self.model(batch_inputs)

                # Calculate the loss
                loss = criterion(outputs, batch_labels)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Print the average loss for this epoch
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f}")

    def __call__(self):
        self.train()


# Usage Example:
# Assuming you have a PyTorch model and a training data file path
# Define your model and specify the data_path
# model = YourCustomModel()
# data_path = "path/to/training/data.csv"
# trainer = Trainer(model, data_path)
# trainer()  # Call the Trainer to start training
