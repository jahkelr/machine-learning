# Required packages
import os
from transformers import AutoModelForSeq2SeqLM, RagConfig, RagRetriever
from transformers import RagTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

# Paths and filenames
model_output_dir = ".deployments/rag-container/rag_model"
model_name = "facebook/rag-token-base"  # Pretrained RAG model
train_dataset_name = "truthful_qa"

# Load RAG tokenizer
tokenizer = RagTokenizer.from_pretrained(model_name)

# Initialize the retriever
# retriever = RagRetriever.from_pretrained(os.path.join(model_output_dir, "index.faiss"))
retriever = RagRetriever.from_pretrained(model_name)

# Create training arguments
training_args = TrainingArguments(
    output_dir=model_output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=2,  # Adjust as needed
    dataloader_num_workers=4,
    save_steps=10_000,
    evaluation_strategy="steps",
    logging_dir="./logs",
    learning_rate=2e-5,  # Adjust as needed
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
)

# Initialize the RAG model
config = RagConfig.from_pretrained(
    model_name, retriever=retriever, title_sep=" / ", doc_sep=" / "
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

# Data collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    model_type="rag",
    pad_to_max_length=True,
    max_length=512,  # Adjust as needed
)

# Load dataset
dataset = load_dataset(train_dataset_name, "generation")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["validation"],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(os.path.join(model_output_dir, "model"))
tokenizer.save_pretrained(os.path.join(model_output_dir, "tokenizer"))

# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(results)

# Save the retriever separately for later use
retriever.save_pretrained(os.path.join(model_output_dir, "retriever"))

# Save the configuration as well
config.save_pretrained(os.path.join(model_output_dir, "config"))
