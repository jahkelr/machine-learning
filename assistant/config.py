# Define model and tokenizer loading configurations
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Optional configurations for memory and other system parameters
CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # For memory embeddings
    "max_memory_size": 100,  # Define max number of memories to store
    "llm_model": MODEL_NAME,  # Model name for direct loading
    "use_pipeline": False,
}
