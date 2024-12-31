from memory_manager import MemoryManager
from model_handler import ModelHandler
from rag import RAG


def main():
    # Initialize components
    memory_manager = MemoryManager()
    model_handler = ModelHandler()
    rag_system = RAG(model_handler, memory_manager)

    while True:
        # User input
        user_input = input("You: ")

        # Store relevant facts in memory
        memory_manager.store_memory(user_input)

        # Get response from RAG model
        response = rag_system.generate_answer(user_input)

        # Display response
        print(f"Response: \n {response}")


if __name__ == "__main__":
    main()
