from memory_manager import MemoryManager

class RAG:
    def __init__(self, model_handler, memory_manager: MemoryManager):
        self.model_handler = model_handler
        self.memory_manager = memory_manager

    def get_relevant_context(self, query):
        return self.memory_manager.retrieve_memory(query)

    def generate_answer(self, query):
        context = self.get_relevant_context(query)
        context_str = "\n".join(context)
        prompt = f"{query}"
        return self.model_handler.generate_response(prompt, context_str)
