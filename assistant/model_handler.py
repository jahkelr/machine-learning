from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from config import CONFIG

bnb_confing = BitsAndBytesConfig(
    load_in_8bit=True
)

class ModelHandler:
    def __init__(self):
        # Initialize based on CONFIG options
        self.use_pipeline = CONFIG["use_pipeline"]

        if self.use_pipeline:
            # Use pipeline approach for text generation
            self.pipe, self.messages = self.get_pipeline()
        else:
            # Load model directly
            self.tokenizer, self.model = self.load_model()

    def get_pipeline(self):
        """Initialize and return the text generation pipeline."""
        messages = [{"role": "user", "content": "Who are you?"}]
        pipe = pipeline("text-generation", model=CONFIG["llm_model"])
        return pipe, messages

    def load_model(self):
        """Load model and tokenizer directly."""
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"])
        model = AutoModelForCausalLM.from_pretrained(CONFIG["llm_model"], quantization_config=bnb_confing)
        return tokenizer, model

    def generate_response(self, prompt, user_context, max_new_tokens=2048):
        """Generate a response from the model or pipeline based on CONFIGuration."""
        if self.use_pipeline:
            # If using pipeline, we can directly generate the response
            self.messages.append({"role": "user", "content": prompt})
            return self.pipe(self.messages)
        else:
            messages = [
                {"role": "system", "content": "You are Q, created by me, Jahkel. You are a helpful assistant that responds to the best of your ability."},
                {"role": "system", "content": f"The user has entered the following relvant context to aid your response: {user_context}"},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_return_sequences=1
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
