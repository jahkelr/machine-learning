from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RagRetriever, RagConfig
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Load the RAG model and tokenizer
model_path = "facebook/rag-token-base" # Update to use fine-tuned
retriever = RagRetriever.from_pretrained(model_path)
config = RagConfig.from_pretrained(model_path, retriever=retriever)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the endpoint to answer questions
@app.get("/answer/")
def answer_question(question: str):
    try:
        # Generate answers for the given question
        inputs = tokenizer(
            "question: " + question,
            return_tensors="pt",
            padding="max_length",
            max_length=50,
            truncation=True,
        )
        outputs = model.generate(**inputs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate an answer")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
