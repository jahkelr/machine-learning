from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

# Define a Pydantic model for input data
class QuestionInput(BaseModel):
    question: str


# Define the endpoint to handle questions
@app.post("/ask")
async def ask_question(question_input: QuestionInput):
    # Prepare the input data
    data = {"question": question_input.question}

    # Send a POST request to the RAG container using the alias
    response = requests.post("http://rag-container:8001/answer", json=data)

    if response.status_code == 200:
        # Extract and return the generated text
        answer = response.json().get("answer")
        return {"answer": answer}
    else:
        raise HTTPException(
            status_code=500, detail="Failed to get an answer from the RAG pipeline"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
