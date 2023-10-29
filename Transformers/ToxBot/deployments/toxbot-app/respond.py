from fastapi import FastAPI, HTTPException
import requests
import json

app = FastAPI()

# Define the endpoint to response contexts
@app.get("/")
def respond_to_context(context: str):
    try:
        data = {"context": context}
        data = json.encoder(data)

        # Send a POST request to the RAG container using the alias
        response = requests.get("http://generator-container:8001/run/", json=data)

        return {"context": context, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate an response")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
