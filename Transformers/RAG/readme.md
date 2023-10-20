# Q&A model

## How new models are trained

`python train.py`

## How to create appliaction loacally

```docker-compose build
docker-compose up
```

## How to query endpoint

### Input

```curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the capitol of France?"}' "http://localhost:8000/ask"
```

### Output

`{answer: "paris"}`

## What's missing

The code to create the custom faiss index for use as the retriever is missing. I attempted to get it working for they deployments but couldn't find a reliable way to train similtaneously within the time limit.

## Notes

Two containers allow for the modularization of the two main functions: prediction and running an endpoint. The endpoint containers can be replicated many times without the memory of modeling files taking space and the model can be updated independantly by swapping for a fresh container that can run alongside legacy containers.
