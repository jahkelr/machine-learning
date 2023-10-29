# Dockerfile for pipeline components

# Use a base image with Python and necessary dependencies
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy model files and code into the container
COPY generator.py /app/
COPY requirements.txt /app/

# Install deps
RUN pip install -r requirements.txt

# Define the entry point for running the FastAPI application
CMD ["uvicorn", "generator:app", "--host", "0.0.0.0", "--port", "8001"]
