# Dockerfile for FastAPI endpoint

# Use a base image with Python and necessary dependencies
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy FastAPI code into the container
COPY serve.py /app/
COPY requirements.txt /app/

# Install deps
RUN pip install -r requirements.txt

# Define the entry point for running FastAPI application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
