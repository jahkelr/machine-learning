# Dockerfile for FastAPI endpoint

# Use a base image with Python and necessary dependencies
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy train code into the container
COPY main.py /app/
COPY bin/ /app/bin/
COPY data/ /app/data/
COPY requirements.txt /app/

# Install deps
RUN pip install -r requirements.txt

# Expose endpoint
EXPOSE 8080

CMD ["python", "main.py"]