# Use a modern, official lightweight Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies (compiler and git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src /app/src
COPY run_pipeline.py /app/

# Expose FastAPI default port
EXPOSE 8000

# Default command launches FastAPI app
CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
