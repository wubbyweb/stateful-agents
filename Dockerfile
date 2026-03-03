# ============================================================================
# Dockerfile for Multi-Agent Research System
# ============================================================================

# Use a slim Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some Python libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set a non-root user for security (best practice for Azure Container Apps)
RUN useradd -m myuser
USER myuser

# Default command runs the CLI (can be overridden by entrypoint in Azure)
ENTRYPOINT ["python", "main.py"]
CMD ["--topic", "AI agents in enterprise", "--dry-run"]
