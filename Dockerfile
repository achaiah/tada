FROM python:3.11-slim

# Set environment variables to prevent python from writing .pyc files
# and to prevent the stdout/stderr stream from being buffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the HuggingFace cache directory to a well-known location
# This allows mounting a volume here to cache models across container restarts.
ENV HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install API dependencies
COPY api_requirements.txt /app/
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy source code and install the TADA package
COPY . /app/

# Install the package and its requirements
RUN pip install --no-cache-dir -e .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run uvicorn server on port 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
