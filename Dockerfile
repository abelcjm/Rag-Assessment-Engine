FROM python:3.11-slim

# Install system dependencies required by PyMuPDF, OpenCV, and Docling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the rest of the requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Overwrite PyTorch and Torchvision with explicit CUDA 12.4 support
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Copy application source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000