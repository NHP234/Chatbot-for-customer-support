    # 1. Base Image: Use an NVIDIA CUDA base image
    # We choose a CUDA 12.1 image to match the 'cu121' in your requirements.txt.
    # Using the devel image provides us with tools to build packages if needed.
    FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
    
    # 2. Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # 3. Install Python and other essentials
    # The NVIDIA image is based on Ubuntu, so we use 'apt-get'.
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
    
    # 4. Set up working directory
    WORKDIR /app
    
    # 5. Install Python dependencies
    COPY requirements.txt .

    # First, install critical build dependencies that other packages (like flash-attn) need.
    # We install torch and packaging explicitly before the rest.
    RUN pip install --no-cache-dir --upgrade pip setuptools wheel
    RUN pip install --no-cache-dir packaging
    RUN pip install --no-cache-dir torch

    # Now, install the remaining requirements.
    RUN pip install --no-cache-dir -r requirements.txt
    
    # 6. Copy application code
    COPY . .
    
    # 7. Set up the run command
    # Gunicorn will listen on the port specified by the PORT variable,
    # which can be set when running the container (defaults to 8080).
    # Increased timeout for potentially long model loading times.
    # Using JSON array format for CMD is a best practice.
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]