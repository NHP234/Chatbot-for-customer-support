    # 1. Base Image: Use an NVIDIA CUDA base image
    # We choose a CUDA 12.1 image to match the 'cu121' in your requirements.txt.
    # Using the devel image provides us with tools to build packages if needed.
    FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
    
    # 2. Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # 3. Install Python and other essentials
    # The NVIDIA image is based on Ubuntu, so we use 'apt-get'.
    # We also install 'git' because some pip packages need it.
    # We add 'ninja-build' which can speed up C++/CUDA extensions compilation for packages like flash-attn.
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        git \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*
    
    # 4. Set up working directory
    WORKDIR /app
    
    # 5. Copy requirements file
    COPY requirements.txt .
    
    # 6. Install Python dependencies
    # We upgrade pip, install a specific version of torch.
    # Then we install packages from requirements.txt and unsloth in a single command.
    # This helps pip resolve all dependencies together and respect the version constraints.
    RUN pip install --no-cache-dir --upgrade pip
    RUN pip install --no-cache-dir torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    RUN pip install --no-cache-dir \
        -r requirements.txt \
        "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
    
    # 7. Copy application code
    COPY . .
    
    # 8. Set up the run command
    # Gunicorn will listen on the port specified by the PORT variable,
    # which can be set when running the container (defaults to 8080).
    # Increased timeout for potentially long model loading times.
    # Using JSON array format for CMD is a best practice.
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "300", "app:app"]