# --------------------------------------------
# Base Stage: Use a Miniconda image as the starting point
# --------------------------------------------
FROM continuumio/miniconda3:latest AS base
WORKDIR /app

# System-level optimizations for performance
RUN echo "vm.swappiness=10" >> /etc/sysctl.conf && \
    echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf && \
    echo "vm.dirty_background_ratio=5" >> /etc/sysctl.conf && \
    echo "vm.dirty_ratio=10" >> /etc/sysctl.conf && \
    # Install any needed system utilities
    apt-get update && apt-get install -y --no-install-recommends \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------
# Builder Stage: Create the conda environment and build any extras
# --------------------------------------------
FROM base AS builder

# Install mamba for faster package management
RUN conda install -y mamba -n base -c conda-forge

# Copy the exported environment file into the image
COPY environment.yml .

# Create the conda environment as defined in environment.yml.
RUN mamba env create -f environment.yml && conda clean -afy

# Switch to the conda environment "mlc-prebuilt" for subsequent commands
SHELL ["conda", "run", "-n", "mlc-prebuilt", "/bin/bash", "-c"]

# Install build dependencies needed for additional packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      cmake \
      curl \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install Rust and set specific version
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    . $HOME/.cargo/env && \
    rustup default 1.70.0 && \
    rustup show

# Add Rust to bashrc for later steps
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

# Rename conda's internal linker to avoid glibc conflicts
# Only do this if absolutely necessary for your build
RUN mv "$CONDA_PREFIX/compiler_compat/ld" "$CONDA_PREFIX/compiler_compat/ld_bak" || echo "Linker not found, skipping rename"

# Install llama-cpp-python using the pre-built CPU wheel index
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Clone and build MLC-LLM with cargo/rustc environment loaded
RUN . $HOME/.cargo/env && \
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && \
    cd mlc-llm && \
    mkdir -p build && cd build && \
    # Create config.cmake with build settings - optimize for CPU
    echo 'set(CMAKE_BUILD_TYPE Release)' > ../cmake/config.cmake && \
    echo 'set(USE_CUDA OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_METAL OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_OPENCL OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_VULKAN OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_ROCM OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_OPENMP ON)' >> ../cmake/config.cmake && \
    # Add optimization flags for Ryzen CPU
    echo 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver2 -O3")' >> ../cmake/config.cmake && \
    # Run cmake with settings (these will override any config.cmake settings)
    cmake .. && \
    # Build with parallel jobs - limit to 4 for memory constraints
    cmake --build . --parallel 4

# Install the MLC-LLM Python package in editable mode
RUN cd mlc-llm/python && pip install -e .

# Install TVM Unity (Prebuilt Package)
RUN python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

# Install additional performance packages
RUN pip install uvloop httptools psutil

# Copy the rest of your application code
# Use the dot to copy all files in the current directory
COPY . .

# Upgrade torchvision for compatibility
RUN pip install --upgrade torchvision

# --------------------------------------------
# Final Stage: Create a lightweight runtime image
# --------------------------------------------
FROM continuumio/miniconda3:latest AS final
WORKDIR /app

# System-level optimizations carried to final image
RUN echo "vm.swappiness=10" >> /etc/sysctl.conf && \
    echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf && \
    echo "vm.dirty_background_ratio=5" >> /etc/sysctl.conf && \
    echo "vm.dirty_ratio=10" >> /etc/sysctl.conf

# Install runtime libraries (e.g., for OpenCV or similar)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      procps && \
    rm -rf /var/lib/apt/lists/*

# Copy the conda environment from the builder stage
COPY --from=builder /opt/conda/envs/mlc-prebuilt /opt/conda/envs/mlc-prebuilt

# Set the PATH so that the conda environment is used by default
ENV PATH=/opt/conda/envs/mlc-prebuilt/bin:$PATH
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set performance environment variables
ENV MLC_NUM_THREADS=4 \
    TVM_NUM_THREADS=4 \
    MLC_USE_CUDA=0 \
    MALLOC_TRIM_THRESHOLD_=65536

# Copy all application files from the builder stage
COPY --from=builder /app /app

# Create startup script with optimized uvicorn settings
RUN echo '#!/bin/bash\n\
uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 2 \
  --limit-concurrency 50 \
  --backlog 2048 \
  --timeout-keep-alive 5 \
  --log-level warning \
  --http httptools \
  --loop uvloop \
  --no-access-log\
' > /app/start.sh && chmod +x /app/start.sh

# Expose the port your app listens on
EXPOSE 8000

# Set the default command to run your application with optimized settings
CMD ["/app/start.sh"]