# --------------------------------------------
# Base Stage: Use Miniconda
# --------------------------------------------
FROM continuumio/miniconda3:latest AS base
WORKDIR /app

# --------------------------------------------
# Builder Stage
# --------------------------------------------
FROM base AS builder

# (1) Install mamba for faster dependency resolution
RUN conda install -y mamba -n base -c conda-forge

# (2) Copy environment.yml and create conda env with cleanup
COPY environment.yml .
RUN mamba env create -f environment.yml && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    find /opt/conda/ -name '*.pyc' -delete && \
    find /opt/conda/ -name '__pycache__' -exec rm -rf {} +

# (3) Switch to conda environment for next commands
SHELL ["/bin/bash", "-c"]

# (4) Install minimal OS packages and TeX dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gnupg curl apt-transport-https \
      pandoc \
      build-essential \
      git \
      cmake \
      wget \
      texlive-xetex \
      texlive-fonts-recommended \
      texlive-lang-english \
      texlive-latex-extra \
      lmodern && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# (5) Use conda env shell (mlc-prebuilt) for next commands
SHELL ["conda", "run", "-n", "mlc-prebuilt", "/bin/bash", "-c"]

# (6) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir faiss-cpu==1.10.0 && \
    pip install --no-cache-dir \
      huggingface_hub>=0.24.6 \
      transformers>=4.30.0 \
      datasets>=2.14.0 && \
    pip cache purge

# (7) Install MLC-python and llama-cpp-python
RUN pip install --no-cache-dir \
    mlc-python \
    mlc-ai-nightly-cpu \
    llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu && \
    pip cache purge

# Create directories for models and data
RUN mkdir -p /app/models /app/huggingface/cache /app/storage /app/llm/gguf

# Copy the download script first for better layer caching
COPY download_model.py .

# Copy the rest of your application code
COPY . .

# --------------------------------------------
# Final Stage
# --------------------------------------------
FROM continuumio/miniconda3:latest AS final
WORKDIR /app

# (A) Install minimal runtime libs and TeX dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      pandoc \
      texlive-xetex \
      texlive-fonts-recommended \
      texlive-lang-english \
      lmodern && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# (B) Create minimal runtime conda environment
RUN conda create -n mlc-runtime python=3.11 -y && \
    conda clean -afy && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    find /opt/conda/ -name '*.pyc' -delete

# (C) Copy only the necessary Python packages from builder
COPY --from=builder /opt/conda/envs/mlc-prebuilt/lib/python3.11/site-packages /opt/conda/envs/mlc-runtime/lib/python3.11/site-packages

# (D) Copy application files
COPY --from=builder /app/download_model.py /app/
COPY --from=builder /app/main.py /app/
COPY --from=builder /app/config.py /app/
COPY --from=builder /app/vector_store.py /app/
COPY --from=builder /app/data_loader.py /app/
COPY --from=builder /app/embedding_utils.py /app/
COPY --from=builder /app/llm_utils.py /app/
COPY --from=builder /app/services /app/services
COPY --from=builder /app/templates /app/templates
COPY --from=builder /app/image /app/image
COPY --from=builder /app/*.pt /app/
COPY --from=builder /app/train /app/train

# Copy Docker-specific .env file
COPY .env.docker /app/.env

# (E) Create directories for models and data
RUN mkdir -p /app/models /app/huggingface/cache /app/storage /app/llm/gguf

# (F) Set up environment variables
ENV PATH=/opt/conda/envs/mlc-runtime/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLC_MODEL_DIR="/app/models" \
    HF_HOME="/app/huggingface" \
    HF_CACHE_DIR="/app/huggingface/cache"

# (G) Add entrypoint script to handle model downloads at startup
ENTRYPOINT ["python", "download_model.py"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]