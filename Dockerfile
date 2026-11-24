# MAGIC Gamma Classifier - everything in one image: `docker run -p 8501:8501 magic-gamma`
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    CUDA_VISIBLE_DEVICES=-1

WORKDIR /app

# libgomp is needed by scikit-learn / TensorFlow for multithreading
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x docker/entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

ENTRYPOINT ["docker/entrypoint.sh"]
