FROM jupyter/scipy-notebook:latest

USER root

# Install additional system packages
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Install additional Python packages
RUN pip install --no-cache-dir \
    mlflow \
    feast[redis] \
    evidently \
    great-expectations \
    plotly \
    seaborn \
    xgboost \
    lightgbm \
    transformers \
    torch

# Create work directory
RUN mkdir -p /home/jovyan/work

WORKDIR /home/jovyan/work

# Expose port
EXPOSE 8888
