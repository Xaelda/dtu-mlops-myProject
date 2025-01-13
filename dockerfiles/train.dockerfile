# Base image
FROM python:3.11.11-slim AS base

# Instal Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

## You are most likely going to rebuild your Docker image multiple times,
## either due to an implementation error or the addition of new functionality.
## Therefore, instead of watching pip suffer through downloading torch for the
## 20th time, you can reuse the cache from the last time the Docker image was built.
## Therefore we use RUN --mount=type=cache which mounts local pip cache to imaage.
WORKDIR /
#RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/my_project/train.py"]
