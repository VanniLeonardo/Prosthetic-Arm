FROM python:3.11-slim

WORKDIR /app

# Install necessary packages, including wget for downloading the model
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment using standard uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_SYSTEM_PYTHON=1
# ENV UV_COMPILE_BYTECODE=1

WORKDIR /app/dev

# Create necessary directories
RUN mkdir models scripts

# Copy project files
COPY scripts ./scripts

# Download the model file (adjust the URL if needed)
RUN wget -P./models https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

WORKDIR /app/dev/scripts

COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Explicitly activate the virtual environment and then run the script
CMD [ "/bin/bash", "-c", "uv run main.py" ]