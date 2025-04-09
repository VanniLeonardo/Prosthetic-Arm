FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Set environment variables to make apt-get non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VENV_PATH=/app/venv

# Update, install tzdata, set timezone non-interactively, then install other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y --no-install-recommends \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    # Add other specific OS dependencies
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libgtk2.0-dev\
    pkg-config\
    # Clean up apt caches
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv $VENV_PATH

# Activate the virtual environment and upgrade pip
RUN $VENV_PATH/bin/pip install --no-cache-dir --upgrade pip

# Copy the requirements file first to leverage Docker cache
COPY GUI/requirements.txt /app/
COPY hand_landmarker.task /app/
COPY GUI/constraints.txt /app/

# Install Python packages into the virtual environment
RUN $VENV_PATH/bin/pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt
RUN $VENV_PATH/bin/pip install --no-cache-dir websockets 

# Copy the only the /GUI folder
COPY /GUI /app/GUI

# Expose the WebSocket port the server will listen on
EXPOSE 8765

COPY yolov5s6u.pt /app/
COPY sam2.1_b.pt /app/

CMD ["bash", "-c", "source /app/venv/bin/activate && python /app/GUI/pipeline_server.py"]

# FROM hdgigante/python-opencv:4.11.0-ubuntu