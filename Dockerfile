FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Set environment variables to make apt-get non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VENV_PATH=/app/venv

# TODO CLEAN THESE UP NOT ALL NEEDED
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
    libgtk2.0-dev \
    pkg-config \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $VENV_PATH

RUN $VENV_PATH/bin/pip install --no-cache-dir --upgrade pip

COPY GUI/requirements.txt /app/

RUN $VENV_PATH/bin/pip install --no-cache-dir -r /app/requirements.txt
RUN $VENV_PATH/bin/pip install --no-cache-dir websockets 

COPY hand_landmarker.task /app/
COPY yolov5s6u.pt /app/
COPY sam2.1_b.pt /app/
COPY /GUI /app/GUI

# Expose the WebSocket port the server will listen on
EXPOSE 8765

CMD ["bash", "-c", "source /app/venv/bin/activate && python /app/GUI/pipeline_server.py"]