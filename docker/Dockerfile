# Use a more lightweight ROS2 base image
FROM osrf/ros:humble-ros-base

# Install dependencies for ROS2 and other essential tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rviz2 \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    numpy==2.1.2 \
    opencv-python==4.10.0.84 \
    ultralytics==8.3.18 \
    scikit-learn==1.5.2

# Set up a working directory for ROS workspace
WORKDIR /root/ros2_ws

# Set up the ROS2 environment
RUN /bin/bash -c "source /opt/ros/humble/setup.bash"

# Create and initialize the ROS workspace
RUN mkdir -p /root/ros2_ws/src && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set the entrypoint for the container
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec bash"]
