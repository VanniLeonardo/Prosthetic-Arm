# Use the official ROS2 Humble image
FROM osrf/ros:humble-desktop

# Install dependencies for ROS2 and any additional tools you need
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rviz2 \
    ros-humble-ros-base \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /root/ros2_ws

# Source ROS2 environment automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Set the entrypoint for the container
ENTRYPOINT ["/bin/bash"]
