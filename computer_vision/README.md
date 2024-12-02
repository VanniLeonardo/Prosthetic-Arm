# Intelligent Prosthetic Arm Project - Computer Vision Module

## Overview

This module handles the computer vision aspects of the Intelligent Prosthetic Arm project. It focuses on object detection, grasp detection, hand pose estimation, and real-time feedback to ensure the prosthetic arm can interact safely with various objects. The initial focus is on detecting water bottles as the prototype object, with plans to expand to other objects in future iterations.

## Features
- **Object Detection with YOLOv11:**  Real-time detection of objects such as water bottles using the YOLOv11 model.
- **Grasp Detection and Validation:** Identifying and validating if the prosthetic hand is positioned correctly for grasping an object based on geometric constraints and machine learning models.
- **Hand Pose Estimation with MediaPipe:** Utilizing MediaPipe for real-time hand landmark detection and evaluating hand openness and grasp poses.
- **Grasp Identification:** Combining object detection and hand pose estimation to determine the feasibility of a grasp.
- **Real-Time Feedback:** Efficient communication for real-time object interaction.
- **ROS2 Integration (Upcoming):** Planning to integrate ROS2 for enhanced communication and control mechanisms.

## Project Structure

```markdown
computer_vision/
├── data/                     # Datasets for training and testing
├── models/                   # Pre-trained models for object detection and hand landmarking
│   ├── yolo11l.pt            # YOLOv11 large model for object detection
│   └── hand_landmarker.task  # Model for hand landmark detection
├── scripts/
│   ├── grasp_identifier.py   # Grasp identification combining object and hand detection 
│   ├── grasp_validation.py   # Validates grasps using geometric and ML approaches
│   ├── object_detection.py   # Object detection using YOLOv11
│   ├── hand_landmarks.py     # Hand pose estimation using MediaPipe
│   └── main.py               # Main application script for grasp validation
├── requirements.txt          # Dependencies for the computer vision module
└── README.md                 # Documentation for the computer vision module
```

## Getting Started
### Prerequisites
- Python 3.12+
- OpenCV
- NumPy
- PyTorch
- MediaPipe
- Ultralytics YOLO
- Scikit Learn

Install the required dependencies using: 
``` bash
pip install -r requirements.txt
```

## Running the Application
Note: Ensure that a webcam is connected to your system as the application relies on real-time camera input

### 1. Main Grasp Validation Application
Run the following command to start the main application:
```bash
python main.py
```
This script performs real-time object detection and grasp validation, displaying the annotated video stream with detected objects and graspable indicators.

### 2. Grasp Identifier
To test the grasp identification combining hand landmarks and object detection, execute:
``` bash
python grasp_identifier.py
```
This script analyzes both the position of the detected objects and the hand pose to determine if a grasp action is feasible.

### 3. Hand Landmark Tracking
For testing hand pose estimation and tracking:
``` bash
python hand_landmarks.py
```
This script uses MediaPipe to detect hand landmarks and evaluates hand openness and grasp pose in real-time.

## Upcoming Features
- ROS2 Integration: The project will soon integrate ROS2 for improved communication between different modules of the prosthetic arm system.
- ML grasp Validation: Plans to incorporate first a model with binary output and then VLMs for improved grasp validation.

## Contributors
- **Leonardo Vanni**  - [leonardo.vanni@studbocconi.it](mailto:leonardo.vanni@studbocconi.it)
- **Gianluca Fugante**  - [gianlucafugante@gmail.com](mailto:gianlucafugante@gmail.com)

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
