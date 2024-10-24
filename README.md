# Intelligent Prosthetic Arm Project

## Overview

The Intelligent Prosthetic Arm project is focused on developing a prosthetic hand controlled through EEG brain signals, combined with advanced computer vision and sensors for safe, real-time object interaction. The system incorporates machine learning, real-time image recognition, object detection, and sensor data to enable the prosthetic arm to mimic natural hand movements, providing a highly functional solution for users to interact with everyday objects, such as water bottles and more.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributors](#contributors)
- [License](#license)

## Features
- **EEG-Based Control:** The prosthetic arm is operated using brain signals (EEG), allowing intuitive control and natural movement.
- **Computer Vision for Object Detection and Grasping:** The system uses YOLOv11 for real-time object detection, positioning the prosthetic hand correctly for successful grasps based on feedback.
- **Real-Time Grasp Detection:** Integrated computer vision for determining the best grasp configuration for various objects using bounding boxes, segmentation, and a Vision Language Model (VLM).
- **Sensors for Safe Interaction:** Equipped with force and temperature sensors to ensure safe and adaptive interaction with objects.
- **ROS2 Integration:** ROS2 is used for seamless communication between the arm, computer vision, EEG module, and sensor feedback.

## Project Structure

```markdown

prosthetic_arm/
├── computer_vision/            # Vision processing modules
│   ├── frames/
│   ├── models/
│   ├── scripts/
│   ├── requirements.txt
│   ├── requirements_no_dep.txt
│   └── README.md
│
├── robotics/                   # Robotic control modules
│   └── arm_control.py
│
│── neuroscience/               # EEG Data Processing modules
│   └── eeg_processing.py
│
│── docker/                     # Container configuration
│   └── Dockerfile
│
├── docs/                       # Documentation
│   └── project_overview.md
│           
├── requirements.txt            # Project dependencies
└── README.md                   # Main documentation

```

## Installation

### Local Setup
1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/prosthetic_arm.git
cd prosthetic_arm
```

2. **Create a Python Virtual Environment**
```bash
python3 -m venv prosthetic_env
source prosthetic_env/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Docker Setup
1. **Build the Docker Image**
```bash
docker build -t prosthetic_arm_image .
```

2. **Run the Docker Container**
```bash
docker run -it --rm prosthetic_arm_image
```

## Usage

1. **Object Detection and Grasp Validation**
Run the ObjectDetector and GraspValidator classes to detect objects and validate grasping in real-time using your system camera:
```bash
python3 computer_vision/scripts/main.py
```

## Roadmap
- [x] Set up object detection with YOLOv11
- [x] Implement grasp detection using bounding boxes
- [ ] Integrate EEG signal processing for real-time control
- [ ] Finalize force and temperature sensor integration
- [ ] Optimize communication with Raspberry Pi for real-time operation
- [ ] Expand to detect various object types beyond water bottles
- [ ] Real-world testing on prosthetic hardware
- [ ] Safety features for preventing excessive pressure


## Contributors
- **Leonardo Vanni**  - [leonardo.vanni@studbocconi.it](mailto:leonardo.vanni@studbocconi.it)
- **Anna Notaro** - [anna.notaro@studbocconi.it](mailto:anna.notaro@studbocconi.it)
- **Gianluca Fugante**  - [gianlucafugante@gmail.com](mailto:gianlucafugante@gmail.com)

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
