# Intelligent Prosthetic Arm Project

![Project Banner](banner-image.png) *(Optional: Add a banner if applicable)*

## Overview

The Intelligent Prosthetic Arm project aims to develop a highly functional prosthetic hand controlled via EEG brain signals, integrated with computer vision and sensors for safe, real-time object manipulation. The system will utilize image recognition, object detection, and machine learning to enable the prosthetic arm to interact with everyday objects, such as water bottles, mimicking natural hand movements.

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
- **EEG-Based Control:** The prosthetic arm is controlled using brain signals, allowing for intuitive manipulation.
- **Computer Vision for Object Detection:** The system uses pre-trained YOLO models to detect objects in real-time and adjusts the prosthetic hand’s position for successful grasping.
- **Sensors for Safe Interaction:** The arm is equipped with force and temperature sensors to provide feedback and prevent excessive pressure on objects or the user.
- **ROS2 Integration:** The project utilizes ROS2 for smooth communication between the various components (EEG, sensors, computer vision).
- **Real-Time Processing:** Built for real-time interaction using a powerful RTX3090 GPU, with communication to a Raspberry Pi for practical use.

## Project Structure

```markdown

prosthetic_arm/
├── computer_vision/            # Vision processing modules
│   ├── data/
│   └── models/
│   └── scripts/
│   └── requirements.txt
│   └── README.md       
│
├── robotics/                   # Robotic control modules
│
│── neuroscience/               # EEG Data Processing modules
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

1. **Object Detection**
Run the object detection script to detect objects in real-time using your system camera:
```bash
python3 computer_vision/scripts/object_detection.py
```

2. **Grasp Detection**
Execute grasp detection for ensuring the correct positioning of the prosthetic hand:
```bash
python3 computer_vision/scripts/grasp_detection.py
```

3. **Real-Time Testing**
Run the system in real-time, sending data to the Raspberry Pi for inference during use.

## Roadmap
- [x] Set up object detection with YOLOv8
- [ ] Implement grasp detection using bounding boxes and segmentation
- [ ] Integrate sensor feedback with computer vision
- [ ] EEG signal integration for real-time control
- [ ] Refine ROS2 communication for seamless module interaction
- [ ] Expand to detect various object types beyond water bottles
- [ ] Real-world testing on prosthetic hardware


## Contributors
- **Leonardo Vanni**  - [info@leonardovanni.com](mailto:info@leonardovanni.com)
- **Anna Notaro** - [anna.notaro@studbocconi.it](mailto:anna.notaro@studbocconi.it)
- **Gianluca Fugante**  - [gianlucafugante@gmail.com](mailto:gianlucafugante@gmail.com)

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
