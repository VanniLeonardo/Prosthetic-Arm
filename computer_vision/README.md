# Computer Vision Module - Intelligent Prosthetic Arm Project

## Overview

This module handles the computer vision aspects of the Intelligent Prosthetic Arm project. It focuses on object detection, grasp detection, and real-time feedback to ensure the prosthetic arm can interact safely with various objects. The initial focus is on detecting water bottles as the prototype object and expanding to other objects in future iterations.

## Features
- **Object Detection with YOLOv11:** Real-time detection of objects such as water bottles. 
- **Grasp Detection:** Identifying if the prosthetic hand is positioned correctly for grasping an object.
- **Vision Language Model (VLM) Evaluation:** Ensuring the hand is positioned correctly for successful grasps by evaluating the environment with a vision-language model.
- **Real-Time Feedback:** Efficient communication with a Raspberry Pi for real-time object interaction.

## Project Structure

```markdown
computer_vision/
├── data/                     # Datasets for training and testing
├── models/                   # Pre-trained models for object detection
├── scripts/                  # Python scripts for object and grasp detection
│   ├── object_detection.py   # YOLOv8-based object detection
│   ├── grasp_detection.py    # Grasp detection using bounding boxes and segmentation
│   └── vlm_evaluation.py     # VLM for grasp evaluation
├── requirements.txt          # Dependencies for the computer vision module
└── README.md                 # Documentation for the computer vision module
```

## Contributors
- **Leonardo Vanni**  - [leonardo.vanni@studbocconi.it](mailto:leonardo.vanni@studbocconi.it)
- **Gianluca Fugante**  - [gianlucafugante@gmail.com](mailto:gianlucafugante@gmail.com)

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
