# Intelligent Prosthetic Arm Project

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<!-- Optional: Add other badges like build status if applicable -->

## Overview

The Intelligent Prosthetic Arm project aims to develop a functional prosthetic limb allowing users to perform daily tasks with dexterity approaching that of a human arm. By integrating **EEG-based control for interpreting user intent, advanced computer vision for scene understanding and grasp validation, and sensor feedback for safe interaction**, this system enables users to interact naturally with their environment. This project combines principles from robotics, neuroscience, and artificial intelligence to restore a significant degree of natural hand function for individuals with limb differences.

<!-- Consider adding a high-level conceptual image or GIF here if you have one -->

## Key Features

*   **Intuitive Control:** Utilizes non-invasive EEG signals processed via machine learning algorithms (e.g., EEGNet) to interpret user intent for arm movement and grasping actions.
*   **Advanced Visual Perception:** Employs a sophisticated computer vision pipeline (YOLO, SAM, DepthAnythingV2, MediaPipe) for real-time object detection, segmentation, depth estimation, 3D pose understanding, and hand tracking.
*   **Intelligent Grasping Validation:** Analyzes visual data (object properties, 3D pose) and user hand pose (if applicable) to validate the feasibility of intended grasps before execution. *(Grasp planning implementation is ongoing).*
*   **Sensor-Driven Interaction:** Plans to integrate force and temperature sensors to provide tactile feedback for grip modulation and ensure safe interaction with objects.
*   **Modular Architecture:** Designed with distinct modules for neuroscience, computer vision, and robotic control, planned for integration via **ROS2** running on a central **Raspberry Pi** controller.
*   **Real-Time Capable Components:** Core computer vision components are optimized for real-time processing, forming the basis for a responsive system.

## System Architecture

The intended operational pipeline is as follows:

1.  **EEG Acquisition:** Brain signals are captured.
2.  **Signal Processing & Intent Recognition:** The `neuroscience` module filters the raw EEG data, extracts relevant features, and uses a trained model (e.g., EEGNet) to classify the user's intended action (e.g., target object, desired grasp type). *(Currently under development, real-time capability planned).*
3.  **Computer Vision Analysis:** Simultaneously, the `computer_vision` module, running potentially accelerated on appropriate hardware, processes input from a camera. It detects relevant objects, estimates their 3D pose and properties, and optionally tracks the user's residual limb or control interface.
4.  **Grasp Validation & Command Generation:** The central controller (planned: Raspberry Pi) receives the user's high-level intent and the detailed visual scene analysis. It validates if the intended action (e.g., grasping the detected bottle) is feasible based on object location, orientation, and hand position/state. If valid, it generates appropriate commands for the robotic arm.
5.  **Robotic Execution:** The controller sends commands (planned: via ROS2) to the `robotics` module, which drives the custom-designed prosthetic arm. *(Hardware under construction, control logic in development).*
6.  **Sensor Feedback & Refinement:** During interaction, integrated sensors (planned: force, temperature) provide feedback. This data will be used to refine the grip, confirm successful grasps, or implement safety stops.
7.  **Communication:** All inter-module communication is planned to be handled via **ROS2** topics and services.

**System Diagram:**

```markdown
+-------------------+ +-------------------------+ +-----------------+
| EEG Hardware |----->| Neuroscience Module |----->| Central Control |
| (TBD) | | (Filtering, EEGNet) | | (Raspberry Pi) |
+-------------------+ +-------------------------+ +-------+---------+
^ |
| Intent | Command + Validation
| v
+-------------------+ +-------------------------+ +---------+-------+ +-----------------+
| Camera |----->| Computer Vision Module |----->| (ROS2 Planned) |<---->| Robotics Module |
| | | (YOLO, Depth, SAM, ...) | +-----------------+ | (Custom Arm) |
+-------------------+ +-------------------------+ +--------+--------+
| Scene Info, |
| Grasp Feasibility | Sensor Data
v |
+-------------------+ +-------+---------+
| Central Control |<----------------------------------------| Sensors |
| (Raspberry Pi) | | (Force, Temp) |
+-----------------+ +-----------------+
```
*Note: This diagram represents the planned data flow.*

**(A more detailed graphical block diagram will be inserted here later)**
`[Link to System Architecture Diagram]`

## Technology Stack

*   **Programming Language:** Python 3.12+
*   **Core Libraries:** PyTorch, OpenCV, NumPy, Scikit-learn, MediaPipe, Ultralytics, Transformers, FilterPy, PySide6 (for GUI)
*   **Computer Vision Models:** YOLOv5/v9/v11 (Object Detection), SAM (Segmentation), DepthAnythingV2 (Depth Estimation), MediaPipe HandLandmarker
*   **Robotics:**
    *   Hardware: Custom-designed multi-DOF prosthetic arm (Under construction)
    *   Control Software: Custom Python scripts (`arm_control.py` - *implementation TBD*), ROS2 (Planned)
*   **Neuroscience:**
    *   Hardware: TBD
    *   Signal Processing: Custom Python scripts (`eeg_processing.py`), potentially MNE, SciPy, TensorFlow/Keras (for EEGNet)
*   **Communication:** ROS2 (Planned)
*   **Controller:** Raspberry Pi (Planned)
*   **Sensors:** Force Sensors, Temperature Sensors (Specific models TBD)

## Project Structure

```markdown
prosthetic_arm/
├── computer_vision/ # Vision processing (Detection, Depth, Segmentation, Hand Tracking, Grasping Validation)
│ ├── models/ # Pre-trained CV models (YOLO, SAM, Depth, Hand)
│ ├── scripts/ # Executable CV scripts (main.py, grasp_identifier.py, etc.)
│ ├── requirements.txt # CV-specific dependencies
│ └── README.md # CV module documentation
│
├── robotics/ # Robotic control and hardware interface (Implementation Planned)
│ └── arm_control.py # Placeholder for arm control logic
│
├── neuroscience/ # EEG signal processing and intent recognition (Partially Implemented)
│ └── eeg_processing.py # EEG processing logic
│
├── sensors/ # Sensor interfacing (Planned)
│ └── [Sensor handling scripts]
│
├── docs/ # Documentation files
│ └── [Architecture diagrams, etc.]
│
├── requirements.txt # Overall project dependencies
├── GUI.py # Graphical User Interface for CV pipeline demonstration
├── app.py # Core CV pipeline application logic used by GUI
├── CONST.py # Shared constants (e.g., paths)
└── README.md # This file
```

## Installation

**Prerequisites:**

*   Python 3.12+
*   Git
*   `pip` and `venv`
*   (Recommended) NVIDIA GPU with CUDA support for accelerated CV/ML tasks.
*   (Planned) ROS2 Installation (e.g., Humble Hawksbill on Ubuntu 22.04)

**Setup:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/VanniLeonardo/Prosthetic-Arm 
    cd Prosthetic-Arm
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv_prosthetic
    source venv_prosthetic/bin/activate  # On Linux/macOS
    # venv_prosthetic\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    *   Install core and computer vision dependencies:
        ```bash
        pip install -r requirements.txt # Still to be fully setup
        ```
    *   *(Pending)* Installation steps for Neuroscience dependencies will be added here.
    *   *(Pending)* Installation steps for Robotics/ROS2 dependencies will be added here.

4.  **Download Models:**
    *   The required computer vision models (YOLO, SAM, Depth, Hand Landmark) should be placed in `computer_vision/models/`. Please refer to `computer_vision/README.md` for details on obtaining these models. 
    *   *(Pending)* Steps for obtaining EEG classification models will be added here.

5.  **Hardware Setup (For Full System - Currently CV Demo Only):**
    *   Connect a webcam for the computer vision demo.
    *   *(Pending)* Connect EEG Device, Prosthetic Arm, Sensors, and configure Raspberry Pi.

## Usage

**1. Running the Computer Vision GUI Demo:**

This demo showcases the real-time object detection, depth estimation, segmentation, grasp validation, and hand tracking capabilities of the CV module.

```bash
python GUI.py
```

Use the GUI to select the video source (camera index or file path), configure model sizes and features, then click "Start Pipeline". The different processing stage outputs will be displayed.

###  [Link to CV GUI Demo Video]

**2. Running the Integrated System:**

(Instructions for running the fully integrated system involving EEG, CV, and Robotics will be added here once those modules are implemented and integrated.)

## Current Status & Roadmap

    ✅ = Completed, ⏳ = In Progress, ⬜ = Planned

    ✅ Computer Vision:

        ✅ Object Detection (YOLO)

        ✅ Depth Estimation (DepthAnythingV2)

        ✅ Segmentation (SAM)

        ✅ Hand Landmark Tracking (MediaPipe)

        ✅ 3D Bounding Box Estimation & Visualization

        ✅ Geometric & Hybrid Grasp Validation Logic

        ✅ Asynchronous GUI for Pipeline Demonstration

        ⬜ Grasp Planning (Determining how to grasp, not just if)

    ⏳ Neuroscience:

        ⏳ EEG Signal Acquisition Setup (Hardware TBD)

        ⏳ Signal Preprocessing & Filtering

        ⏳ Feature Extraction Methods

        ⏳ Intent Recognition Model Training (e.g., EEGNet)

        ⬜ Real-time Inference Capability

    ⬜ Robotics:

        ⏳ Custom Arm Hardware Construction

        ⬜ Hardware Interface & Driver Development

        ⬜ Low-Level Joint Control Implementation

        ⬜ Inverse Kinematics / Motion Planning

        ⬜ Grasp Primitive Execution Library

    ⬜ Sensors:

        ⬜ Sensor Selection (Force, Temperature)

        ⬜ Sensor Integration & Calibration

        ⬜ Feedback Loop Implementation for Control

    ⬜ System Integration:

        ⬜ ROS2 Communication Backbone Setup

        ⬜ Central Control Logic on Raspberry Pi

        ⬜ Integration of all Modules

## Contributors
- **Leonardo Vanni**  - [leonardo.vanni@studbocconi.it](mailto:leonardo.vanni@studbocconi.it)
- **Anna Notaro** - [anna.notaro@studbocconi.it](mailto:anna.notaro@studbocconi.it)
- **Gianluca Fugante**  - [gianlucafugante@gmail.com](mailto:gianlucafugante@gmail.com)

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.