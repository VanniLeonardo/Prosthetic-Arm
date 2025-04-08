# Intelligent Prosthetic Arm Project - Computer Vision Module

## Overview

This document outlines the setup and installation steps for the Computer Vision module of the Intelligent Prosthetic Arm project, including the necessary software and dependencies.

## Setup

### Requirements

-   **Python**: 3.12.0

## Installing CUDA

To ensure that the CUDA setup is configured correctly for the project, follow the steps below:

### 1. Install Visual Studio

-   Download and install **Visual Studio Community** from [here](https://visualstudio.microsoft.com/downloads/).
-   During installation, select the **Desktop development with C++** workload.
-   Restart your computer after installation is complete.

### 2. Install CUDA Toolkit

-   Download **CUDA Toolkit 12.6** from [NVIDIA's website](https://developer.nvidia.com/cuda-12-6-0-download-archive).
-   Follow the installation prompts to complete the installation.

### 3. Install cuDNN

-   Download **cuDNN v8.9.7 for CUDA 12.x** from the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive).
-   Select the version that matches your operating system.
-   After downloading, open the CUDA folder (usually located at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`).
-   Extract the cuDNN archive files into the CUDA folder.

## Check if CUDA is Installed

-   Open the command line (Terminal on macOS/Linux, Command Prompt on Windows).
-   Type the following command and press Enter:

    ```bash
    nvcc --version
    ```

    If CUDA is installed correctly, this command will return the version of CUDA installed on your system. If you see an error message, CUDA might not be installed properly.

## Setting Up the Virtual Environment

1. **Clone the GitHub Repository**  
   Clone the repository containing the project files onto your local machine.

2. **Create a Virtual Environment**  
   Navigate to the project folder and create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**  
   To activate the virtual environment:

    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install Dependencies**  
   Navigate to the `computer_vision` folder and install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. **Install CUDA-compatible PyTorch**  
   Install the version of PyTorch that is compatible with CUDA 12.6 by running the following command:

    ```bash
    pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    ```

## Verify CUDA with PyTorch

-   To check if CUDA is correctly recognized by PyTorch, run the following Python code:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

    If the result is `True`, it means PyTorch is able to access CUDA and your GPU is ready for use. If it returns `False`, there might be an issue with your CUDA setup.
