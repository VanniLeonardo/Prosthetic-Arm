import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
CV_PATH = os.path.join(ROOT_DIR, "computer_vision")
ROBOTICS_PATH = os.path.join(ROOT_DIR, "robotics")
NEUROSCIENCE_PATH = os.path.join(ROOT_DIR, "neuroscience")

MIRRORED_CAMERA = True
CAMERA_TYPE = 0
CAMERA_REVERSE = True