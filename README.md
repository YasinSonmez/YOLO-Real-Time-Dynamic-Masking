# Real-Time Screen Object Detection and Masking

This project implements a real-time object detection system that overlays bounding boxes around detected objects on your screen. It utilizes the YOLOv8 (You Only Look Once) model for object detection, combined with a PyQt5 application for the overlay, to highlight objects of interest directly on your monitor.

## Key Features

- **Real-Time Detection**: Dynamically detects objects on the screen and updates bounding boxes in real-time.
- **Transparent Overlay**: Uses a transparent overlay to display the bounding boxes, ensuring minimal interference with user activities.
- **Customizable Object IDs**: Users can specify which objects the model should detect by configuring the object IDs according to the YOLO model's output.

Here is an example of masking only the detected humans in a YouTube video opened on the screen in real-time. Flickering is due to clearing the bounding boxes for taking screenshots and redrawing them (TODO: Faster screenshot to reduce flickering)

https://github.com/YasinSonmez/YOLO-Real-Time-Dynamic-Masking/assets/37510173/e9a28a7a-b627-4816-ba2b-ce08fc1c379c


## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed on your machine. This project also requires a YOLO model file (`yolov8s.pt` or similar) that must be obtained separately and placed in an accessible directory.

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/YasinSonmez/YOLO-Real-Time-Dynamic-Masking.git
cd YOLO-Real-Time-Dynamic-Masking
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```
You don't have to install the model parameters manually, the program will download it at the beginning.

### Running the Application

Execute the main script to start the object detection overlay:

```bash
python object_masker.py
```

Adjust the `MODEL_PATH`, `UPDATE_INTERVAL`, and `OBJECT_IDS` variables in the script as needed to configure the detection settings and model path.

## Configuration Options
Parameters are at the beginning of the object_masker.py file
```
# Configurable Parameters
MODEL_PATH = 'yolov8m.pt'  # Path to the YOLO model file for object detection [yolov8n, yolov8s, yolov8m, yolov8l, yolov8x]
UPDATE_INTERVAL = 0.1  # Time in seconds between each update/detection in the detection thread
OBJECT_IDS = [0]  # List of object IDs that YOLO model should detect (IDs can be found at coco.names)
SCREENSHOT_PAUSE = 0.02 # Time in seconds before taking a screnshot to clear the bounding boxes
```

## Dependencies

This project leverages the following major libraries:

- `opencv-python`: For image processing tasks.
- `PyQt5`: For creating the GUI overlay.
- `ultralytics`: For utilizing the YOLO model.
- `mss`: For capturing screen content.
- `pyautogui`: For obtaining screen dimensions.

## TODO
- [ ] Find a faster way to screenshot so that the flickering is reduced
