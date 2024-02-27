import sys  # Used for accessing system-specific parameters and functions
import numpy as np  # Fundamental package for scientific computing with Python
import cv2  # OpenCV library for computer vision tasks
from PyQt5 import QtWidgets, QtGui, QtCore  # Import PyQt5 for GUI application development
from PyQt5.QtCore import pyqtSignal, QThread  # Import specific modules from PyQt5 for threading and signals
from ultralytics import YOLO  # Import YOLO object detection model (ensure the correct path)
from mss import mss  # Import mss for taking screenshots
import time  # Import time library for time-related functions
import pyautogui  # Import pyautogui for GUI automation tasks

# Configurable Parameters
MODEL_PATH = 'yolov8m.pt'  # Path to the YOLO model file for object detection [yolov8n, yolov8s, yolov8m, yolov8l, yolov8x]
UPDATE_INTERVAL = 0.1  # Time in seconds between each update/detection in the detection thread
OBJECT_IDS = [0]  # List of object IDs that YOLO model should detect (IDs can be found at coco.names
SCREENSHOT_PAUSE = 0.02 # Time in seconds before taking a screnshot to clear the bounding boxes

net = YOLO(MODEL_PATH) # Initialize the YOLO model with the specified model file

# DetectionThread is a custom QThread for handling object detection in a separate thread
class DetectionThread(QThread):
    update_signal = pyqtSignal(np.ndarray)  # This signal is emitted with the detection results to update the GUI

    def __init__(self, update_interval=UPDATE_INTERVAL):
        super().__init__()
        self.update_interval = update_interval  # How often to perform detection (in seconds)
        self.last_update_time = time.time()  # Timestamp of the last update, used to maintain the update interval
        self.detections = None  # Stores the latest detection results

    def run(self):
        # Continuous loop to perform detections at specified intervals
        while True:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                image = self.capture_screen_mss()  # Capture the current screen
                self.detections = self.detect_objects(image, OBJECT_IDS)  # Detect specified objects in the captured image
                # Emit signal with detection results (or an empty array if no detections)
                self.update_signal.emit(self.detections if self.detections is not None else np.array([]))
                self.last_update_time = current_time  # Update the timestamp of the last update

    def detect_objects(self, image, object_ids):
        # Perform object detection on the image using YOLO
        outs = net(image)[0]  # Run the detection model on the image
        # Filter detections to only those with IDs in the specified object_ids list
        detected_objects_idxs = np.isin(outs.boxes.cls, object_ids)
        if detected_objects_idxs.any():
            # Extract and return the bounding boxes of detected objects
            xyxys = outs.boxes.xyxy[detected_objects_idxs] / 2
            return xyxys.numpy().astype(int)
        return None

    def capture_screen_mss(self):
        # Capture the current screen using the mss library
        with mss() as sct:
            monitor = sct.monitors[1]  # Select the primary monitor
            self.update_signal.emit(np.array([]))  # Temporarily clear detections for capturing
            time.sleep(SCREENSHOT_PAUSE)  # Short pause to ensure the screen is captured without overlays
            sct_img = sct.grab(monitor)  # Grab the screenshot
            # Re-emit the previous detections if they exist
            if self.detections is not None:
                self.update_signal.emit(self.detections)
            img = np.array(sct_img)  # Convert the screenshot to a numpy array
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Convert the image to RGB format
            return img

# TransparentOverlay is a QWidget that creates a transparent overlay window for drawing detections
class TransparentOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.detected_objects = np.array([])  # Initialize an empty array to store detected objects
        self.initUI()  # Initialize the UI components

        self.det_thread = DetectionThread()  # Create a DetectionThread instance
        self.det_thread.update_signal.connect(self.update_detected_objects)  # Connect the signal to update detected objects
        self.det_thread.start()  # Start the detection thread

    def initUI(self):
        # Set up the UI for the transparent overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # Make the window background translucent
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # Make the window ignore mouse events
        # Set the geometry of the window to cover the entire screen
        self.setGeometry(0, 0, pyautogui.size().width, pyautogui.size().height)
        self.show()  # Show the window

    def paintEvent(self, event):
        # Handler for painting/drawing on the overlay
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor("red"), 2))  # Set the pen color and width for drawing
        qp.setBrush(QtGui.QColor(0, 0, 0, 255))  # Set the brush color
        # Draw rectangles around detected objects
        for obj in self.detected_objects:
            qp.drawRect(obj[0], obj[1], obj[2] - obj[0], obj[3] - obj[1])

    def update_detected_objects(self, objects):
        # Update the list of detected objects and trigger a repaint
        self.detected_objects = objects
        self.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # Create a QApplication instance
    overlay = TransparentOverlay()  # Create the TransparentOverlay widget
    sys.exit(app.exec_())  # Start the application's event loop
