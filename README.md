# video-difference-tracker

This project implements a simple video motion detection system using Python and the OpenCV library. It identifies moving objects by comparing video frames against an initial background frame, refines the detection using morphological operations, and visualizes the results by drawing contours around moving objects and adding a trailing effect to the output video.

## Features

* **Background Subtraction**: Uses the first frame of the video as a static background to compute the difference with subsequent frames.
* **Image Preprocessing**: Applies grayscale conversion, Gaussian blur, and histogram equalization to enhance detection accuracy.
* **Motion Area Extraction**: Generates a foreground mask by thresholding the difference image.
* **Region of Interest (ROI)**: Allows defining a specific area for detection, ignoring changes outside this region.
* **Noise Reduction**: Employs morphological opening and closing operations to remove noise and small holes in the binary mask.
* **Contour Detection and Drawing**: Finds contours in the foreground mask and draws polygonal bounding boxes around them on the original video frame.
* **Trailing Effect**: Adds a visual trail to detected motion areas to show movement history.
* **Real-time Display**: Shows both the processed video output (with bounding boxes and trails) and the foreground mask simultaneously.

## Prerequisites

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy (`numpy`)

## Installation

1.  Clone this repository (if you haven't already):
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  Install the required Python libraries. Using `requirements.txt` (if you created one) is recommended:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install them manually:
    ```bash
    pip install opencv-python numpy
    ```

## Usage

1.  **Prepare Video File**: Place the video file you want to analyze in the project directory. By default, the script looks for a file named `11.mp4`.
    * **Important**: If your video file has a different name or path, modify the `cv2.VideoCapture()` line in the script:
        ```python
        # Open the video file
        # Replace "11.mp4" with the path to your video file
        cap = cv2.VideoCapture(r"your_video.mp4")
        ```
2.  **Run the Script**: Execute the Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual filename you saved the code as, e.g., `motion_detector.py`)
3.  **View Results**: The program will open two windows:
    * `Difference Detection with Trailing Effect`: Shows the original video stream with green bounding boxes around detected moving objects and the trailing effect.
    * `Foreground Mask`: Displays the processed binary foreground mask.
4.  **Exit Program**: Press the `q` key on your keyboard to close all windows and stop the script.

## Configuration

You can adjust several parameters directly within the Python script to fine-tune the detection behavior:

* **Video Source**: `cv2.VideoCapture(r"11.mp4")` - Change the file path.
* **Gaussian Blur Kernel Size**: `cv2.GaussianBlur(..., (5, 5), 0)` - Adjust `(5, 5)` to change the amount of blur.
* **Difference Threshold**: `cv2.threshold(diff_frame, 30, 255, ...)` - `30` is the threshold for distinguishing foreground/background; adjust based on lighting and scene.
* **Region of Interest (ROI)**: `cv2.rectangle(roi_mask, (0, 200), (w - 100, h), 255, -1)` - Modify the coordinates `(0, 200)` and `(w - 100, h)` to change the detection area.
* **Morphological Kernel Sizes**: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))` and `(7, 7)` - Control the intensity of the opening/closing operations.
* **Trailing Effect Duration**: `trailing_decay = 255 // (frame_rate * 5)` - The `5` represents roughly 5 seconds duration for the trail; modify this value as needed.
* **Minimum Contour Area**: `if area < 200:` - `200` is the threshold to filter out small contours; adjust based on the expected size of objects to detect.
