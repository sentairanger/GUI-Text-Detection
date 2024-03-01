import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import matplotlib
from picamera2 import Picamera2
from time import sleep
from datetime import datetime

# Define timestamp
timestamp = datetime.now().isoformat()

matplotlib.use('GTK3Agg')

ie = Core()

model = ie.read_model(model="model/horizontal-text-detection-0001.xml")
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output("boxes")

# initiate the camera and define the location of the images
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
image_sample = "data/text.jpg"

# Take picture
def capture():
    picam2.start()
    print("image processing")
    sleep(3)
    picam2.capture_file(image_sample)
    picam2.stop()
    print("success")
        
                
# Text detection models expects image in BGR format
def image_input():
    image = cv2.imread(image_sample)
    return image

# N,C,H,W = batch size, number of channels, height, width
N, C, H, W = input_layer_ir.shape

# Resize image to meet network expected input sizes
def resize_image():
    resized_image = cv2.resize(image_input(), (W, H))
    return resized_image

# Reshape to network input shape
def reshape_image():
    input_image = np.expand_dims(resize_image().transpose(2, 0, 1), 0)
    return input_image

# Create inference request
def box_detect():
    boxes = compiled_model([reshape_image()])[output_layer_ir]
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    return boxes

# For each detection, the description has the format: [x_min, y_min, x_max, y_max, conf]
# Image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib we use cvtColor function
def convert_result(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
    # Define colors for boxes and descriptions
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Fetch image shapes to calculate ratio
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert base image from bgr to rgb format
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Iterate through non-zero boxes
    for box in boxes:
        # Pick confidence factor from last place in array
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio
            # In case that bounding box is found at the top of the image,
            # we position upper box bar little lower to make it visible on image
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            # Draw box based on position, parameters in rectangle function are: image, start_point, end_point, color, thickness
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # Add text to image based on position and confidence
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image

def show_image(raw_image):
    print("rendering image")
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(raw_image);
    plt.savefig("data/image_%s.jpg" % timestamp)
    print("success")

