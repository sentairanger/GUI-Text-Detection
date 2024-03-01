# import libraries
from guizero import App, PushButton, Drawing
from detect_text import *

# Define functions for text detection and show the result
def text_detection():
    text_image = show_image(convert_result(image_input(), resize_image(), box_detect(), conf_labels=False))
    return text_image

def show_text():
    viewer.image(20, 10, "data/image_%s.jpg" % timestamp)

# Define the main app, buttons and viewer
app = App(title="Text Detection GUI")
button = PushButton(app, text="Take Picture", command=capture)
button2 = PushButton(app, text="Detect Text", command=text_detection)
button3 = PushButton(app, text="Display Text", command=show_text)
viewer = Drawing(app, width="fill", height="fill")
app.display()
