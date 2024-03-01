# GUI-Text-Detection
This project uses Guizero to allow the user to take a picture with a Pi Camera Module and detect text in the image using OpenVINO's text detection model

## Getting Started

To get started, first make sure to have a Camera Module (any will suffice), and a Pi. For the purposes of this project, it's best to use either a Pi 4 or 3B+. This will be tested on the Pi 5 once I get my hands on one. You will also need the Intel NCS2 which can be found on various sites like eBay. Optionally a camera mount is recommended. Next, you will need to install OpenVINO. This has been tested on 64-bit Pi OS Bullseye but will be tested on Bookworm. Next, follow this updated [guide](https://gist.github.com/sentairanger/caf11a2432ceebd715c6b33c224f4960) to install OpenVINO. Note that this only works with a Desktop so it will be best to use this either with a monitor or enable VNC. If using VNC be sure to access the Pi using a phone or any other device that has VNC Viewer installed. Next, run the main code with `python3 text-gui.py`. The GUI should appear and then capture an image, detect the text and display the final result. A sample pic is shown below.

![picture](https://github.com/sentairanger/GUI-Text-Detection/blob/main/text-gui.png)
