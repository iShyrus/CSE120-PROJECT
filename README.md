# Can Damage Detection
Vision system for detecting can height and lid seal quality in San Benito Neil Jones Food Company's tomato canning process

## Overview
This project is being developed by students from the University of California, Merced as a tool for the San Benito Neil Jones Food Company to improve the quality and efficiency of their tomato canning process. The vision system uses Python, OpenCV, PyQt, and YOLO to detect cans on a conveyor belt, and classify them as a damaged or undamaged can.

## Requirements
Python 3.6 or later
OpenCV 4.0 or later
3 cameras connected to the computer running the software
A conveyor belt with cans passing in front of the cameras

## Installation
Clone the repository:
<pre><code>git clone https://github.com/iShyrus/CSE120-PROJECT.git
</code></pre>

Install the required Python packages:
<pre><code>pip install -r requirements.txt
</code></pre>

Connect the cameras to the computer and make sure it is recognized by OpenCV.

<pre><code>python main.py
</code></pre>

## Usage
When you run the software, the interface will have a button to the left of the screen to start the cameras and object detection.
When turned on it will enable all the cameras and begin capturing images to analyze them. The app defaults to TinyYOLO which uses solely CPU. You may select the higher performance option that ultilizes a GPU by selecting the YoloV8 button on the upper left corner.

The detected cans will be marked with a bounding box along with a label of their classication. If a can is detected with any damage, a warning message will be displayed on top of the screen.

The images below demonstrate the application when detecting a good and bad can.

<img src="https://hackmd.io/_uploads/Hkz_ctuVh.png)
" width=600><br>

<img src="https://hackmd.io/_uploads/Skg3cFuNh.png)
" width=600><br>

A CSV file will be generated if one does not already exist in the directory. The CSV file should make a row for every object detected with the can's classification and the timestamp.
