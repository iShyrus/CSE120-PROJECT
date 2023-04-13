# Can Damage Detection
Vision system for detecting can height and lid seal quality in San Benito Neil Jones Food Company's tomato canning process

## Overview
This project is being developed by students from the University of California, Merced as a tool for the San Benito Neil Jones Food Company to improve the quality and efficiency of their tomato canning process. The vision system uses Python and OpenCV to detect cans on a conveyor belt, measure their height, and analyze the lid for any gaps or improper seals.

## Requirements
Python 3.6 or later
OpenCV 4.0 or later
3 cameras connected to the computer running the software
A conveyor belt with cans passing in front of the camera

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
When you run the software, the main interface will have a power button to start the cameras and begin capturing images and analyzing them. The detected cans will be marked with a bounding box, and their height and lid seal quality will be displayed on the screen. If a can is detected with an improper seal, a warning message will be displayed and an appropriate descrete variable will be triggered. The discrete variable can be tied to a mechanical output to physically reject a can as needed.

This project is licensed under the MIT License - see the LICENSE.md file for details.
