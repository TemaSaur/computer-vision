# Image Recognition

The program takes a video stream as input and outputs a log of found object

The recognized objects include:

* transport vehicle
  * The type of the vehicle
  * vehicle plate number
  * plate number text
* human
* animal

The log contains the list of found object, their type and number

Technologies:

* python
* opencv
* yolo8

## Usage

Setting up the virtual environment

	python -m venv venv

	# For windows
	venv\scripts\Activate

	# For linux, macos
	source venv/bin/activate

Installing dependencies:

	pip install -r requirements.txt

Running the program:

	python main.py
