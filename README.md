# TFG-LynxIBDetect
![Project Logo](/figuras/figura.gif)

TFG-LynxIBDetect is a project developed as part of a Final Degree Project (TFG) by Jesús Gamero. The project focuses on the detection of the Iberian Lynx, an endangered species native to the Iberian Peninsula, using a camera and advanced computer vision techniques.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/jgamero-ibrobotics/TFG-LynxIBDetect.git 
```

```bash
cd TFG-LynxIBDetect
```

2. Install virtualenv and create a virtual environment:

```bash
sudo pip3 install virtualenv 
```
Then, create the "tflite1-env" virtual environment by issuing:

```bash
python3 -m venv tflite1-env
```

3. Activate the environment by issuing:

```bash
source tflite1-env/bin/activate
```
A virtual environment is an isolated environment that allows you to install Python packages and their dependencies without affecting other projects or the global Python environment on your system.

## Install TensorFlow Lite dependencies and OpenCV

This shell script will automatically download and install all the packages and dependencies:

```bash
bash get_pi_requirements.sh
```

## Run the TensorFlow Lite model

Runing the main script that simulates the camera trap behaviour:

```bash
python3 Camara_trampa.py --modeldir=LynxDetectv2
```
