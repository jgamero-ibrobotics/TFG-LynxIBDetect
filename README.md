# TFG-LynxIBDetect

![Project Logo](/figuras/figura1.png)

TFG-LynxIBDetect is a project developed as part of a Final Degree Project (TFG) by Jesús Gamero. The project focuses on the detection of the Iberian Lynx, an endangered species native to the Iberian Peninsula, using a camera and advanced computer vision techniques.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/jgamero-ibrobotics/TFG-LynxIBDetect.git 
```


2. Install virtualenv:

```bash
sudo pip3 install virtualenv 
```


3. Activate the environment by issuing:

```bash
source tflite1-env/bin/activate
```

# Install TensorFlow Lite dependencies and OpenCV

1. This shell script will automatically download and install all the packages and dependencies:

```bash
bash get_pi_requirements.sh
```

# Run the TensorFlow Lite model

1. Runing the main script that simulates the camera trap behaviour:

```bash
python3 captura_com.py --modeldir=LynxDetectv2
```
