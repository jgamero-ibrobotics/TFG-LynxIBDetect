# -*- coding: utf-8 -*-

# Import packages
import os
import argparse
import cv2
import time
import numpy as np
import sys
import glob
import importlib.util
import dropbox
import threading
import queue
from dropbox.exceptions import ApiError


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',
                    action='store_true')
parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)',
                    action='store_false')

args = parser.parse_args()


# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)

save_results = args.save_results # Defaults to False
show_results = args.noshow_results # Defaults to True

IM_NAME = args.image
IM_DIR = args.imagedir
CWD_PATH = os.getcwd() # Get path to current working directory
RESULTS_DIR ='results' # Folder to save results images and data to


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
from tflite_runtime.interpreter import Interpreter

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=PATH_TO_CKPT) # Cargar el modelo
interpreter.allocate_tensors() # Asignar tensores

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32) 

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


# Obtén un token de acceso válido para utilizar la API de Dropbox
TOKEN = 'sl.Bfb4XTOmCE2i0tqE3KgT4__IcvNqCyqY1k55oU7jUNapU3LmKvJsSl9xdkeWoK8rPFX9m6kD4e7aJOwhD0962lCRDzy2ZDCvXN6RES_kLoMZyZW1lXNQd6tExNWW-cqomzoAb39bpBk'
# Crea una instancia del cliente de Dropbox
dbx = dropbox.Dropbox(TOKEN)
# Ruta al directorio en Dropbox donde se guardarán las imágenes y los resultados
remote_directory = '/Detections'
#dbx.files_delete(remote_directory)
try:
    dbx.files_get_metadata(remote_directory)
    dbx.files_delete(remote_directory)
    # print(f"Directorio {remote_directory} eliminado correctamente")
except dropbox.exceptions.ApiError as e:
    error = e.error
    if isinstance(error, dropbox.files.GetMetadataError) and error.is_path():
        if error.get_path().is_not_found():
            print(f"El directorio {remote_directory} no existe")
        else:
            print(f"Error al obtener metadatos del directorio {remote_directory}: {error}")
    else:
        print(f"Error desconocido al eliminar el directorio {remote_directory}: {e}")


def hilo(): 
    while True:
        print(f"\nHILO1: Presiona la tecla 'Enter' para capturar imagenes")
        input()  # Esperar a que se presione la tecla Enter
        capture_images(10)

imagen_queue = queue.Queue(maxsize=0)
mutex = threading.Lock()
capture_thread = threading.Thread(
        target=hilo, args=()
    )
capture_thread.start()


# Función para capturar una imagen de la cámara web
def capture_image():
    cap = cv2.VideoCapture(0)  # Abrir la cámara web
    ret, frame = cap.read()  # Capturar un frame de video
    cap.release()  # Liberar la cámara

    return frame

# Función para guardar la imagen en un archivo
def save_image(image, file_name):
    cv2.imwrite(file_name, image)

# Capturar imágenes después de pulsar una tecla y esperar 10 segundos entre cada captura
def capture_images(num_images):
    for i in range(num_images):
        if i > 0:
            print("HILO1: Esperando 0.5 segundos...") 
            time.sleep(0.5)
        image = capture_image()  # Capturar imagen de la cámara web 
        imagen_queue.put(image,timeout=None)

def process_images():
    global j
    image = imagen_queue.get(block=True, timeout=None)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    detections = []
        # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
                
    # Get filenames and paths
    image_fn = os.path.basename("/home/pi/TFG-LynxIBDetect")
    image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
    
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn +'.txt'
    txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)
    if not detections==[]: # Si hay detecciones subo. FILTRO
        # Save image with bounding boxes and labels
        file_name = f'/home/pi/TFG-LynxIBDetect/Detections/image{j+1}.jpg'
        save_image(image, file_name)  # Guardar imagen en un archivo
        # Sube la imagen a Dropbox
        with open(file_name, 'rb') as f:
            remote_filename = f'{remote_directory}/image{j+1}.jpg'
            #print(remote_filename)
            dbx.files_upload(f.read(), remote_filename)
        # Elimina el archivo local
        os.remove(file_name)
        # Guarda los resultados de detección en un archivo
        # (asumiendo que los resultados están en la variable 'detections')
        detections_env = []
        detections_env.append('\n'.join(map(str, detections)))
        results = ', '.join(detections_env) 
        results_filename = f'/home/pi/TFG-LynxIBDetect/Detections/results_image{j+1}.txt'
        with open(results_filename, 'w') as f:
            f.write(results)
        # Sube el archivo de resultados a Dropbox
        with open(results_filename, 'rb') as f:
            remote_results_filename = f'{remote_directory}/results_image{j+1}.txt'
            dbx.files_upload(f.read(), remote_results_filename)
        # Elimina el archivo local
        os.remove(results_filename)
        j=1+j

# Llamar a la función para capturar una imagen después de pulsar una tecla y esperar 10 segundos entre cada captura
j=0
while(True):
    process_images()


