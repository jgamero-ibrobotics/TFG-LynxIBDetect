import os
import dropbox
import importlib.util
import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def read_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    references = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        references.append([obj.find('name').text,xmin, ymin, xmax, ymax])
 
    return references

def apply_detection_model(image_path):
    # Cargar la imagen con OpenCV
    image = cv2.imread(image_path)
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

    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax]) # Add detection data to list

    return detections

def calculate_iou(boxA, boxB):
    # Obtener las coordenadas de los rectángulos
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])


    # Calcular el área de la intersección
    intersection_area = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # Calcular el área de los rectángulos
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calcular el coeficiente de IoU
    iou = intersection_area / (boxA_area + boxB_area - intersection_area)

    return iou


########################CODIGO PRINCIPAL########################

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


# Configuración de Dropbox
access_token = 'sl.BeOmu_xGFxNh_V_wriQuorlkFNxFMFLd1PrqhtYotKMJjcGCBpjMlSw4-yHMO7Js8ZKpAgTySW1mYTULb7LYTiPSepJQ94-YxqZlVSp2RYP6gt-RoujZmJ7KbpE70QFe1o-ZMEgFECA'
dropbox_folder = '/test'

# Directorio local para descargar los archivos
local_folder = '/home/pi/TFG-LynxIBDetect/IoU'

# Crear la carpeta local si no existe
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# Crear una instancia del cliente de Dropbox
client = dropbox.Dropbox(access_token)

# Obtener la lista de archivos en la carpeta de Dropbox
file_list = client.files_list_folder(dropbox_folder).entries


for nombre_archivo in os.listdir("/home/pi/TFG-LynxIBDetect/IoU"):
    ruta_archivo = os.path.join("/home/pi/TFG-LynxIBDetect/IoU", nombre_archivo)
    if os.path.isfile(ruta_archivo):
        os.remove(ruta_archivo)

j = 0
k = 0
iou_total = 0
Tp = 0
Fp = 0
Fn = 0
for file_entry in file_list:
    # Obtener el nombre del archivo
    file_name = file_entry.name

    # Verificar si es un archivo XML
    if file_name.endswith('.xml'):
        # Descargar el archivo XML desde Dropbox
        xml_path = os.path.join(local_folder, file_name)
        client.files_download_to_file(xml_path, f'{dropbox_folder}/{file_name}')

        # Leer las coordenadas de boxA desde el archivo XML
        references = read_xml_file(xml_path)
  
        # Obtener el nombre del archivo de imagen relacionado
        image_file = file_name.replace('.xml', '.jpg')

        # Descargar el archivo JPG desde Dropbox
        image_path = os.path.join(local_folder, image_file)
        client.files_download_to_file(image_path, f'{dropbox_folder}/{image_file}')
        # Aplicar el modelo de detección y obtener las predicciones de boxB
        detections = apply_detection_model(image_path)
        image = cv2.imread(image_path)

        # Elimina el archivo local
        os.remove(image_path)
        os.remove(xml_path)

        iou_scores = [] 
        for reference in references: # recorre cada box en el archivo xml
            boxA = [reference[1],  reference[2],  reference[3],  reference[4]] # [xmin, ymin, xmax, ymax]
            clase_ref = reference[0]
            print('box Referencia')
            print(boxA)
            iou_scores_box = []
            cv2.rectangle(image, (boxA[0],  boxA[1]), (boxA[2],  boxA[3]), (0, 255, 0), 2) # ground truth box
            if detections == []: # si no hay detecciones
                for i in range(len(boxA)):
                    detections.append(["NONE", 0, 0, 0, 0, 0])
            for detection in detections: # recorre cada box por objeto detectado
                boxB = [detection[2],  detection[3],  detection[4],  detection[5]] # [xmin, ymin, xmax, ymax]
                clase_det = detection[0]
                cv2.rectangle(image, (boxB[0],  boxB[1]), (boxB[2],  boxB[3]), (0, 0, 255), 2) # detection box
                print('box Detección')
                print(boxB)

                # Calcular el coeficiente de IoU
                iou_score = calculate_iou(boxA, boxB)
                iou_scores_box.append(iou_score)
            max_iou_score = max(iou_scores_box)

            if max_iou_score > 0.7 and clase_ref == clase_det:
                Tp = Tp + 1
            elif clase_ref == clase_det and max_iou_score < 0.7:
                Fp = Fp + 1
            elif max_iou_score > 0.7 and clase_ref != clase_det:
                Fn = Fn + 1 

            iou_scores.append(max_iou_score)
            k = k + 1

        file_name = f'/home/pi/TFG-LynxIBDetect/IoU/image_IoU_{j+1}.jpg'
        j=j+1
        #cv2.imwrite(file_name,image)  # Guardar imagen en un archivo
        iou_total = sum(iou_scores) + iou_total

        #print(f"Archivo XML: {file_name}")
        print("El coeficiente de IoU es:", iou_scores)
        print("---------------------------------------------")

print("   ")
print("--------------RESULTADOS FINALES--------------")
print("Verdaderos Positivos:", Tp)
print("Falsos Positivos:", Fp)
print("Falsos Negativos:", Fn)
precision = Tp/(Tp+Fp)
recall = Tp/(Tp+Fn)
f1_score = 2*((precision*recall)/(precision+recall))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)