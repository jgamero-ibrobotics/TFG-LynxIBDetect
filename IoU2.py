import os
import dropbox
import importlib.util
import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def elimina_no_maximos(matriz, col, indice_preservar):
  
    for i in range(len(matriz[:,col])):
        if i != indice_preservar:
            matriz[i,col] = 0
    return matriz

def darken_if_dark_image(image, threshold_brightness, darkness_factor):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)

    if brightness < threshold_brightness:
        darkened_image = np.copy(image)
        # 1 es más brillante, 0 es más oscuro
        darkened_image = darkened_image.astype(np.float32) * darkness_factor # Multiplicar por un factor de oscuridad
        darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8) # Limitar los valores entre 0 y 255
        return darkened_image
    else:
        return image

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

def apply_detection_model(image):
    # Cargar la imagen con OpenCV
    # image = cv2.imread(image_path)
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
parser.add_argument('--darken_image', help='Darken image for testing in low light conditions',
                    action='store_true')
parser.add_argument('--archivo_resultados', help='Nombre del archivo donde se guardan los resultados',
                    default='resultados.txt')


args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
darken_image = args.darken_image
archivo_resultados = args.archivo_resultados


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
access_token = 'sl.BfGY_Op3_kejbNqPlHpyfbDW3GHT5CFayg6T8zIXw43La1fSI0I55RRlzWV1R-L1bE7Kz3pwo8YeAQ3JzgEHOwwNk4BokSkHqczDqV8ytIlWW25rN9M7KSp89Evwm-CXx9XGJgcFDCc'
dropbox_folder_xml = '/pruebaXML'
dropbox_folder_jpg = '/pruebaJPG'
# dropbox_folder_xml = '/prueba _lince_pocaluzXML'
# dropbox_folder_jpg = '/prueba _lince_pocaluzJPG'

# Directorio local para descargar los archivos
local_folder = '/home/pi/TFG-LynxIBDetect/IoU'

# Crear la carpeta local si no existe
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# Crear una instancia del cliente de Dropbox
client = dropbox.Dropbox(access_token)

# Obtener la lista de archivos en la carpeta de Dropbox
file_list = client.files_list_folder(dropbox_folder_xml).entries


for nombre_archivo in os.listdir("/home/pi/TFG-LynxIBDetect/IoU"):
    ruta_archivo = os.path.join("/home/pi/TFG-LynxIBDetect/IoU", nombre_archivo)
    if os.path.isfile(ruta_archivo):
        os.remove(ruta_archivo)

j = 0
k = 0
iou_total = 0
resultados = {
    "person-like": {"TP": 0, "FP": 0, "FN": 0},
    "person": {"TP": 0, "FP": 0, "FN": 0},
    "Lynx pardinus": {"TP": 0, "FP": 0, "FN": 0},
    "Fox": {"TP": 0, "FP": 0, "FN": 0},
    "Rabbit": {"TP": 0, "FP": 0, "FN": 0},
    "Cat": {"TP": 0, "FP": 0, "FN": 0},
}
size = len(file_list)
print("Número de archivos:", size)
for file_entry in file_list:
    # Obtener el nombre del archivo
    file_name = file_entry.name
    print("Archivo: ", file_name)

    # Verificar si es un archivo XML
    if file_name.endswith('.xml'):
        # Descargar el archivo XML desde Dropbox
        xml_path = os.path.join(local_folder, file_name)
        client.files_download_to_file(xml_path, f'{dropbox_folder_xml}/{file_name}')

        # Leer las coordenadas de boxA desde el archivo XML
        references = read_xml_file(xml_path)
  
        # Obtener el nombre del archivo de imagen relacionado
        image_file = file_name.replace('.xml', '.jpg')

        # Descargar el archivo JPG desde Dropbox
        image_path = os.path.join(local_folder, image_file)
        client.files_download_to_file(image_path, f'{dropbox_folder_jpg}/{image_file}')
        # Aplicar el modelo de detección y obtener las predicciones de boxB
        image = cv2.imread(image_path)

        if darken_image == True:
            image = darken_if_dark_image(image, 255, 0.15) # 1 0.8 0.6 0.4 0.2
            cv2.imwrite("/home/pi/TFG-LynxIBDetect/IoU/image_darken.jpg", image)

        detections = apply_detection_model(image)
        

        # Elimina el archivo local
        os.remove(image_path)
        os.remove(xml_path)

        iou_scores = []
        max_iou_scores = []
        clases_iou_max = []
        clases_referencia = []

        num_detections = len(detections)
        num_references = len(references)

        print("numero de referencias")    
        print(len(references))
        print("numero de detecciones")
        print(len(detections))

        
        # asigno a cada box de referencia un coeficiente maximo de IoU
        if detections == []: # si no hay detecciones
            for i in range(len(references)):
                detections.append(["NONE", 0, 0, 0, 0, 0])

        elif len(detections) < len(references):
            for i in range(len(references)):
                if i < len(detections):
                    continue
                else:
                    detections.append(["NONE", 0, 0, 0, 0, 0])
 

        matrix_iou = np.zeros((len(detections), len(references)))
        
        col = 0
        for reference in references: # recorre cada box en el archivo xml
            boxA = [reference[1],  reference[2],  reference[3],  reference[4]] # [xmin, ymin, xmax, ymax]
            print('box Referencia')
            print(boxA)
            cv2.rectangle(image, (boxA[0],  boxA[1]), (boxA[2],  boxA[3]), (0, 255, 0), 2) # ground truth box
            fil = 0 
            for detection in detections: # recorre cada box por objeto detectado
                boxB = [detection[2],  detection[3],  detection[4],  detection[5]] # [xmin, ymin, xmax, ymax]
                cv2.rectangle(image, (boxB[0],  boxB[1]), (boxB[2],  boxB[3]), (0, 0, 255), 2) # detection box
                print('box Detección')
                print(boxB)
                # Calcular el coeficiente de IoU
                iou_score = calculate_iou(boxA, boxB)
                matrix_iou[fil][col] = iou_score
                fil = fil + 1
            col = col + 1
        # print(matrix_iou)
        
        iou_max_score = []
        for col in range(num_references):
            column = matrix_iou[:,col]
            iou_max_index_col  = np.argmax(column)
            row = matrix_iou[iou_max_index_col,:]
            iou_max_index_row  = np.argmax(row)
            if [iou_max_index_col, iou_max_index_row] == [iou_max_index_col, col]: # si el maximo IoU de la columna es el mismo que el maximo IoU de la fila
                iou_max_score.append(column[iou_max_index_col])
                clases_referencia.append(references[col][0])
                clases_iou_max.append(detections[iou_max_index_col][0])
                matrix_iou = elimina_no_maximos(matrix_iou, col, iou_max_index_col)
            else:
                while [iou_max_index_col, iou_max_index_row] != [iou_max_index_col, col]:
                    column[iou_max_index_col] = 0
                    iou_max_index_col  = np.argmax(column)
                    
                    row = matrix_iou[iou_max_index_col,:]
                    iou_max_index_row  = np.argmax(row)
                    if np.all(column == 0.0): # no se han producido detecciones para esta referencia
                        break
                matrix_iou = elimina_no_maximos(matrix_iou, col, iou_max_index_col)
                iou_max_score.append(column[iou_max_index_col])
                clases_referencia.append(references[col][0])
                clases_iou_max.append(detections[iou_max_index_col][0])

        print("El coeficiente de IoU es:", iou_max_score)


        # menos detecciones de las esperadas, hay falsos negativos
        # igual numero de detecciones que de referencias, no hay falsos negativos
        if len(references) >= num_detections:
            for i in range(1,len(references)+1):
                if i <= num_detections: # solo contabiliza hasta las detecciones que hay 
                    # tomamos primero las referencias con mayor iou ya que son las que tienen 
                    # mayor probabilidad de ser correctas el resto seran falsos negativos
                    iou_index  = np.argmax(iou_max_score) # indice del maximo coeficiente de IoU
                    iou = iou_max_score[iou_index] # maximo coeficiente de IoU
                    clase = clases_iou_max[iou_index] # clase del objeto detectado con el maximo coeficiente de IoU
                    clase_ref = clases_referencia[iou_index] # clase de la referencia con el maximo coeficiente de IoU
                    
                    clases_iou_max.pop(iou_index)
                    iou_max_score.pop(iou_index)

                    if iou > 0.5 and clase_ref == clase: #Verdadero Positivo
                        resultados[clase_ref]["TP"] += 1
                    elif clase_ref != clase and iou > 0.5: #Falso Positivo
                        resultados[clase_ref]["FP"] += 1
                    else: #Falso Negativo
                        resultados[clase_ref]["FN"] += 1
                # no hay mas detecciones, hay falsos negativos
                else: 
                    # tomamos primero las referencias con mayor iou ya que son las que tienen mayor probabilidad de ser correctas
                    # el resto seran falsos negativos
                    iou_index  = np.argmax(iou_max_score) # indice del maximo coeficiente de IoU
                    clase_ref = references[iou_index][0] # esta mal

                    clases_iou_max.pop(iou_index)
                    iou_max_score.pop(iou_index)

                    resultados[clase_ref]["FN"] += 1 # asigna un falso negativo a la clase de la referencia

        # mas detecciones de las esperadas, hay falsos positivos         
        else:
            for i in range(1,num_detections+1): 
                if i <= len(references): # solo contabiliza hasta las referencias que hay
                    
                    # tomamos primero las referencias con mayor iou ya que son las que tienen
                    # mayor probabilidad de ser correctas el resto seran falsos positivos
                    iou_index  = np.argmax(iou_max_score) # indice del maximo coeficiente de IoU
                    iou = iou_max_score[iou_index] # maximo coeficiente de IoU
                    clase = clases_iou_max[iou_index] # clase del objeto detectado con el maximo coeficiente de IoU
                    clase_ref = clases_referencia[iou_index]


                    clases_iou_max.pop(iou_index)
                    iou_max_score.pop(iou_index)

                    if iou > 0.5 and clase_ref == clase: #Verdadero Positivo
                        resultados[clase_ref]["TP"] += 1
                    elif clase_ref != clase and iou > 0.5: #Falso Positivo
                        resultados[clase_ref]["FP"] += 1
                    else: #Falso Negativo
                        resultados[clase_ref]["FN"] += 1
                
                # no hay mas referencias, hay falsos positivos
                else:
                    for i in range(num_detections):
                        if all(matrix_iou[i,:]) == all(np.zeros(num_references)):
                            resultados[detections[i][0]]["FP"] += 1 # asigna un falso positivo a la clase de la deteccion
        
        k = k + 1

        # # file_name = f'/home/pi/TFG-LynxIBDetect/IoU/image_IoU_{j+1}.jpg'
        file_name = f'/home/pi/TFG-LynxIBDetect/IoU/image_IoU.jpg'
        j=j+1
        cv2.imwrite(file_name,image)  # Guardar imagen en un archivo

        #print(f"Archivo XML: {file_name}")
        print("---------------------------------------------")

# Calcular los valores totales de TP, FP, FN
Tp = sum(conteos["TP"] for conteos in resultados.values())
Fp = sum(conteos["FP"] for conteos in resultados.values())
Fn = sum(conteos["FN"] for conteos in resultados.values())


print("   ")
print("--------------RESULTADOS POR CLASE--------------")
for clase, conteos in resultados.items():
    print(clase)
    print("Verdaderos Positivos (TP):", conteos["TP"])
    print("Falsos Positivos (FP):", conteos["FP"])
    print("Falsos Negativos (FN):", conteos["FN"])

    precision = conteos["TP"]/(conteos["TP"]+conteos["FP"])
    recall = conteos["TP"]/(conteos["TP"]+conteos["FN"])
    f1_score = 2*((precision*recall)/(precision+recall))

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("---------------------------------------------")

precision = Tp/(Tp+Fp)
recall = Tp/(Tp+Fn)
f1_score = 2*((precision*recall)/(precision+recall))

print("   ")
print("--------------RESULTADOS GLOBALES--------------")
print("Verdaderos Positivos:", Tp)
print("Falsos Positivos:", Fp)
print("Falsos Negativos:", Fn)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("---------------------------------------------")

with open(archivo_resultados, "w") as file:
    file.write("\n")
    file.write("--------------RESULTADOS POR CLASE--------------\n")
    for clase, conteos in resultados.items():
        file.write(clase + "\n")
        file.write("Verdaderos Positivos (TP): {}\n".format(conteos["TP"]))
        file.write("Falsos Positivos (FP): {}\n".format(conteos["FP"]))
        file.write("Falsos Negativos (FN): {}\n".format(conteos["FN"]))

        precision = conteos["TP"] / (conteos["TP"] + conteos["FP"])
        recall = conteos["TP"] / (conteos["TP"] + conteos["FN"])
        f1_score = 2 * ((precision * recall) / (precision + recall))

        file.write("Precision: {}\n".format(precision))
        file.write("Recall: {}\n".format(recall))
        file.write("F1 Score: {}\n".format(f1_score))
        file.write("---------------------------------------------\n")

    file.write("\n")
    file.write("--------------RESULTADOS GLOBALES--------------\n")
    file.write("Verdaderos Positivos: {}\n".format(Tp))
    file.write("Falsos Positivos: {}\n".format(Fp))
    file.write("Falsos Negativos: {}\n".format(Fn))

    precision = Tp / (Tp + Fp)
    recall = Tp / (Tp + Fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    file.write("Precision: {}\n".format(precision))
    file.write("Recall: {}\n".format(recall))
    file.write("F1 Score: {}\n".format(f1_score))
    file.write("---------------------------------------------\n")