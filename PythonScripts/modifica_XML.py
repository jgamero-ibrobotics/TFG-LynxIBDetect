import xml.etree.ElementTree as ET
import os

# Carpeta que contiene los archivos XML y JPG
folder_path = 'C:\\Users\\jesus\\Desktop\\gatos'

# Recorrer todos los archivos XML en la carpeta
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xml'):
        # Obtener el nombre de archivo sin extensión
        filename = os.path.splitext(file_name)[0]

        # Construir la ruta del archivo de imagen
        image_path = os.path.join(folder_path, f"{filename}.jpg")

        # Cargar el archivo XML
        file_path = os.path.join(folder_path, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Acceder al elemento "filename" y cambiar su valor
        filename_element = root.find('filename')
        filename_element.text = f"{filename}.jpg"

        # Guardar el archivo XML actualizado
        tree.write(file_path)