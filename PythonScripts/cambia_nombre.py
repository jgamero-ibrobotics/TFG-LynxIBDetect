import os

# Ruta de la carpeta donde se encuentran los archivos
#ruta = 'C:\\Users\\jesus\\OneDrive - UNIVERSIDAD DE SEVILLA\\universidad\\4\\2C\\TFG\\dataset rabbit'
#ruta = 'C:\\Users\\jesus\\Desktop\\cap_lince'
# Obtener una lista de los archivos en la carpeta
archivos = os.listdir(ruta)
j=283
k=283
# Iterar a través de cada archivo y cambiar su nombre
for i, archivo in enumerate(archivos):
    # Obtener el nombre y la extensión del archivo
    nombre, extension = os.path.splitext(archivo)
    # Crear el nuevo nombre para el archivo
    if(extension == ".xml"):
        nuevo_nombre = f"lince{j}{extension}"
        j += 1
        os.rename(os.path.join(ruta, archivo), os.path.join(ruta, nuevo_nombre))
    if(False):#extension == ".jpg"):
        nuevo_nombre = f"lince{k}{extension}"
        k += 1
        os.rename(os.path.join(ruta, archivo), os.path.join(ruta, nuevo_nombre))
    # Renombrar el archivo
    
