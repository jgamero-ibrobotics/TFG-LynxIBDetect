import os
import random
import shutil

import matplotlib.pyplot as plt

def generar_diagrama_circular(etiquetas, tamaños):
    # Configurar los tamaños y etiquetas del diagrama circular
    fig, ax = plt.subplots()
    ax.pie(tamaños, labels=etiquetas, autopct='%1.1f%%', startangle=90)

    # Añadir título
   # ax.set_title("Diagrama Circular")

    # Mostrar el diagrama
    plt.show()



# Porcentaje de archivos a tomar
porcentaje_train = 0.8
porcentaje_test = 0.1
porcentaje_validation = 0.1

# Ruta del directorio que contiene las carpetas con los archivos originales
directorio_origen = 'C:\\Users\\jesus\\Desktop\\clases'

# Ruta del directorio donde se moverán los archivos seleccionados
directorio_destino_train = 'C:\\Users\\jesus\\Desktop\\train'
directorio_destino_test = 'C:\\Users\\jesus\\Desktop\\test'
directorio_destino_validation = 'C:\\Users\\jesus\\Desktop\\validation'

# Obtener la lista de carpetas en el directorio de origen
carpetas = os.listdir(directorio_origen)

# Recorrer cada carpeta y realizar la operación por separado
for carpeta in carpetas:
    # Ruta completa de la carpeta actual
    ruta_carpeta = os.path.join(directorio_origen, carpeta)
    
    # Obtener una lista de los archivos en la carpeta actual
    archivos = os.listdir(ruta_carpeta)
    
    # Obtener una lista de los archivos .jpg en la carpeta actual
    archivos_jpg = [archivo for archivo in archivos if archivo.endswith('.jpg')]
    
    # Obtener una lista de los archivos .xml en la carpeta actual
    archivos_xml = [archivo for archivo in archivos if archivo.endswith('.xml')]
    
    # Unir los archivos .jpg y .xml en una lista de parejas
    parejas = [(jpg, xml) for jpg in archivos_jpg for xml in archivos_xml if jpg[:-4] == xml[:-4]]
    
    # Calcular la cantidad de archivos a seleccionar para cada carpeta
    cantidad_total = len(parejas)
    cantidad_train = int(cantidad_total * porcentaje_train)
    cantidad_test = int(cantidad_total * porcentaje_test)
    cantidad_validation = cantidad_total - cantidad_train - cantidad_test
    
    print('---------CLASE--------:  ' + carpeta)
    print("Entrenamiento =", cantidad_train * 2)
    print("Prueba =", cantidad_test * 2)
    print("Validacion =", cantidad_validation * 2)

    if carpeta == 'lince':
        lince = cantidad_train
    elif carpeta == 'zorro':
        zorro = cantidad_train
    elif carpeta == 'gatos':
        gatos = cantidad_train
    elif carpeta == 'humanos':
        humanos = cantidad_train
    elif carpeta == 'conejo':
        conejo = cantidad_train
    
    # Seleccionar aleatoriamente las parejas de archivos para cada carpeta
    random.shuffle(parejas)
    parejas_train = parejas[:cantidad_train]
    parejas_test = parejas[cantidad_train:(cantidad_train + cantidad_test)]
    parejas_validation = parejas[(cantidad_train + cantidad_test):]
    
    # Crear los directorios de destino si no existen
    os.makedirs(directorio_destino_train, exist_ok=True)
    os.makedirs(directorio_destino_test, exist_ok=True)
    os.makedirs(directorio_destino_validation, exist_ok=True)
    
    # Mover las parejas seleccionadas a los directorios de destino
    for carpeta_destino, parejas_seleccionadas in zip([directorio_destino_train, directorio_destino_test, directorio_destino_validation],
                                                    [parejas_train, parejas_test, parejas_validation]):
        for pareja in parejas_seleccionadas:
            archivo_jpg = pareja[0]
            archivo_xml = pareja[1]
            ruta_origen_jpg = os.path.join(ruta_carpeta, archivo_jpg)
            ruta_origen_xml = os.path.join(ruta_carpeta, archivo_xml)
            ruta_destino_jpg = os.path.join(carpeta_destino, archivo_jpg)
            ruta_destino_xml = os.path.join(carpeta_destino, archivo_xml)
            # shutil.copy(ruta_origen_jpg, ruta_destino_jpg)
            # shutil.copy(ruta_origen_xml, ruta_destino_xml)

etiquetas = ['Lince', 'Zorro', 'Gato', 'Humano', 'Conejo']
total = lince + zorro + gatos + humanos + conejo
tamaños = [(lince*total)/100, (zorro*total)/100, (gatos*total)/100, (humanos*total)/100, (conejo*total)/100]
generar_diagrama_circular(etiquetas, tamaños)

