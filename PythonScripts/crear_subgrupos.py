import os
import random
import shutil

# Porcentaje de archivos a tomar
porcentaje_train = 0.8
porcentaje_test = 0.1
porcentaje_validation = 0.1

# Ruta de la carpeta donde se encuentran los archivos originales
#ruta_origen = 'C:\\Users\\jesus\\Desktop\\cap_lince'

# Ruta de la carpeta donde se moverán los archivos seleccionados
ruta_destino_train = 'C:\\Users\\jesus\\Desktop\\train'
ruta_destino_test = 'C:\\Users\\jesus\\Desktop\\test'
ruta_destino_validation = 'C:\\Users\\jesus\\Desktop\\validation'

# Obtener una lista de los archivos en la carpeta de origen
archivos = os.listdir(ruta_origen)

# Obtener una lista de los archivos .jpg en la carpeta de origen
archivos_jpg = [archivo for archivo in archivos if archivo.endswith('.jpg')]

# Obtener una lista de los archivos .xml en la carpeta de origen
archivos_xml = [archivo for archivo in archivos if archivo.endswith('.xml')]

# Unir los archivos .jpg y .xml en una lista de parejas
parejas = [(jpg, xml) for jpg in archivos_jpg for xml in archivos_xml if jpg[:-4] == xml[:-4]]

# Calcular la cantidad de archivos a seleccionar para cada carpeta
cantidad_total = len(parejas)
cantidad_train = int(cantidad_total * porcentaje_train)
cantidad_test = int(cantidad_total * porcentaje_test)
cantidad_validation = cantidad_total - cantidad_train - cantidad_test

print("Entrenamiento=",cantidad_train*2)
print("Test=",cantidad_test*2)
print("Validacion=",cantidad_validation*2)

# Seleccionar aleatoriamente las parejas de archivos para cada carpeta
random.shuffle(parejas)
parejas_train = parejas[:cantidad_train]
parejas_test = parejas[cantidad_train:(cantidad_train+cantidad_test)]
parejas_validation = parejas[(cantidad_train+cantidad_test):]


# Mover las parejas seleccionadas a la carpeta de destino
for carpeta, parejas_seleccionadas in zip([ruta_destino_train, ruta_destino_test, ruta_destino_validation], 
                                          [parejas_train, parejas_test, parejas_validation]):
    for pareja in parejas_seleccionadas:
        archivo_jpg = pareja[0]
        archivo_xml = pareja[1]
        ruta_origen_jpg = os.path.join(ruta_origen, archivo_jpg)
        ruta_origen_xml = os.path.join(ruta_origen, archivo_xml)
        ruta_destino_jpg = os.path.join(carpeta, archivo_jpg)
        ruta_destino_xml = os.path.join(carpeta, archivo_xml)
        shutil.copy(ruta_origen_jpg, ruta_destino_jpg)
        shutil.copy(ruta_origen_xml, ruta_destino_xml)

