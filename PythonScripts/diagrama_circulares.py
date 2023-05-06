# import os
# import re
# import matplotlib.pyplot as plt

# def generar_diagrama_circular(datos):
#     # Obtener los nombres y contar las ocurrencias
#     nombres = list(datos.keys())
#     ocurrencias = list(datos.values())

#     # Configurar los tamaños y etiquetas del diagrama circular
#     fig, ax = plt.subplots()
#     ax.pie(ocurrencias, labels=nombres, autopct='%1.1f%%', startangle=90)

#     # Añadir título
#     #ax.set_title("Diagrama Circular Clases")

#     # Mostrar el diagrama
#     plt.show()

# def contar_nombres_jpg(ruta_directorio):
#     datos = {}

#     # Obtener la lista de archivos en el directorio
#     archivos = os.listdir(ruta_directorio)

#     # Expresión regular para extraer solo las letras del nombre del archivo
#     patron = re.compile('[^a-zA-Z]+')

#     # Recorrer los archivos
#     for archivo in archivos:
#         nombre, extension = os.path.splitext(archivo)

#         # Solo procesar archivos .jpg
#         if extension == ".jpg":
#             # Extraer solo las letras del nombre
#             nombre_sin_numero = re.sub(patron, '', nombre).lower()

#             # Contar las ocurrencias del nombre
#             if nombre_sin_numero in datos:
#                 datos[nombre_sin_numero] += 1
#             else:
#                 datos[nombre_sin_numero] = 1

#     return datos

# # Directorio de ejemplo
# directorio = 'C:\\Users\\jesus\\Desktop\\test'

# # Contar los nombres de archivos .jpg en el directorio
# datos = contar_nombres_jpg(directorio)

# # Generar el diagrama circular
# generar_diagrama_circular(datos)



import matplotlib.pyplot as plt

def generar_diagrama_circular(etiquetas, tamaños):
    # Configurar los tamaños y etiquetas del diagrama circular
    fig, ax = plt.subplots()
    ax.pie(tamaños, labels=etiquetas, autopct='%1.1f%%', startangle=90)

    # Añadir título
   # ax.set_title("Diagrama Circular")

    # Mostrar el diagrama
    plt.show()

# Ejemplo de uso
# etiquetas = ['Fototrampeo', 'Repositorios cientificos', 'Google fotos y similares']
etiquetas = ['Validation', 'Test', 'Train']

total = 348
fototrampeo = 155
repositorios = 100
google_fotos = 92

#tamaños = [(fototrampeo*total)/100, (repositorios*total)/100, (google_fotos*total)/100]
tamaños = [10, 10, 80]

generar_diagrama_circular(etiquetas, tamaños)