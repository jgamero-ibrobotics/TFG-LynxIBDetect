# Código para registrar los recursos del sistema en un archivo de texto 

# Creado: 27 May 2023
# Última modificación: 02 jun 2023

# @author: Jesús Gamero Borrego

#-----------------------------------------------------------------------------------------


import psutil
import time
import sys
import matplotlib.pyplot as plt

# Nombre del archivo de salida
nombre_archivo = "/home/pi/TFG-LynxIBDetect/Resultados/mediciones.txt"

# Función para escribir los resultados en el archivo
def escribir_resultados(cpu_percent, memory_percent, temperature):
    with open(nombre_archivo, "a") as archivo:
        archivo.write("{:.2f}\t{:.2f}\t{:.2f}\n".format(cpu_percent, memory_percent, temperature))

# Bucle infinito hasta que se presione la tecla "c"
try:
    while True:
        # Obtener el uso de CPU
        cpu_percent = psutil.cpu_percent(interval=1)

        # Obtener el uso de memoria
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Obtener la temperatura de la CPU
        temperature = psutil.sensors_temperatures()['cpu_thermal'][0].current

        # Escribir los resultados en el archivo
        escribir_resultados(cpu_percent, memory_percent, temperature)

        # Imprimir los resultados en la consola
        print("Uso de CPU: {:.2f}%\tUso de memoria: {:.2f}%\tTemperatura de la CPU: {:.2f}°C".format(cpu_percent, memory_percent, temperature))

except KeyboardInterrupt:
    # Capturar la excepción KeyboardInterrupt (tecla "c" presionada)
    print("Programa interrumpido. Se han guardado los resultados en el archivo '{}'".format(nombre_archivo))
    sys.exit(0)