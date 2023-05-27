import time
import sys
import matplotlib.pyplot as plt

nombre_archivo = "mediciones.txt"
# Capturar la excepción KeyboardInterrupt (tecla "c" presionada)
print("Programa interrumpido. Se han guardado los resultados en el archivo '{}'".format(nombre_archivo))
# Listas para almacenar los datos
cpu_percentajes = []
memory_percentajes = []
temperaturas = []

# Leer los resultados del archivo y almacenarlos en las listas
with open(nombre_archivo, "r") as archivo:
    lineas = archivo.readlines()
    for linea in lineas:
        datos = linea.strip().split("\t")
        cpu_percentajes.append(float(datos[0]))
        memory_percentajes.append(float(datos[1]))
        temperaturas.append(float(datos[2]))

# Crear la gráfica de Uso de CPU
plt.plot(cpu_percentajes, label="Uso de CPU")
plt.xlabel("Tiempo")
plt.ylabel("Porcentaje de Uso")
plt.title("Uso de CPU a lo largo del tiempo")
plt.legend()
plt.grid(True)
plt.show()

# Crear la gráfica de Uso de Memoria
plt.plot(memory_percentajes, label="Uso de Memoria")
plt.xlabel("Tiempo")
plt.ylabel("Porcentaje de Uso")
plt.title("Uso de Memoria a lo largo del tiempo")
plt.legend()
plt.grid(True)
plt.show()

# Crear la gráfica de Temperatura
plt.plot(temperaturas, label="Temperatura de la CPU")
plt.xlabel("Tiempo")
plt.ylabel("Temperatura (°C)")
plt.title("Temperatura de la CPU a lo largo del tiempo")
plt.legend()
plt.grid(True)
plt.show()
