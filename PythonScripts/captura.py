import cv2
import time

# Función para capturar una imagen de la cámara web
def capture_image():
    cap = cv2.VideoCapture(0)  # Abrir la cámara web

    ret, frame = cap.read()  # Capturar un frame de video

    cap.release()  # Liberar la cámara

    return frame

# Función para guardar la imagen en un archivo
def save_image(image, file_name):
    cv2.imwrite(file_name, image)
    print(f'Imagen guardada como {file_name}')

# Capturar imágenes después de pulsar una tecla y esperar 10 segundos entre cada captura
def capture_images(num_images):
    for i in range(num_images):
        if i > 0:
            print("Esperando 1 segundos...")
            time.sleep(1)
        
        image = capture_image()  # Capturar imagen de la cámara web
        file_name = f'image_{i+1}.jpg'
        save_image(image, file_name)  # Guardar imagen en un archivo

# Llamar a la función para capturar una imagen después de pulsar una tecla y esperar 10 segundos entre cada captura
while True:
    print(f"\nPresiona la tecla 'Enter' para capturar imagenes")
    input()  # Esperar a que se presione la tecla Enter
    capture_images(5)

