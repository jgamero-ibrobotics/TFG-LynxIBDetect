import cv2

def capture_frames(video_path, output_folder, interval):
    # Abre el archivo de video
    video = cv2.VideoCapture(video_path)

    # Verifica si el archivo de video se ha abierto correctamente
    if not video.isOpened():
        print("No se pudo abrir el archivo de video")
        return

    # Variables para el control del tiempo
    current_time = 0
    frame_count = 0

    while True:
        # Lee el siguiente frame del video
        ret, frame = video.read()

        # Verifica si se pudo leer el frame correctamente
        if not ret:
            break

        # Incrementa el tiempo actual
        current_time += 1

        # Verifica si ha pasado el intervalo deseado
        if current_time % interval == 0:
            # Genera el nombre del archivo de salida
            output_path = f"{output_folder}/frame_{frame_count}.jpg"

            # Guarda el frame como una imagen
            cv2.imwrite(output_path, frame)

            # Incrementa el contador de frames
            frame_count += 1

    # Cierra el archivo de video
    video.release()

    print(f"Se han capturado {frame_count} frames en total")

# Ejemplo de uso
video_path = r"C:\Users\jesus\Downloads\vid4.mp4"  # Ruta al archivo de video
output_folder = r"C:\Users\jesus\Desktop\cap_lince"  # Ruta a la carpeta de salida
interval = 30  # Intervalo de tiempo en el que se capturarán los frames (en segundos)

capture_frames(video_path, output_folder, interval)
