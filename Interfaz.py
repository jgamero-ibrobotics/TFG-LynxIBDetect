# Código para la interfaz gráfica de la aplicación LynxDetect
# permite visualizar todas las imagenes que has resultado positivas 
# en la detección las cuales se encuentran en la carpeta compartida
# de Dropbox. Además, se muestra la información de cada imagen y 
# la información total de todas las imágenes detectadas.

# Creado: 10 may 2023
# Última modificación: 02 jun 2023

# @author: Jesús Gamero Borrego

#-----------------------------------------------------------------------------------------

import sys
import os
import dropbox
from dropbox.exceptions import AuthError
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QMessageBox,QStyleFactory
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer


class ImageGallery(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LynxDetect')
        self.images = []
        self.txt_files = []
        self.current_image_index = 0

        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        self.previous_button = QPushButton("Anterior")
        self.next_button = QPushButton("Siguiente")

        self.previous_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        self.button_layout.addWidget(self.previous_button)
        self.button_layout.addWidget(self.next_button)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.client = dropbox.Dropbox("sl.Bfg-B5v-2JGRSkmes8jUFZ-lLjYkD3Dm-dnUET_W33eev1iNwAViMkqsYI2qCWmAuUTnbKw4DJrqn05eW9jcrvGfQ22nnwABy9bU7tDyYZLQ3_OrQp0V_bWkhgkFeHoZOdmQAqF9zcY")

        self.info_total = ""
        self.total_class_counts = {
                "lynx pardinus": 0,
                "fox": 0,
                "rabbit": 0,
                "cat": 0,
                "person": 0,
                "person-like": 0
            }
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_updates)
        self.timer.start(5000)  # Verificar actualizaciones cada 5 segundos

        self.load_images()

        if len(self.images) > 0:
            self.show_current_image()

    def load_images(self):
        try:
            response = self.client.files_list_folder("/Detections")
            for entry in response.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(".jpg"):
                    image_path = f"{entry.path_display}"
                    image_name = os.path.splitext(entry.name)[0]
                    txt_path = f"/Detections/results_{image_name}.txt"
                    self.info_total = self.process_txt_file_total(txt_path)

                    if image_path not in self.images:  # Evitar duplicados
                        self.images.append(image_path)
                        self.txt_files.append(txt_path)
                # if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith(".txt"):
                    
        except AuthError:
            QMessageBox.warning(self, "Error", "Error de autenticación. Por favor, verifica tu token de acceso.")

    def show_current_image(self):
        if len(self.images) > 0:
            try:
                _, response = self.client.files_download(self.images[self.current_image_index])
                data = response.content
                pixmap = QPixmap()
                pixmap.loadFromData(data)

                label = QLabel()
                label.setPixmap(pixmap)

                filename = os.path.basename(self.images[self.current_image_index])
                filename_label = QLabel(filename)
                filename_label.setStyleSheet("font-size: 23px;  font-weight: bold;")
                
                info = self.process_txt_file(self.txt_files[self.current_image_index])
                info_label = QLabel()
                info_label.setText(f"Información:\n{info}")
                info_label.setStyleSheet("font-size: 16px;")

                info_total_label = QLabel()
                info_total_label.setText(f"Información Total:\n{self.info_total}")
                info_total_label.setStyleSheet("font-size: 16px;")


                # Eliminar cualquier widget existente antes de agregar la imagen, nombre de archivo e información
                while self.layout.count():
                    item = self.layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()

                self.layout.addWidget(filename_label)
                self.layout.addWidget(label)
                self.layout.addWidget(info_label)
                self.layout.addWidget(info_total_label)
                self.layout.addLayout(self.button_layout)  # Agregar el diseño de los botones al layout principal

            except dropbox.exceptions.HttpError:
                QMessageBox.warning(self, "Error", "No se pudo descargar la imagen.")

    def process_txt_file(self, txt_file):
        try:
            _, response = self.client.files_download(txt_file)
            content = response.content.decode("utf-8")
            lines = content.split("\n")
            class_counts = {
                "lynx pardinus": 0,
                "fox": 0,
                "rabbit": 0,
                "cat": 0,
                "person": 0,
                "person-like": 0
            }
            for line in lines:
                line = line.strip()  # Ignorar líneas vacías
                if line:
                    class_name = line.split(",")[0].strip().strip("[]").replace("'", "").lower()
                    class_counts[class_name] += 1
            info = ""
            for class_name, count in class_counts.items():
                if count > 0:
                    info += f"{class_name}: {count}\t"
            return info
        except dropbox.exceptions.HttpError:
            return ""
        
    def process_txt_file_total(self, txt_file):
        try:
            _, response = self.client.files_download(txt_file)
            content = response.content.decode("utf-8")
            lines = content.split("\n")
            for line in lines:
                line = line.strip()  # Ignorar líneas vacías
                if line:
                    class_name = line.split(",")[0].strip().strip("[]").replace("'", "").lower()
                    self.total_class_counts[class_name] += 1
            self.info_total = ""
            for class_name, count in self.total_class_counts.items():
                if count > 0:
                    self.info_total += f"{class_name}: {self.total_class_counts[class_name]}\t"
            return self.info_total
        except dropbox.exceptions.HttpError:
            return ""

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1 and self.current_image_index < len(self.txt_files) - 1:
            self.current_image_index += 1
            self.show_current_image()

    def check_updates(self):
        # Verificar si hay cambios en la carpeta compartida
        try:
            response = self.client.files_list_folder('/Detections')
            updated_images = []
            updated_txt_files = []
            for entry in response.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.jpg'):
                    download_path = f'{entry.path_display}'
                    if download_path not in self.images:
                        updated_images.append(download_path)
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.txt'):
                    download_path = f'{entry.path_display}'
                    if download_path not in self.txt_files:
                        self.info_total = self.process_txt_file_total(download_path)
                        updated_txt_files.append(download_path)
            
            if updated_images:
                self.images.extend(updated_images)
                self.txt_files.extend(updated_txt_files)
                self.show_current_image()
            
        except dropbox.exceptions.AuthError:
            QMessageBox.warning(self, 'Error', 'Error de autenticación. Por favor, verifica tu token de acceso.')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gallery = ImageGallery()
    app.setStyle(QStyleFactory.create("Fusion"))
    gallery.show()
    sys.exit(app.exec_())
