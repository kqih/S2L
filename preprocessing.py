import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QMessageBox, QSpinBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from skimage.io import imread, imsave
from cellpose import denoise

AVAILABLE_MODELS = [
    'denoise_cyto3', 'deblur_cyto3', 'upsample_cyto3', 'oneclick_cyto3',
    'denoise_cyto2', 'deblur_cyto2', 'upsample_cyto2', 'oneclick_cyto2',
    'denoise_nuclei', 'deblur_nuclei', 'upsample_nuclei', 'oneclick_nuclei'
]

CHANNELS = [1, 2]

class WorkerThread(QThread):
    result_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, image, model_name, diameter):
        super().__init__()
        self.image = image
        self.model_name = model_name
        self.diameter = diameter

    def run(self):
        try:
            model_type = self.model_name.split('_')[-1]
            model = denoise.CellposeDenoiseModel(
                gpu=True,
                model_type=model_type,
                restore_type=self.model_name,
                chan2_restore=True
            )
            _, _, _, imgs_dn = model.eval([self.image], channels=CHANNELS, diameter=self.diameter)
            self.result_ready.emit(imgs_dn[0])
        except Exception as e:
            self.error_occurred.emit(str(e))

class CellposeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellpose Image Enhancer")
        self.image = None
        self.processed_image = None
        self.image_path = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.model_box = QComboBox()
        self.model_box.addItems(AVAILABLE_MODELS)
        main_layout.addWidget(self.model_box)

        diameter_layout = QHBoxLayout()
        diameter_label = QLabel("Diameter:")
        self.diameter_input = QSpinBox()
        self.diameter_input.setRange(1, 1000)
        self.diameter_input.setValue(50)
        diameter_layout.addWidget(diameter_label)
        diameter_layout.addWidget(self.diameter_input)
        main_layout.addLayout(diameter_layout)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        main_layout.addWidget(self.load_button)

        self.apply_button = QPushButton("Apply Model")
        self.apply_button.clicked.connect(self.apply_model)
        main_layout.addWidget(self.apply_button)

        self.save_button = QPushButton("Save Processed Image")
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setEnabled(False)
        main_layout.addWidget(self.save_button)

        self.image_layout = QHBoxLayout()
        self.original_label = QLabel("Original Image")
        self.processed_label = QLabel("Processed Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)
        main_layout.addLayout(self.image_layout)

        self.worker_thread = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.image_path = path
            self.image = imread(path)
            pixmap = self.convert_np_to_qpix(self.image)
            if not pixmap.isNull():
                self.original_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
            else:
                QMessageBox.warning(self, "Warning", "Failed to display original image.")
            self.processed_label.clear()
            self.save_button.setEnabled(False)

    def apply_model(self):
        if self.image is None:
            QMessageBox.critical(self, "Error", "No image loaded.")
            return

        model_name = self.model_box.currentText()
        diameter = float(self.diameter_input.value())

        self.apply_button.setEnabled(False)
        self.worker_thread = WorkerThread(self.image, model_name, diameter)
        self.worker_thread.result_ready.connect(self.handle_result)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.start()

    def handle_result(self, result_image):
        self.processed_image = result_image
        pixmap = self.convert_np_to_qpix(result_image)
        if not pixmap.isNull():
            self.processed_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
        self.save_button.setEnabled(True)
        self.apply_button.setEnabled(True)

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.apply_button.setEnabled(True)

    def save_output(self):
        if self.processed_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "output.png", "PNG Image (*.png)")
        if path:
            img_to_save = self.processed_image
            if img_to_save.ndim == 2:
                img_to_save = np.stack([img_to_save]*3, axis=-1)
            elif img_to_save.shape[2] == 1:
                img_to_save = np.repeat(img_to_save, 3, axis=2)
            if img_to_save.max() <= 1.0:
                img_to_save = (img_to_save * 255).astype(np.uint8)
            else:
                img_to_save = img_to_save.astype(np.uint8)
            imsave(path, img_to_save)
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")

    def convert_np_to_qpix(self, img: np.ndarray) -> QPixmap:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if img.ndim == 2:
            qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8)
        else:
            if img.shape[2] == 4:
                qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGBA8888)
            else:
                qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CellposeGUI()
    window.resize(750, 550)
    window.show()
    sys.exit(app.exec())
