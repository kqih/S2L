import sys
import os
import subprocess
import re
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from cellpose import io, models, train
from PIL import Image
import libs.roi_visualizer
from libs.segment import StopFlag as Cellpose
from libs.mastersheet import genmasterSheet
from libs.train import Trainer


def install_himena_plugins():
    try:
        subprocess.call("himena --install himena-image")
    except:
        pass


def check_cuda_availability():
    try:
        import cupy
    except ImportError:
        return False

    try:
        return cupy.is_available()
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
        return False


class WorkerThread(QThread):
    finished = pyqtSignal()

    def __init__(self, base_dir, output_dir, diameter, run_segmentation, run_labels2rois, model_type):
        super().__init__()
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.diameter = diameter
        self.run_segmentation = run_segmentation
        self.run_labels2rois = run_labels2rois
        self.model_type = model_type
        self.cellpose = Cellpose(model_type=self.model_type)

    def run(self):
        if self.run_segmentation:
            self.cellpose.segment(self.base_dir, diameter=self.diameter)

        if self.run_labels2rois:
            masks = [os.path.join(self.base_dir, f) for f in os.listdir(self.base_dir) if "cp_masks" in f]
            if not masks:
                return

            for mask in masks:
                label_image_path = mask
                original_image_path = self.get_original_image_path(mask)
                plot_output_path = os.path.join(self.output_dir, f"{self.get_file_base_name(mask)}_ROI.png")
                excel_output_path = os.path.join(self.output_dir, f"{self.get_file_base_name(mask)}.xlsx")

                if not os.path.exists(original_image_path):
                    continue

                visualizer = libs.roi_visualizer.ROIVisualizer(
                    label_image_path, original_image_path, excel_output_path, plot_output_path, show_labels=True
                )
                visualizer.save_rois_to_excel()

            summary_dir = os.path.join(self.output_dir, "SummarySheet")
            os.makedirs(summary_dir, exist_ok=True)
            genmasterSheet(self.output_dir, os.path.join(summary_dir, "Summary.xlsx"))

        self.finished.emit()

    def get_original_image_path(self, mask_path):
        mask_base_name = self.get_file_base_name(mask_path)
        pattern = re.sub(r'_cp_masks$', '', mask_base_name)

        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if pattern in file and file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                    return os.path.join(root, file)
        return None

    def get_file_base_name(self, file_path):
        return os.path.splitext(os.path.basename(file_path))[0]


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.base_dir = ''
        self.output_dir = ''
        self.diameter = 0
        self.cuda_available = check_cuda_availability()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Base Directory
        self.base_dir_entry = QLineEdit()
        base_dir_button = QPushButton("Browse")
        base_dir_button.clicked.connect(self.select_base_dir)
        form_layout.addRow("Base Directory:", self.base_dir_entry)
        form_layout.addRow("", base_dir_button)

        # Output Directory
        self.output_dir_entry = QLineEdit()
        output_dir_button = QPushButton("Browse")
        output_dir_button.clicked.connect(self.select_output_dir)
        form_layout.addRow("Output Directory:", self.output_dir_entry)
        form_layout.addRow("", output_dir_button)

        # Segmentation Options
        self.segmentation_checkbox = QCheckBox("Run Segmentation")
        self.labels2rois_checkbox = QCheckBox("Run Label to ROI")
        form_layout.addRow(self.segmentation_checkbox)
        form_layout.addRow(self.labels2rois_checkbox)

        # Model Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "cyto", "cyto2", "cyto3", "nuclei",
            "tissuenet_cp3", "livecell_cp3", "deepbacs_cp3", "Custom Model"
        ])
        form_layout.addRow("Model Type:", self.model_combo)

        self.custom_model_entry = QLineEdit()
        self.custom_model_entry.setPlaceholderText("Path to custom model (if selected)")
        form_layout.addRow("Custom Model Path:", self.custom_model_entry)

        # Diameter
        self.diameter_spinbox = QSpinBox()
        self.diameter_spinbox.setRange(1, 100)
        self.diameter_spinbox.setValue(30)
        self.diameter_spinbox.valueChanged.connect(self.update_diameter)
        form_layout.addRow("Cellpose Diameter:", self.diameter_spinbox)

        # Run Button
        self.run_button = QPushButton("Run Process")
        self.run_button.clicked.connect(self.run_process)

        # Cellpose GUI Button
        self.cellpose_button = QPushButton("Open Cellpose GUI")
        self.cellpose_button.clicked.connect(self.run_cellpose)

        # CUDA Status
        self.cuda_status_label = QLabel("CUDA detected." if self.cuda_available else "CUDA not detected.")
        self.cuda_status_label.setStyleSheet(
            f"color: {'#00e676' if self.cuda_available else '#ff3d00'}; font-weight: bold;"
        )

        # Layout Packing
        layout.addLayout(form_layout)
        layout.addWidget(self.run_button)
        layout.addWidget(self.cellpose_button)
        layout.addWidget(self.cuda_status_label)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #1F1F1F; color: white;")

    def select_base_dir(self):
        self.base_dir = QFileDialog.getExistingDirectory(self, "Select Base Directory")
        self.base_dir_entry.setText(self.base_dir)

    def select_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        self.output_dir_entry.setText(self.output_dir)

    def update_diameter(self):
        self.diameter = self.diameter_spinbox.value()

    def run_process(self):
        if not self.base_dir or not self.output_dir:
            QMessageBox.warning(self, "Input Error", "Please select both base and output directories.")
            return

        model_type = self.model_combo.currentText()
        if model_type == "Custom Model":
            model_type = self.custom_model_entry.text()
            if not os.path.exists(model_type):
                QMessageBox.warning(self, "Model Error", "Invalid custom model path.")
                return

        self.worker_thread = WorkerThread(
            self.base_dir, self.output_dir, self.diameter_spinbox.value(),
            self.segmentation_checkbox.isChecked(),
            self.labels2rois_checkbox.isChecked(),
            model_type
        )
        self.worker_thread.finished.connect(self.process_finished)
        self.worker_thread.start()

    def process_finished(self):
        QMessageBox.information(self, "Done", "Image processing complete.")

    def run_cellpose(self):
        try:
            subprocess.Popen([sys.executable, "-m", "cellpose"])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open Cellpose GUI: {e}")


class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        form_layout = QFormLayout()

        self.train_dir_entry = QLineEdit()
        train_dir_button = QPushButton("Browse")
        train_dir_button.clicked.connect(self.browse_train_dir)
        form_layout.addRow("Training Directory:", self.train_dir_entry)
        form_layout.addRow("", train_dir_button)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "cyto", "cyto2", "cyto3", "nuclei",
            "tissuenet_cp3", "livecell_cp3", "deepbacs_cp3"
        ])
        form_layout.addRow("Model Type:", self.model_combo)

        self.channels_input = QLineEdit("1,2")
        form_layout.addRow("Channels (e.g., 1,2):", self.channels_input)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)
        form_layout.addRow("Epochs:", self.epochs_spinbox)

        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setValue(0.1)
        self.lr_spinbox.setSingleStep(0.01)
        form_layout.addRow("Learning Rate:", self.lr_spinbox)

        self.normalize_checkbox = QCheckBox("Normalize Images")
        self.normalize_checkbox.setChecked(True)
        form_layout.addRow(self.normalize_checkbox)

        self.model_name_entry = QLineEdit("my_new_model")
        form_layout.addRow("Model Name:", self.model_name_entry)

        self.mask_filter_entry = QLineEdit("_cp_masks")
        form_layout.addRow("Mask Filter:", self.mask_filter_entry)

        self.img_filter_entry = QLineEdit("_img")
        form_layout.addRow("Image Filter:", self.img_filter_entry)

        train_button = QPushButton("Train")
        train_button.clicked.connect(self.start_training)

        layout.addLayout(form_layout)
        layout.addWidget(train_button)
        self.setLayout(layout)

    def browse_train_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Directory")
        self.train_dir_entry.setText(directory)

    def start_training(self):
        train_dir = self.train_dir_entry.text()
        model_name = self.model_name_entry.text()
        model_type = self.model_combo.currentText()
        channels = [int(x) for x in self.channels_input.text().split(",")]
        epochs = self.epochs_spinbox.value()
        lr = self.lr_spinbox.value()
        normalize = self.normalize_checkbox.isChecked()
        mask_filter = self.mask_filter_entry.text()
        img_filter = self.img_filter_entry.text()

        if not train_dir:
            QMessageBox.warning(self, "Input Error", "Please select a training directory.")
            return

        trainer = Trainer(train_dir=train_dir, mask_filter=mask_filter, img_filter=img_filter)

        try:
            model_path, train_losses, test_losses = trainer.train(
                model_name=model_name,
                channels=channels,
                epochs=epochs,
                learning_rate=lr,
                normalize=normalize
            )

            QMessageBox.information(self, "Training Complete",
                                    f"Model saved at: {model_path}\nTrain Loss: {train_losses[-1]}")
        except Exception as e:
            QMessageBox.warning(self, "Training Failed", str(e))


class Helpers(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.cellpose_button = QPushButton("Open Cellpose GUI")
        self.cellpose_button.clicked.connect(self.run_cellpose)
        layout.addWidget(self.cellpose_button)

        self.enhancer_button = QPushButton("Open Preprocessing")
        self.enhancer_button.clicked.connect(self.run_enhancer)
        layout.addWidget(self.enhancer_button)

        self.himena_button = QPushButton("Open Himena")
        self.himena_button.clicked.connect(self.run_himena)
        layout.addWidget(self.himena_button)

        self.setLayout(layout)

    def run_cellpose(self):
        subprocess.Popen([sys.executable, "-m", "cellpose"])

    def run_enhancer(self):
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "preprocessing.py")])

    def run_himena(self):
        try:
            subprocess.Popen(["himena"])
        except FileNotFoundError:
            print("Himena executable not found. Ensure it's in your system PATH.")


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2L")
        self.setStyleSheet("background-color: #121212; color: white;")

        self.tabs = QTabWidget()
        self.tabs.addTab(SegmentationApp(), "Segmentation")
        self.tabs.addTab(TrainingApp(), "Training")
        self.tabs.addTab(Helpers(), "Helpers")

        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background: #1F1F1F;
            }
            QTabBar::tab {
                background: #333333;
                color: white;
                padding: 10px;
                border: 1px solid #444444;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #555555;
            }
        """)

        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    install_himena_plugins()
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
