import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QHBoxLayout, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt  # Import QtCore and Qt here
from cellpose import io, models, train
from PIL import Image
import re
import libs.roi_visualizer
from libs.segment import StopFlag as Cellpose
from libs.mastersheet import genmasterSheet
import cupy
from libs.train import Trainer
import subprocess, sys


# Helper functions
def check_cuda_availability():
    """Check if CUDA Toolkit is available using cupy."""
    try:
        return cupy.is_available()
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
        return False


# Worker Thread for Segmentation
class WorkerThread(QThread):
    finished = pyqtSignal()

    def __init__(self, base_dir, output_dir, diameter, run_segmentation, run_labels2rois):
        super().__init__()
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.diameter = diameter
        self.run_segmentation = run_segmentation
        self.run_labels2rois = run_labels2rois
        self.cellpose = Cellpose()

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
        self.cuda_available = check_cuda_availability()  # Check for CUDA availability
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.base_dir_entry = QLineEdit()
        base_dir_button = QPushButton("Browse")
        base_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #1D1D1D;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #4CAF50;
            }
            QPushButton:hover {
                background-color: #333;
            }
        """)
        base_dir_button.clicked.connect(self.select_base_dir)
        form_layout.addRow("Base Directory:", self.base_dir_entry)
        form_layout.addRow("", base_dir_button)

        self.output_dir_entry = QLineEdit()
        output_dir_button = QPushButton("Browse")
        output_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #1D1D1D;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #4CAF50;
            }
            QPushButton:hover {
                background-color: #333;
            }
        """)
        output_dir_button.clicked.connect(self.select_output_dir)
        form_layout.addRow("Output Directory:", self.output_dir_entry)
        form_layout.addRow("", output_dir_button)

        self.segmentation_checkbox = QCheckBox("Run Segmentation")
        self.segmentation_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 16px;
            }
        """)
        self.labels2rois_checkbox = QCheckBox("Run Label to ROI")
        self.labels2rois_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                font-size: 16px;
            }
        """)


        
        form_layout.addRow(self.segmentation_checkbox)
        form_layout.addRow(self.labels2rois_checkbox)

        # Adjust the styling of the diameter SpinBox and ensure it is visible
        self.diameter_spinbox = QSpinBox()
        self.diameter_spinbox.setRange(1, 100)
        self.diameter_spinbox.setValue(0)
        self.diameter_spinbox.setStyleSheet("background-color: #2B2B2B; color: white; border-radius: 5px; padding: 10px; height: 30px; font-size: 14px;")
        self.diameter_spinbox.valueChanged.connect(self.update_diameter)
        form_layout.addRow("Cellpose Diameter:", self.diameter_spinbox)

        self.run_button = QPushButton("Run Process")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_button.clicked.connect(self.run_process)


        self.cellpose_button = QPushButton("Open Cellpose GUI", self)
         # Set position and size
        self.cellpose_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.cellpose_button.clicked.connect(self.run_cellpose)

        # Set dynamic CUDA status label
        self.cuda_status_label = QLabel("CUDA detected." if self.cuda_available else "CUDA not detected.")
        self.cuda_status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: """ + ("#00e676" if self.cuda_available else "#ff3d00") + """;
            }
        """)

        layout.addLayout(form_layout)
        layout.addWidget(self.run_button)
        layout.addWidget(self.cuda_status_label)
        layout.addWidget(self.cellpose_button)

        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #1F1F1F;
                color: white;
                font-family: Arial, sans-serif;
            }
        """)

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

        self.worker_thread = WorkerThread(
            self.base_dir, self.output_dir, self.diameter,
            self.segmentation_checkbox.isChecked(),
            self.labels2rois_checkbox.isChecked()
        )
        self.worker_thread.finished.connect(self.process_finished)
        self.worker_thread.start()

    def run_cellpose(self):
        subprocess.Popen([sys.executable, "-m", "cellpose"])

    def process_finished(self):
        QMessageBox.information(self, "Process Complete", "The image processing is complete.")



class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.train_dir_entry = QLineEdit()
        train_dir_button = QPushButton("Browse")
        train_dir_button.setStyleSheet("""
            QPushButton {
                background-color: #1D1D1D;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #4CAF50;
            }
            QPushButton:hover {
                background-color: #333;
            }
        """)
        train_dir_button.clicked.connect(self.browse_train_dir)
        form_layout.addRow("Training Directory:", self.train_dir_entry)
        form_layout.addRow("", train_dir_button)

        # self.test_dir_entry = QLineEdit()
        # test_dir_button = QPushButton("Browse")
        # test_dir_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #1D1D1D;
        #         color: white;
        #         padding: 10px;
        #         border-radius: 5px;
        #         border: 1px solid #4CAF50;
        #     }
        #     QPushButton:hover {
        #         background-color: #333;
        #     }
        # """)
        # test_dir_button.clicked.connect(self.browse_test_dir)
        # form_layout.addRow("Testing Directory (Optional):", self.test_dir_entry)
        # form_layout.addRow("", test_dir_button)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["cyto", "cyto2", "cyto3", "nuclei"])
        form_layout.addRow("Model Type:", self.model_combo)

        self.channels_input = QLineEdit("1,2")
        form_layout.addRow("Channels (comma-separated, e.g., 1,2):", self.channels_input)

        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 1000)
        self.epochs_spinbox.setValue(100)
        form_layout.addRow("Number of Epochs:", self.epochs_spinbox)

        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 1.0)
        self.lr_spinbox.setSingleStep(0.01)
        self.lr_spinbox.setValue(0.1)
        form_layout.addRow("Learning Rate:", self.lr_spinbox)

        self.normalize_checkbox = QCheckBox("Normalize Images")
        self.normalize_checkbox.setChecked(True)
        form_layout.addRow(self.normalize_checkbox)

        self.model_name_entry = QLineEdit("my_new_model")
        form_layout.addRow("Model Name:", self.model_name_entry)

        # Mask filter field (new addition)
        self.mask_filter_entry = QLineEdit("_cp_masks")  # Default value is "_cp_masks"
        form_layout.addRow("Mask Filter (e.g., _cp_masks):", self.mask_filter_entry)

        # Image filter field (new addition)
        self.img_filter_entry = QLineEdit("_img")  # Default value is "_img"
        form_layout.addRow("Image Filter (e.g., _img):", self.img_filter_entry)

        train_button = QPushButton("Train")
        train_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                border: none;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        train_button.clicked.connect(self.start_training)

        layout.addLayout(form_layout)
        layout.addWidget(train_button)
    
        self.setLayout(layout)

    def browse_train_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Training Directory")
        self.train_dir_entry.setText(directory)

    def browse_test_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Testing Directory")
        self.test_dir_entry.setText(directory)

    def start_training(self):
        # Get user inputs
        train_dir = self.train_dir_entry.text()
        # test_dir = self.test_dir_entry.text()
        model_name = self.model_name_entry.text()
        model_type = self.model_combo.currentText()
        channels = [int(x) for x in self.channels_input.text().split(",")]
        epochs = self.epochs_spinbox.value()
        learning_rate = self.lr_spinbox.value()
        normalize = self.normalize_checkbox.isChecked()
        mask_filter = self.mask_filter_entry.text()  # Retrieve the mask filter value
        img_filter = self.img_filter_entry.text()  # Retrieve the image filter value

        print(model_name)
        print(model_type)

        if not train_dir:
            QMessageBox.warning(self, "Input Error", "Please select a training directory.")
            return

        # # Debug log for optional test directory
        # if not test_dir:
        #     print("No test directory provided. Proceeding without validation.")

        # Create Trainer instance
        trainer = Trainer(
            train_dir=train_dir,
            # test_dir=test_dir if test_dir else None,  # Pass None if test_dir is empty
            mask_filter=mask_filter,
            img_filter=img_filter,
        )

        # Train the model
        try:
            model_path, train_losses, test_losses = trainer.train(
                model_name=model_name,
                channels=channels,
                epochs=epochs,
                learning_rate=learning_rate,
                normalize=normalize
            )

            
            test_loss_msg = ""
            
            try:
                QMessageBox.information(self, "Training Complete", 
                                        f"Training Complete!\nModel saved at: {model_path}\nTrain Loss: {train_losses[-1]}{test_loss_msg}\nPrecision: {trainer.evaluate_precision()}")
            except ValueError:
                QMessageBox.information(self, "Training Complete", 
                                        f"Training Complete!\nModel saved at: {model_path}\nTrain Loss: {train_losses[-1]}{test_loss_msg}")


        except Exception as e:
            QMessageBox.warning(self, "Error", f"Training failed: {e}")
         



# Main App
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S2L")
        self.setStyleSheet("background-color: #121212; color: white;")

        self.tabs = QTabWidget()
        self.tabs.addTab(SegmentationApp(), "Segmentation")
        self.tabs.addTab(TrainingApp(), "Training")

        self.setCentralWidget(self.tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
