import sys
import os
import pandas as pd
from fpdf import FPDF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit, QMessageBox, QGridLayout, QComboBox, QLineEdit, QScrollArea, QCheckBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,  QThread, pyqtSignal

from face_ai_kit.FaceRecognition import FaceRecognition
from progress_window import ProgressWindow
import cv2
import logging

import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("face_recognition_app.log"),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger()



class FaceRecognitionWorker(QThread):
    progress = pyqtSignal(int)
    console_output = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result_ready = pyqtSignal(list, list)  # New signal for results and embeddings
    
    def __init__(self, face_lib, ref_image_paths, comp_image_paths, confidence, parent=None):
        super().__init__(parent)
        self.face_lib = face_lib
        self.ref_image_paths = ref_image_paths
        self.comp_image_paths = comp_image_paths
        self.confidence = confidence
        self.results = []
        self.embeddings = []

    def run(self):
        try:
            total_tasks = len(self.ref_image_paths) * len(self.comp_image_paths)
            completed_tasks = 0

            max_width = None
            if self.parent().resize_checkbox.isChecked() and self.parent().resize_input.text().isdigit():
                max_width = int(self.parent().resize_input.text())

            for ref_image_path in self.ref_image_paths:
                ref_image = cv2.imread(ref_image_path)
                
                # Resize the image if max_width is set
                if max_width:
                    ref_image = self.resize_image(ref_image, max_width=max_width)
                    
                results1 = self.face_lib.face_detection(ref_image, align='keypoints')

                if len(results1) == 0:
                    self.error.emit(f"No face found in the reference image {ref_image_path}.")
                    return

                ref_image = results1[0]["img"]
                ref_encoding = self.face_lib.represent(ref_image)

                self.embeddings.append({'filename': os.path.basename(ref_image_path), 'embedding': ref_encoding[0].tolist()})

                for img_path in self.comp_image_paths:
                    frame_2 = cv2.imread(img_path)
                    
                    # Resize the image if max_width is set
                    if max_width:
                        frame_2 = self.resize_image(frame_2, max_width=max_width)
                        
                    start_time = time.time()  # Start timing the inference
                    
                    results1 = self.face_lib.face_detection(frame_2, align='keypoints')
                    if len(results1) == 0:
                        self.console_output.emit(f"No face found in the image {os.path.basename(img_path)}.")
                        continue

                    img = results1[0]["img"]
                    img_encoding = self.face_lib.represent(img)
                    score = self.face_lib.calculate_distance(ref_encoding, img_encoding, "euclidean_l2")

                    inference_time = time.time() - start_time  # Calculate inference time

                    self.results.append({'filename1': os.path.basename(ref_image_path), 'filename2': os.path.basename(img_path), 'score': score})
                    self.embeddings.append({'filename': os.path.basename(img_path), 'embedding': img_encoding[0].tolist()})

                    completed_tasks += 1
                    self.progress.emit(int((completed_tasks / total_tasks) * 100))
                    
                    # Log the inference time
                    self.console_output.emit(f"Compared {os.path.basename(ref_image_path)} with {os.path.basename(img_path)}: "
                                            f"Score = {score:.5f}, Inference Time = {inference_time:.2f} seconds")

            # Emit results and embeddings when done
            self.result_ready.emit(self.results, self.embeddings)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


    def resize_image(self, image, max_width):
        height, width = image.shape[:2]
        if width > max_width:
            ratio = max_width / float(width)
            new_dimensions = (max_width, int(height * ratio))
            resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
            return resized_image
        return image

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face AI Kit UI - Face Recognition")
        self.setGeometry(100, 100, 800, 600)
        
        self.initUI()
        
    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setPixmap(QPixmap("logo.png").scaled(100, 100, Qt.KeepAspectRatio))
        self.layout.addWidget(self.logo_label, 0, 0, 1, 1)

        # Face recognition model selection
        self.model_label = QLabel("Select Face Recognition Model:")
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(["magface_cwh", "arcface"])
        
        self.layout.addWidget(self.model_label, 1, 0, 1, 3)
        self.layout.addWidget(self.model_combo_box, 1, 3, 1, 3)
        
        # Confidence threshold input
        self.confidence_label = QLabel("Set Confidence Threshold:")
        self.confidence_input = QLineEdit()
        self.confidence_input.setText("0.9")
        
        self.layout.addWidget(self.confidence_label, 2, 0, 1, 3)
        self.layout.addWidget(self.confidence_input, 2, 3, 1, 3)
        
        # Reference image section
        self.reference_image_label = QLabel("No reference image selected")
        self.reference_image_path = QTextEdit()
        self.reference_image_path.setReadOnly(True)
        self.ref_image_button = QPushButton("Select Reference Image")
        self.ref_image_button.clicked.connect(self.select_reference_image)
        self.reference_image_display = QLabel()
        
        self.layout.addWidget(self.reference_image_label, 3, 0, 1, 3)
        self.layout.addWidget(self.reference_image_path, 4, 0, 1, 3)
        self.layout.addWidget(self.ref_image_button, 5, 0, 1, 3)
        self.layout.addWidget(self.reference_image_display, 6, 0, 1, 3)
        
        # Comparison images section
        self.comparison_images_label = QLabel("No comparison images selected")
        self.comparison_images_paths = QTextEdit()
        self.comparison_images_paths.setReadOnly(True)
        self.comp_images_button = QPushButton("Select Comparison Images")
        self.comp_images_button.clicked.connect(self.select_comparison_images)
        
        self.layout.addWidget(self.comparison_images_label, 3, 3, 1, 3)
        self.layout.addWidget(self.comparison_images_paths, 4, 3, 1, 3)
        self.layout.addWidget(self.comp_images_button, 5, 3, 1, 3)

        # Add the checkbox and input field for image resizing
        self.resize_checkbox = QCheckBox("Resize images to max width:")
        self.resize_input = QLineEdit()
        self.resize_input.setText("1500")  # Default value
        self.resize_input.setEnabled(False)  # Initially disabled
        
        self.resize_checkbox.toggled.connect(self.toggle_resize_input)
        
        self.layout.addWidget(self.resize_checkbox, 6, 0, 1, 3)
        self.layout.addWidget(self.resize_input, 6, 3, 1, 3)
        
        # Compare button
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare_faces)
        self.layout.addWidget(self.compare_button, 7, 0, 1, 6)
        
        # Export buttons
        self.export_pdf_button = QPushButton("Export PDF")
        self.export_pdf_button.clicked.connect(self.export_to_pdf)
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.export_embeddings_button = QPushButton("Export Embeddings")
        self.export_embeddings_button.clicked.connect(self.export_embeddings_to_csv)
        
        self.layout.addWidget(self.export_pdf_button, 8, 0, 1, 2)
        self.layout.addWidget(self.export_csv_button, 8, 2, 1, 2)
        self.layout.addWidget(self.export_embeddings_button, 8, 4, 1, 2)
        
        # Results section
        self.results_label = QLabel("Results will be displayed here.")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        self.layout.addWidget(self.results_label, 9, 0, 1, 6)
        self.layout.addWidget(self.results_text, 10, 0, 1, 6)
        
        # Scroll area for detected faces
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        
        self.layout.addWidget(self.scroll_area, 11, 0, 1, 6)

        # Author Name at the bottom
        self.author_label = QLabel("Author: Tomas Goldmann, STRaDe FIT VUT, https://strade.fit.vutbr.cz/ ")
        self.author_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.author_label, 12, 0, 1, 6)
        
        self.results = []
        self.embeddings = []

        
    def set_buttons_enabled(self, enabled):
        self.model_combo_box.setEnabled(enabled)
        self.confidence_input.setEnabled(enabled)
        self.ref_image_button.setEnabled(enabled)
        self.comp_images_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.export_pdf_button.setEnabled(enabled)
        self.export_csv_button.setEnabled(enabled)
        self.export_embeddings_button.setEnabled(enabled)
        
    def select_reference_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Select Reference Image", "", "All Files (*);;Image Files (*.png;*.jpg;*.jpeg)", options=options)
        if files:
            self.reference_image_path.setText(", ".join(files))
            self.reference_image_label.setText(f"Selected {len(files)} reference images.")
            self.reference_image_display.setPixmap(QPixmap(files).scaled(200, 200))
        
    def select_comparison_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Select Comparison Images", "", "All Files (*);;Image Files (*.png;*.jpg;*.jpeg)", options=options)
        if files:
            self.comparison_images_paths.setText(", ".join(files))
            self.comparison_images_label.setText(f"Selected {len(files)} comparison images.")

    def toggle_resize_input(self):
        if self.resize_checkbox.isChecked():
            self.resize_input.setEnabled(True)
        else:
            self.resize_input.setEnabled(False)

            
    def on_results_ready(self, results, embeddings):
        # Assign the results and embeddings to the main thread variables
        self.results = results
        self.embeddings = embeddings
        
    def compare_faces(self):
        model = self.model_combo_box.currentText()
        self.face_lib = FaceRecognition(recognition=model)

        ref_image_paths = self.reference_image_path.toPlainText().split(", ")
        comp_image_paths = self.comparison_images_paths.toPlainText().split(", ")
        confidence = self.confidence_input.text()
        if not ref_image_paths or ref_image_paths == ['']:
            QMessageBox.warning(self, "Warning", "Please select at least one reference image.")
            logger.warning("No reference images selected.")
            return

        # Check if comparison image paths are empty
        if not comp_image_paths or comp_image_paths == ['']:
            QMessageBox.warning(self, "Warning", "Please select at least one comparison image.")
            logger.warning("No comparison images selected.")
            return

        if not confidence:
            QMessageBox.warning(self, "Warning", "Please set a confidence threshold.")
            logger.warning("No confidence threshold set.")
            return
        
        try:
            confidence = float(confidence)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid confidence threshold.")
            logger.error("Invalid confidence threshold entered.")
            return
        
        self.set_buttons_enabled(False)

        # Show progress window
        self.progress_window = ProgressWindow()
        self.progress_window.show()
        
        # Start worker thread
        self.worker = FaceRecognitionWorker(self.face_lib, ref_image_paths, comp_image_paths, confidence, parent=self)
        self.worker.progress.connect(self.progress_window.update_progress)
        self.worker.console_output.connect(self.progress_window.append_console)
        self.worker.result_ready.connect(self.on_results_ready)  # Connect the results signal
        self.worker.finished.connect(self.on_comparison_finished)
        self.worker.error.connect(self.on_comparison_error)
        self.worker.start()

    def on_comparison_finished(self):
        self.progress_window.append_console("Comparison complete.")
        self.progress_window.update_progress(100)
        logger.info("Face comparison completed.")
        self.display_results()
        self.display_detected_faces(self.worker.comp_image_paths)
        self.set_buttons_enabled(True)
        
        self.progress_window.close()

    def on_comparison_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        logger.error(f"Error during face comparison: {error_message}")
        self.progress_window.append_console(f"Error: {error_message}")
        self.set_buttons_enabled(True)
        
    def display_results(self):
        if not self.results:
            self.results_label.setText("No results to display.")
            return
        
        results_text = "Comparison Results:\n"
        for result in self.results:
            results_text += f"{result['filename1']} - {result['filename2']} : {result['score']}\n"
        
        self.results_text.setPlainText(results_text)
        
    def export_to_pdf(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Face Recognition Comparison Results", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        for row in self.results:
            pdf.cell(200, 10, txt=f"{row['filename1']} - {row['filename2']} : {row['score']}", ln=True)
        
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)", options=options)
        if file_path:
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            pdf.output(file_path)
            QMessageBox.information(self, "Success", f"PDF exported to {file_path}")
        
    def export_to_csv(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        df = pd.DataFrame(self.results)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", f"CSV exported to {file_path}")

    def export_embeddings_to_csv(self):
        if not self.embeddings:
            QMessageBox.warning(self, "Warning", "No embeddings to export.")
            return
        
        # Create a list of dictionaries where each dictionary contains the filename and each dimension of the embedding
        embeddings_data = []
        for embedding in self.embeddings:
            embedding_dict = {'filename': embedding['filename']}
            for i, value in enumerate(embedding['embedding']):
                embedding_dict[f'dim_{i+1}'] = value
            embeddings_data.append(embedding_dict)

        df = pd.DataFrame(embeddings_data)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Embeddings CSV", "", "CSV Files (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", f"Embeddings CSV exported to {file_path}")
            
    def display_detected_faces(self, comp_image_paths):
        # Clear the previous images
        for i in reversed(range(self.scroll_layout.count())): 
            widgetToRemove = self.scroll_layout.itemAt(i).widget()
            self.scroll_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
        
        # Add new images in a grid layout
        row, col = 0, 0
        for img_path in comp_image_paths:
            img_label = QLabel()
            img_label.setPixmap(QPixmap(img_path).scaled(100, 100, Qt.KeepAspectRatio))
            self.scroll_layout.addWidget(img_label, row, col)
            col += 1
            if col == 3:  # Assuming 3 columns per row
                col = 0
                row += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaceRecognitionApp()
    ex.show()
    sys.exit(app.exec_())
