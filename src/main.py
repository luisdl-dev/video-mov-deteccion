import sys
import os
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PyQt5.QtCore import QTime
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QMessageBox, QComboBox, QProgressBar, QDoubleSpinBox, QTimeEdit, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from openvino import Core
import numpy as np

# =======================
# CONFIGURACIÓN GENERAL
# =======================

MOTION_RATIO_THRESHOLD = 0.001
MOTION_RATIO_NOTORIO = 0.02
DNN_CONFIDENCE_MIN = 0.50
BBOX_MOTION_MIN = 200
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 25
MOG2_DETECT_SHADOWS = True
MORPH_OPEN_KERNEL = 3
SKIP_AFTER_SAVE_SECONDS = 60
ENABLE_SKIP_AFTER_SAVE = True
REQUIRE_OBJECT_FOR_SAVE = True
FILTER_SHADOWS = True
DEBUG_MODE = False

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "VIDEOEDIT_v2.log")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Inicio de nueva sesión ---\n")


class FrameExtractorThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(int)

    def __init__(self, video_path, output_dir, interval_frames, image_format, start_time, end_time,
                 resize_option, use_motion, show_timestamp, debug_mode):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval_frames = max(1, int(interval_frames))
        self.image_format = image_format
        self.start_time = start_time
        self.end_time = end_time
        self.resize_option = resize_option
        self.use_motion = use_motion
        self.show_timestamp = show_timestamp
        self.debug_mode = bool(debug_mode)

        self.enable_person_openvino = True
        self.enable_face_openvino = True

        core = Core()
        model_path = os.path.join(BASE_DIR, "person-detection-retail-0013.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError("No se encontró el modelo person-detection-retail-0013.xml/.bin")
        self.model = core.compile_model(model_path, "CPU")
        self.input_layer = self.model.inputs[0]
        self.output_layer = self.model.outputs[0]

        model_path_face = os.path.join(BASE_DIR, "face-detection-retail-0005.xml")
        if not os.path.exists(model_path_face):
            raise FileNotFoundError("No se encontró el modelo face-detection-retail-0005.xml/.bin")
        self.model_face = core.compile_model(model_path_face, "CPU")
        self.input_face = self.model_face.inputs[0]
        self.output_face = self.model_face.outputs[0]

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

        start_frame = int(fps * self.start_time)
        end_frame = int(fps * self.end_time)
        end_frame = min(end_frame, max(0, total_frames - 1))

        interval_frames = self.interval_frames
        count = max(0, start_frame)
        extracted = 0
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]

        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS
        ) if self.use_motion else None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)) if MORPH_OPEN_KERNEL else None

        while count <= end_frame and count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            ret, frame = cap.read()
            if not ret:
                break

            guardar = False
            motivo = "NINGUNO"
            log_info = ""

            if self.use_motion:
                fgmask = fgbg.apply(frame, learningRate=0.01)
                if MOG2_DETECT_SHADOWS and FILTER_SHADOWS:
                    _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
                if kernel is not None:
                    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                movimiento = cv2.countNonZero(fgmask)
                (h, w) = frame.shape[:2]
                umbral_mov_global = max(1, int((h * w) * MOTION_RATIO_THRESHOLD))

                if movimiento > umbral_mov_global:
                    if REQUIRE_OBJECT_FOR_SAVE:
                        object_detected = False
                        face_detected = False

                        # --- DETECCIÓN DE PERSONAS ---
                        if self.enable_person_openvino:
                            resized = cv2.resize(frame, (544, 320))
                            input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0)
                            results = self.model([input_tensor])[self.output_layer]
                            for det in results[0][0]:
                                conf = float(det[2])
                                if conf < DNN_CONFIDENCE_MIN:
                                    continue
                                x1, y1 = int(det[3] * w), int(det[4] * h)
                                x2, y2 = int(det[5] * w), int(det[6] * h)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                region = fgmask[y1:y2, x1:x2]
                                if region.size == 0:
                                    continue
                                mov_obj = cv2.countNonZero(region)
                                if mov_obj >= BBOX_MOTION_MIN:
                                    object_detected = True
                                    motivo = "PERSONA"
                                    log_info = f"conf={conf:.2f} bbox=({x1},{y1},{x2},{y2}) mov_bbox={mov_obj}"
                                    break

                        # --- DETECCIÓN DE ROSTROS ---
                        if not object_detected and self.enable_face_openvino:
                            resized_face = cv2.resize(frame, (300, 300))
                            input_tensor_face = np.expand_dims(resized_face.transpose(2, 0, 1), 0)
                            results_face = self.model_face([input_tensor_face])[self.output_face]
                            for det in results_face[0][0]:
                                conf_face = float(det[2])
                                if conf_face < DNN_CONFIDENCE_MIN:
                                    continue
                                x1, y1 = int(det[3] * w), int(det[4] * h)
                                x2, y2 = int(det[5] * w), int(det[6] * h)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                face_detected = True
                                motivo = "ROSTRO"
                                log_info = f"conf_face={conf_face:.2f} bbox=({x1},{y1},{x2},{y2})"
                                break

                        guardar = object_detected or face_detected
                    else:
                        guardar = True
                        motivo = "MOVIMIENTO"
                        log_info = f"mov_global={movimiento} umbral={umbral_mov_global}"

            if guardar:
                # Ajuste para exportar en tamaño YOLOv8 si se elige "640x640 (YOLOv8)"
                if self.resize_option != "Original":
                    if "640x360" in self.resize_option:
                        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

                    elif "960x540" in self.resize_option:
                        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)

                    elif "1280x720" in self.resize_option:
                        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

                    elif "960x960" in self.resize_option:
                        frame = cv2.resize(frame, (960, 960), interpolation=cv2.INTER_AREA)

                    elif "640x640" in self.resize_option:
                        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
                if self.show_timestamp:
                    tiempo_segundos = int(count / fps) if fps > 0 else 0
                    horas = tiempo_segundos // 3600
                    minutos = (tiempo_segundos % 3600) // 60
                    segundos = tiempo_segundos % 60
                    texto_tiempo = f"{horas:02}:{minutos:02}:{segundos:02}"
                    (h, w) = frame.shape[:2]
                    pos = (10, h - 10)
                    cv2.putText(frame, texto_tiempo, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{base_name}_frame_{count}_sec_{int(count/fps) if fps>0 else 0}_{timestamp}{self.image_format}"
                cv2.imwrite(os.path.join(self.output_dir, filename), frame)
                print(f"[SAVE-{motivo}] {filename}")
                extracted += 1

                if self.debug_mode:
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {filename} | motivo={motivo} | {log_info}\n")

                if ENABLE_SKIP_AFTER_SAVE:
                    count += int(fps * SKIP_AFTER_SAVE_SECONDS)
                    continue

            progress = int((count / max(1, total_frames)) * 100)
            self.progress_updated.emit(progress)
            count += interval_frames

        cap.release()
        self.progress_updated.emit(100)
        self.finished.emit(extracted)


class FrameExtractor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extractor de Imágenes (Auditoría de Video)")
        self.video_path = ""
        self.output_dir = ""

        self.load_btn = QPushButton("Cargar Video")
        self.load_btn.clicked.connect(self.load_video)

        self.frames_input = QDoubleSpinBox()
        self.frames_input.setMinimum(1)
        self.frames_input.setMaximum(1000)
        self.frames_input.setSingleStep(1)
        self.frames_input.setValue(10)
        self.frames_input.setSuffix(" frames")

        self.format_combo = QComboBox()
        self.format_combo.addItems([".jpg"])

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)

        self.extract_btn = QPushButton("Procesar y Guardar")
        self.extract_btn.clicked.connect(self.extract_frames)

        self.label = QLabel("Ruta del video no cargada.")
        self.label.setWordWrap(True)

        self.start_time_input = QTimeEdit()
        self.start_time_input.setDisplayFormat("HH:mm:ss")
        self.start_time_input.setTime(QTime(0, 0, 0))

        self.end_time_input = QTimeEdit()
        self.end_time_input.setDisplayFormat("HH:mm:ss")
        self.end_time_input.setTime(QTime(0, 5, 0))

        self.resize_combo = QComboBox()
        #self.resize_combo.addItems(["640x640 (YOLOv8)", "Original"])
        self.resize_combo.addItems([
            "Original",
            "640x360 (16:9)",
            "960x540 (16:9)",
            "1280x720 (16:9)",
            "960x960 (cuadrado)",
            "640x640 (cuadrado)"
        ])

        self.motion_checkbox = QCheckBox("Activar detección por movimiento")
        self.motion_checkbox.setChecked(True)

        # NUEVOS CHECKBOXES
        self.person_openvino_checkbox = QCheckBox("Usar detección de personas (OpenVINO)")
        self.person_openvino_checkbox.setChecked(True)
        self.face_openvino_checkbox = QCheckBox("Usar detección de rostros (OpenVINO)")
        self.face_openvino_checkbox.setChecked(True)

        self.text_overlay_checkbox = QCheckBox("Mostrar tiempo en la captura (HH:MM:SS)")
        self.text_overlay_checkbox.setChecked(True)

        self.debug_checkbox = QCheckBox("Debug mode")
        self.debug_checkbox.setChecked(False)

        self.skip_save_checkbox = QCheckBox("Salto de 60s tras captura")
        self.skip_save_checkbox.setChecked(ENABLE_SKIP_AFTER_SAVE)

        self.skip_seconds_input = QDoubleSpinBox()
        self.skip_seconds_input.setMinimum(0)
        self.skip_seconds_input.setMaximum(600)
        self.skip_seconds_input.setSingleStep(10)
        self.skip_seconds_input.setValue(SKIP_AFTER_SAVE_SECONDS)
        self.skip_seconds_input.setSuffix(" s")

        layout = QVBoxLayout()
        layout.addWidget(self.load_btn)
        layout.addWidget(self.frames_input)
        layout.addWidget(self.skip_save_checkbox)
        layout.addWidget(self.skip_seconds_input)
        layout.addWidget(QLabel("Formato de imagen:"))
        layout.addWidget(self.format_combo)
        layout.addWidget(self.label)
        layout.addWidget(QLabel("Tamaño de imagen:"))
        layout.addWidget(self.resize_combo)
        layout.addWidget(QLabel("Hora inicio:"))
        layout.addWidget(self.start_time_input)
        layout.addWidget(QLabel("Hora fin:"))
        layout.addWidget(self.end_time_input)
        layout.addWidget(self.motion_checkbox)
        layout.addWidget(self.person_openvino_checkbox)
        layout.addWidget(self.face_openvino_checkbox)
        layout.addWidget(self.text_overlay_checkbox)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.debug_checkbox)
        self.setLayout(layout)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.label.setText(f"Video cargado: {os.path.basename(path)}")
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps and fps > 0:
                    duration_seconds = int(total_frames / fps)
                    duration_qtime = QTime(0, 0).addSecs(duration_seconds)
                    self.start_time_input.setTime(QTime(0, 0))
                    self.end_time_input.setTime(duration_qtime)

    def extract_frames(self):
        start_qtime = self.start_time_input.time()
        end_qtime = self.end_time_input.time()
        start_time = QTime(0, 0).secsTo(start_qtime)
        end_time = QTime(0, 0).secsTo(end_qtime)

        resize_option = self.resize_combo.currentText()

        if not self.video_path:
            QMessageBox.warning(self, "Error", "Primero debes cargar un video.")
            return

        interval_frames_user = int(self.frames_input.value())
        output_dir = QFileDialog.getExistingDirectory(self, "Selecciona carpeta de destino")
        if not output_dir:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.extract_btn.setEnabled(False)

        image_format = self.format_combo.currentText().lower()
        show_timestamp = self.text_overlay_checkbox.isChecked()
        debug_mode = self.debug_checkbox.isChecked()

        global ENABLE_SKIP_AFTER_SAVE, SKIP_AFTER_SAVE_SECONDS
        ENABLE_SKIP_AFTER_SAVE = self.skip_save_checkbox.isChecked()
        SKIP_AFTER_SAVE_SECONDS = int(self.skip_seconds_input.value())

        self.thread = FrameExtractorThread(
            self.video_path, output_dir, interval_frames_user,
            image_format, start_time, end_time, resize_option,
            self.motion_checkbox.isChecked(),
            show_timestamp,
            debug_mode
        )

        # NUEVO: aplicar los checks al hilo
        self.thread.enable_person_openvino = self.person_openvino_checkbox.isChecked()
        self.thread.enable_face_openvino = self.face_openvino_checkbox.isChecked()

        self.thread.progress_updated.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, total):
        self.extract_btn.setEnabled(True)
        QMessageBox.information(self, "Completado", f"Se extrajeron {total} imágenes.")
        self.progress_bar.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    extractor = FrameExtractor()
    extractor.resize(500, 300)
    extractor.show()
    sys.exit(app.exec_())
