import sys
import math
import numpy as np
import cv2
import json
import os
from PIL import Image

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt, Signal, Slot, QTimer
import moderngl

# ==========================================
# GLSL Shaders
# ==========================================

VERTEX_SHADER = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D tex;
uniform float fov_rad;
uniform float aspect_ratio;
uniform float pitch;
uniform float yaw;

in vec2 v_uv;
out vec4 f_color;

mat3 eulerToMatrix(float pitch, float yaw) {
    float cp = cos(pitch);
    float sp = sin(pitch);
    float cy = cos(yaw);
    float sy = sin(yaw);
    mat3 Rx = mat3(1.0, 0.0,  0.0, 0.0, cp,  -sp, 0.0, sp,   cp);
    mat3 Ry = mat3(cy,  0.0, -sy, 0.0, 1.0,  0.0, sy,  0.0,  cy);
    return Ry * Rx;
}

void main() {
    vec2 pos = v_uv * 2.0 - 1.0;
    pos.y /= aspect_ratio;
    
    float z = 1.0 / tan(fov_rad / 2.0);
    vec3 ray = normalize(vec3(pos.x, pos.y, -z));
    
    mat3 rot = eulerToMatrix(pitch, yaw);
    ray = rot * ray;
    
    float theta = atan(ray.x, -ray.z);
    float phi = asin(ray.y);
    float pi = 3.14159265359;
    float tex_u = (theta / (2.0 * pi)) + 0.5;
    float tex_v = (phi / pi) + 0.5;
    
    f_color = texture(tex, vec2(tex_u, tex_v));
}
"""

# ==========================================
# OpenGL ViewPort Widget
# ==========================================

class PanoViewWidget(QOpenGLWidget):
    def __init__(self, yaw_offset_deg=0.0, frameless=False):
        super().__init__()
        self.yaw_offset_deg = yaw_offset_deg
        if frameless:
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
            
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.texture = None
        
        self.raw_image_data = None
        self.raw_width = 0
        self.raw_height = 0
        
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.fov_deg = 90.0
        self.aspect_ratio = 1.0
        
    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
        if self.raw_image_data is not None:
            self._create_gl_texture()
            
    def _create_gl_texture(self):
        if self.texture:
            self.texture.release()
        self.texture = self.ctx.texture((self.raw_width, self.raw_height), 3, self.raw_image_data)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = True
        self.texture.repeat_y = False
        
    def resizeGL(self, w, h):
        if self.ctx is None: return
        self.ctx.viewport = (0, 0, w, h)
        if h > 0: self.aspect_ratio = w / h
            
    def paintGL(self):
        if self.ctx is None: return
        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        fbo.clear(0.1, 0.1, 0.1)
        if self.texture is None: return
        self.texture.use(0)
        if 'tex' in self.prog: self.prog['tex'].value = 0
        if 'aspect_ratio' in self.prog: self.prog['aspect_ratio'].value = self.aspect_ratio
        if 'fov_rad' in self.prog: self.prog['fov_rad'].value = math.radians(self.fov_deg)
        
        final_yaw = self.yaw_deg + self.yaw_offset_deg
        if 'pitch' in self.prog: self.prog['pitch'].value = math.radians(self.pitch_deg)
        if 'yaw' in self.prog: self.prog['yaw'].value = math.radians(final_yaw)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
    def update_texture_data(self, image_data):
        self.raw_image_data = image_data
        if self.texture is not None:
            self.makeCurrent()
            self.texture.write(self.raw_image_data)
            self.update()

    def set_texture_data(self, image_data, width, height):
        self.raw_image_data = image_data
        self.raw_width = width
        self.raw_height = height
        if self.ctx is not None:
            self.makeCurrent()
            self._create_gl_texture()
            self.update()
        
    def update_view(self, pitch, yaw, fov):
        self.pitch_deg = pitch
        self.yaw_deg = yaw
        self.fov_deg = fov
        self.update()

# ==========================================
# Main Application Window
# ==========================================

class MainWindow(QMainWindow):
    camera_updated = Signal(float, float, float)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Triptych360 Controller")
        self.resize(500, 300)
        
        # Load Config
        self.config_path = "config.json"
        self.config = {"yaw": 0.0, "pitch": 0.0, "fov": 90.0, "frameless": False, "last_media": None}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    self.config.update(json.load(f))
            except: pass
            
        self.pitch = self.config["pitch"]
        self.yaw = self.config["yaw"]
        self.fov = self.config["fov"]
        self.idle_time = 0.0
        
        self.dragging = False
        self.last_mouse_pos = None
        self.image_path = None
        self.image_data = None
        self.image_w = 0
        self.image_h = 0
        
        self.video_cap = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.process_video_frame)
        
        self.sm_device = None
        self.init_spacemouse()
        self.sm_timer = QTimer(self)
        self.sm_timer.timeout.connect(self.poll_spacemouse)
        self.sm_timer.start(16)
        
        # Create View Windows
        self.view_windows = []
        is_frameless = self.config.get("frameless", False)
        for i, offset in enumerate([-90.0, 0.0, 90.0]):
            view = PanoViewWidget(yaw_offset_deg=offset, frameless=is_frameless)
            view.setWindowTitle(["Left Wall View", "Front Wall View", "Right Wall View"][i])
            view.resize(800, 800)
            self.camera_updated.connect(view.update_view)
            self.view_windows.append(view)
            
        # ==========================================
        # macOS-like UI Styling
        # ==========================================
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
            }
            QPushButton {
                background-color: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0069D9;
            }
            QPushButton:pressed {
                background-color: #0056B3;
            }
            QLabel#TitleLabel {
                font-size: 14px;
                font-weight: 600;
            }
        """)
        
        # UI Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        self.info_lbl = QLabel("No media loaded.")
        self.info_lbl.setObjectName("TitleLabel")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setWordWrap(True)
        layout.addWidget(self.info_lbl)
        
        btn_load = QPushButton("Load Equirectangular Media")
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self.load_media_prompt)
        layout.addWidget(btn_load)
        
        self.frameless_cb = QCheckBox("Kiosk Mode (Frameless, auto-snaps to monitors on launch)")
        self.frameless_cb.setChecked(is_frameless)
        self.frameless_cb.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.frameless_cb)
        
        lbl_instruct = QLabel("Drag to rotate. Scroll to Zoom. Auto-pans when idle.")
        lbl_instruct.setAlignment(Qt.AlignCenter)
        lbl_instruct.setStyleSheet("color: #7c7c80; font-size: 12px;")
        layout.addWidget(lbl_instruct)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Auto-load previous media
        if self.config.get("last_media") and os.path.exists(self.config["last_media"]):
            # Small delay to ensure windows map correctly after startup
            QTimer.singleShot(500, lambda: self.load_media_file(self.config["last_media"]))

    def position_windows(self):
        screens = QApplication.screens()
        for idx, view in enumerate(self.view_windows):
            if self.config.get("frameless", False) and len(screens) > 1:
                screen_idx = min(idx, len(screens) - 1)
                view.setGeometry(screens[screen_idx].geometry())
                view.showFullScreen()
            else:
                view.show()

    def reset_idle(self):
        self.idle_time = 0.0

    def load_media_prompt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open 360 Media", "", "Media (*.png *.jpg *.jpeg *.mp4 *.mkv *.avi *.mov)")
        if path:
            self.load_media_file(path)
            
    def load_media_file(self, path):
        self.video_timer.stop()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
            
        sm_status = " | SpaceMouse ✅" if self.sm_device else " | UI Idle Pan ✅"
        
        if str(path).lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            self.video_cap = cv2.VideoCapture(path)
            if not self.video_cap.isOpened():
                self.info_lbl.setText("Error: Could not open video file.")
                return
            self.image_w = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.image_h = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                self.image_data = frame.tobytes()
                self.image_path = path
                self.info_lbl.setText(f"Loaded Video: {os.path.basename(path)} @ {fps}fps{sm_status}")
                for view in self.view_windows:
                    view.set_texture_data(self.image_data, self.image_w, self.image_h)
                self.position_windows()
                self.video_timer.start(int(1000 / fps))
        else:
            try:
                img = Image.open(path).convert('RGB')
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                self.image_w, self.image_h = img.size
                self.image_data = img.tobytes()
                self.image_path = path
                self.info_lbl.setText(f"Loaded Image: {os.path.basename(path)}{sm_status}")
                for view in self.view_windows:
                    view.set_texture_data(self.image_data, self.image_w, self.image_h)
                self.position_windows()
            except Exception as e:
                self.info_lbl.setText(f"Error loading image: {str(e)}")

        self.camera_updated.emit(self.pitch, self.yaw, self.fov)

    def process_video_frame(self):
        if not self.video_cap: return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret: return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        self.image_data = frame.tobytes()
        for view in self.view_windows:
            view.update_texture_data(self.image_data)

    def init_spacemouse(self):
        try:
            import pyspacemouse
            self.sm_device = pyspacemouse.open()
        except:
            self.sm_device = None
            
    def poll_spacemouse(self):
        # Handle Idle panning globally
        self.idle_time += 16.0
        if self.idle_time > 5000.0 and self.image_data:
            self.yaw += 0.03
            self.yaw %= 360.0
            self.camera_updated.emit(self.pitch, self.yaw, self.fov)
            
        if self.sm_device and self.image_data:
            try:
                state = self.sm_device.read()
                if state:
                    yaw_input = getattr(state, 'roll', 0.0) 
                    pitch_input = getattr(state, 'pitch', 0.0)
                    if abs(yaw_input) > 0.01 or abs(pitch_input) > 0.01:
                        self.reset_idle()
                        sens = 1.5 
                        self.yaw += yaw_input * sens
                        self.pitch += pitch_input * sens
                        self.pitch = max(-85.0, min(85.0, self.pitch))
                        self.yaw %= 360.0
                        self.camera_updated.emit(self.pitch, self.yaw, self.fov)
            except: pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.position()
            self.reset_idle()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            
    def mouseMoveEvent(self, event):
        if self.dragging and self.image_data:
            self.reset_idle()
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()
            sens = 0.2
            self.yaw -= delta.x() * sens
            self.pitch += delta.y() * sens
            self.pitch = max(-85.0, min(85.0, self.pitch))
            self.yaw %= 360.0
            self.camera_updated.emit(self.pitch, self.yaw, self.fov)
            
    def wheelEvent(self, event):
        self.reset_idle()
        delta = event.angleDelta().y()
        self.fov -= delta * 0.05
        self.fov = max(30.0, min(150.0, self.fov))
        self.camera_updated.emit(self.pitch, self.yaw, self.fov)

    def closeEvent(self, event):
        self.config["yaw"] = self.yaw
        self.config["pitch"] = self.pitch
        self.config["fov"] = self.fov
        self.config["frameless"] = self.frameless_cb.isChecked()
        self.config["last_media"] = self.image_path
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f)
        except: pass
        
        for view in self.view_windows:
            view.close()
        super().closeEvent(event)

if __name__ == '__main__':
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
