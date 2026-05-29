import sys
import moderngl
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
import numpy as np

class MinimalGL(QOpenGLWidget):
    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(
            vertex_shader="#version 330\nin vec2 in_vert; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }",
            fragment_shader="#version 330\nout vec4 f_color; void main() { f_color = vec4(1.0, 0.0, 0.0, 1.0); }"
        )
        self.vbo = self.ctx.buffer(np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4'))
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
    def paintGL(self):
        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        fbo.clear(0.0, 1.0, 0.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

app = QApplication(sys.argv)
w = MinimalGL()
w.show()
QTimer.singleShot(1500, app.quit)
sys.exit(app.exec())
