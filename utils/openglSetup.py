from OpenGL.GL import *
from OpenGL.GLU import *

def init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    gluPerspective(45, (800 / 600), 0.1, 10000.0)
    glTranslatef(0.0, 0.0, -70)