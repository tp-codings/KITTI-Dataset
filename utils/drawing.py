from pygame import font, image
from OpenGL.GL import *


def render_text(x, y, text):                                                
    position = (x, y, 0)
    _font = font.SysFont('arial', 20)
    textSurface = _font.render(text, True, (255, 255, 66, 255)).convert_alpha()
    textData = image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def draw_pygame_box(vertices, axes=[0, 1, 2], color=(0, 0, 0)):
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    glLineWidth(2.0)

    for connection in connections:
        glBegin(GL_LINES)
        glColor3fv(color)
        for vertex in connection:
            glVertex3fv(vertices[:, vertex])
        glEnd()