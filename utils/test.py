import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np

# Funktion zum Erstellen eines Vertex Buffer Objects (VBO) für die Punktwolke
def create_point_vbo():
    points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        # Füge weitere Punkte hinzu...
    ], dtype=np.float32)

    vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)

    return vbo_id

# Funktion zum Zeichnen der Punktwolke mit VBO
def draw_points_with_vbo(vbo_id):
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glColor3f(1.0, 1.0, 1.0)
    glDrawArrays(GL_POINTS, 0, 2)  # Anzahl der Punkte anpassen

    glDisableClientState(GL_VERTEX_ARRAY)

# Funktion zum Zeichnen des 2D-Rechtecks am oberen Rand
def draw_2d_rectangle():
    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)

    glVertex2f(0, 0.9)
    glVertex2f(1, 0.9)
    glVertex2f(1, 1)
    glVertex2f(0, 1)

    glEnd()

# Pygame Initialisierung
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# OpenGL Initialisierung
glOrtho(0, 1, 0, 1, -1, 1)  # Orthografische Projektion für 2D
glTranslatef(0.0, 0.0, -5)

# VBO für Punktwolke erstellen
point_vbo_id = create_point_vbo()

# Haupt-Event-Schleife
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 3, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_points_with_vbo(point_vbo_id)
    
    glLoadIdentity()  # Zurücksetzen der Modellview-Matrix für 2D-Zeichnungen
    draw_2d_rectangle()

    pygame.display.flip()
    pygame.time.wait(10)
