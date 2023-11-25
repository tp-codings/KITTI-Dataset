import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

# Definiere die Eckpunkte des Würfels
vertices = np.array([
    # Vorderseite
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    # Rückseite
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
], dtype=np.float32)

# Definiere die Indizes der Dreiecke
indices = np.array([
    0, 1, 2, 2, 3, 0,  # Vorderseite
    4, 5, 6, 6, 7, 4,  # Rückseite
    1, 5, 6, 6, 2, 1,  # Rechte Seite
    0, 4, 7, 7, 3, 0,  # Linke Seite
    3, 2, 6, 6, 7, 3,  # Oben
    0, 1, 5, 5, 4, 0   # Unten
], dtype=np.uint32)

def render_cube(x, y, z, scale):
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    #glPushMatrix()  # Speichere den aktuellen Zustand der Modelview-Matrix
    glScalef(scale, scale, scale)  
    glTranslatef(x, y, z)  # Setze die Position des Würfels
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
    #glPopMatrix()  # Stelle den vorherigen Zustand der Modelview-Matrix wieder her
    glDisableClientState(GL_VERTEX_ARRAY)