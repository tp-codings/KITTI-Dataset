import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *

# Annahme: dataset_velo ist Ihre Punktwolke

def load_point_vbo(data):
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    return vbo

def render_points(dataset_velo, points_step):
    velo_range = range(0, dataset_velo.shape[0], points_step)

    vbo = load_point_vbo(dataset_velo[velo_range, :-1])

    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)

    # Farbzuweisung basierend auf der Höhe
    heights = dataset_velo[velo_range, -1]
    min_height, max_height = heights.min(), heights.max()

    colors = (heights - min_height) / (max_height - min_height)
    colors = np.column_stack([1 - colors, colors, np.zeros_like(colors)])

    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, colors)

    glPointSize(1.3)
    glDrawArrays(GL_POINTS, 0, len(velo_range))

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    render_points(dataset_velo, points_step)
    glutSwapBuffers()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutCreateWindow("Colored Point Cloud")
    glutDisplayFunc(display)
    glutMainLoop()

if __name__ == "__main__":
    # Setzen Sie Ihre Punktwolke und Schrittweite
    dataset_velo = np.random.rand(1000, 4)  # Beispiel-Punktwolke mit 4 Dimensionen (x, y, z, Höhe)
    points_step = 1

    main()
