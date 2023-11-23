from OpenGL.GL import *

def rotate_scene(angle_x, angle_y, angle_z):
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    glRotatef(angle_z, 0, 0, 1)