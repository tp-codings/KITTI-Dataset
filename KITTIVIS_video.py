from pygame.locals import DOUBLEBUF, OPENGL, MOUSEBUTTONDOWN, MOUSEBUTTONUP
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from utils.loader import load_dataset, load_tracklets_for_frames, load_point_vbo
import pygame

basedir = 'data'

colors = {
    'Car': (0.0, 0.0, 1.0),  # Blau
    'Tram': (1.0, 0.0, 0.0),  # Rot
    'Cyclist': (0.0, 1.0, 0.0),  # Grün
    'Van': (0.0, 1.0, 1.0),  # Cyan
    'Truck': (1.0, 0.0, 1.0),  # Magenta
    'Pedestrian': (1.0, 1.0, 0.0),  # Gelb
    'Sitter': (0.0, 0.0, 0.0),  # Schwarz
    'Misc' : (0.0, 0.4, 1.0)
}

def rotate_scene(angle_x, angle_y, angle_z):
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    glRotatef(angle_z, 0, 0, 1)

def init():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    gluPerspective(45, (800 / 600), 0.1, 1000.0)
    glTranslatef(0.0, 0.0, -70)
    
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


def update_pygame(frame, dataset_velo, tracklet_rects, tracklet_types, colors, points=1.0, rotation_angles=(0, 0, 0), dragging=False, initial_mouse_pos=None, zoom_factor=1.0):

    print(dragging)
    if dragging:
        rel_x, rel_y = pygame.mouse.get_pos()[0] - initial_mouse_pos[0], pygame.mouse.get_pos()[1] - initial_mouse_pos[1]
        rotation_angles = (
            rotation_angles[0] + rel_y * 0.1,  
            rotation_angles[1],  
            rotation_angles[2] + rel_x * 0.1
        )

    glPushMatrix()

    glTranslatef(0.0, 0.0, -70 * zoom_factor)
    rotate_scene(*rotation_angles)

    points_step = int(1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)

    vbo = load_point_vbo(dataset_velo[frame][velo_range, :-1])
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glColor3fv((1.0, 1.0, 1.0))
    glPointSize(1.3)

    num_points = len(velo_range)
    if num_points > 0:
        glDrawArrays(GL_POINTS, 0, num_points)
    glDisableClientState(GL_VERTEX_ARRAY)

    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_pygame_box(t_rects, axes=[0, 1, 2], color=colors[t_type])

    glPopMatrix()

    return rotation_angles, dragging, initial_mouse_pos, zoom_factor

def render_text(x, y, text):                                                
    position = (x, y, 0)
    font = pygame.font.SysFont('arial', 20)
    textSurface = font.render(text, True, (255, 255, 66, 255)).convert_alpha()
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


def draw_3d_plots_pygame(dataset, tracklet_rects, tracklet_types, points=1.0):
    pygame.init()
    display = (800, 600)
    pygame.display.set_caption("KITTI Visualization")
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init()

    pygame.font.init()
    font = pygame.font.SysFont(None, 30)

    clock = pygame.time.Clock()

    dataset_velo = list(dataset.velo)
    rotation_angles = (0, 0, 0)

    dragging = False
    initial_mouse_pos = None
    zoom_factor = 1.0

    frame = 0
    running = True

    while running:
        clock.tick(30)  # Limit to 60 frames per second
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    dragging = True
                    initial_mouse_pos = pygame.mouse.get_pos()
                if event.button == 4:  # scroll wheel up
                    zoom_factor /= 1.1
                if event.button == 5:  # scroll wheel down
                    zoom_factor *= 1.1

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # left mouse button
                    dragging = False
                    initial_mouse_pos = None

        fps = clock.get_fps()
        render_text(-35, 25, str(round(fps, 2)))

        rotation_angles, dragging, initial_mouse_pos, zoom_factor = update_pygame(frame, dataset_velo, tracklet_rects, tracklet_types, colors, 1.0, rotation_angles, dragging, initial_mouse_pos, zoom_factor)


        pygame.display.flip()

        frame += 1
        if frame >= len(dataset_velo):
            frame = 0

    pygame.quit()

if __name__ == "__main__":
    date = '2011_09_26'
    drive = '0051'
    dataset = load_dataset(basedir, date, drive, calibrated=False)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), 'data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

    draw_3d_plots_pygame(dataset, tracklet_rects, tracklet_types)