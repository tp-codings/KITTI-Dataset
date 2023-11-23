import numpy as np
import pykitti
from data import parseTrackletXML as xmlParser
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from data.utilities import print_progress

# Change this to the directory where you store KITTI data
basedir = 'data'
colors = {
    'Car': (0.0, 0.0, 1.0),  # Blau
    'Tram': (1.0, 0.0, 0.0),  # Rot
    'Cyclist': (0.0, 1.0, 0.0),  # Grün
    'Van': (0.0, 1.0, 1.0),  # Cyan
    'Truck': (1.0, 0.0, 1.0),  # Magenta
    'Pedestrian': (1.0, 1.0, 0.0),  # Gelb
    'Sitter': (0.0, 0.0, 0.0)  # Schwarz
}

axes_limits = [
    [-20, 40], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']


def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.
    
    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)

def draw_pygame_box(vertices, axes=[0, 1, 2], color=(0, 0, 0)):
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for connection in connections:
        glBegin(GL_LINES)
        glColor3fv(color)
        for vertex in connection:
            glVertex3fv(vertices[:, vertex])
        glEnd()
def init():
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (800 / 600), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -70)


def update_pygame(frame, dataset_velo, tracklet_rects, tracklet_types, colors, points=1.0):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.7, 0.7, 0.7, 1.0)
    glPushMatrix()

    points_step = int(1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]
    glBegin(GL_POINTS)
    glColor3fv((0.0, 0.0, 0.0))

    for point in velo_frame[:, [0, 1, 2]]:
        glVertex3fv(point)
    glEnd()

    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_pygame_box(t_rects, axes=[0, 1, 2], color=colors[t_type])

    glPopMatrix()
    pygame.display.flip()
    pygame.time.wait(10)

def draw_3d_plots_pygame(dataset, tracklet_rects, tracklet_types, points=1.0):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init()

    dataset_velo = list(dataset.velo)

    for frame in range(len(dataset_velo)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        update_pygame(frame, dataset_velo, tracklet_rects, tracklet_types, colors, points)

    pygame.quit()


if __name__ == "__main__":
    date = '2011_09_26'
    drive = '0001'
    dataset = load_dataset(date, drive)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), 'data/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

    draw_3d_plots_pygame(dataset, tracklet_rects, tracklet_types)