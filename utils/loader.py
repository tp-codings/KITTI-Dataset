from utils import parseTrackletXML as xmlParser
from OpenGL.GL import *
import pykitti
import numpy as np
import os
from utils.utilities import incrementString

class single_raw:
    def __init__(self, basedir, currentFrame, **kwargs):
        self.velo_path = os.path.join(basedir, "Live", "velodyne_points", "data")
        self.oxts_path = os.path.join(basedir, "Live", "oxts", "data")
        self.calib_path = os.path.join(basedir, "Live")
        self.data_path = os.path.join(basedir, 'Live')
        self.imtype = kwargs.get('imtype', 'png')
        self.currentFrame = currentFrame
        
    @property
    def velo(self):
        file_path = os.path.join(self.velo_path, self.currentFrame + ".bin")
        scan = np.fromfile(file_path, dtype=np.float32)
        next_frame = incrementString(self.currentFrame)
        file_path = os.path.join(self.velo_path, next_frame + ".bin")
        if not os.path.exists(file_path):
            next_frame = self.currentFrame
            print(next_frame)
        return scan.reshape((-1, 4)), next_frame
    
    @property
    def oxts(self):
        file_path = os.path.join(self.oxts_path, self.currentFrame + ".txt")

        with open(file_path, 'r') as file:
            first_line = file.readline()

        oxts = [float(zahl) for zahl in first_line.split()[:3]]

        return oxts[0], oxts[1], oxts[2]
        


def load_dataset(basedir, date, drive, calibrated=False, frame_range=None):
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
        dataset._load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset

def load_current_data(basedir, currentFrame, calibrated=False, frame_range=None):

    data = single_raw(basedir, currentFrame)

    return data


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
            frame_tracklets[n_frames] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)

def load_point_vbo(vertices):
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo
