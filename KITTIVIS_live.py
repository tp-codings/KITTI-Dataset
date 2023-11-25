from pygame.locals import DOUBLEBUF, OPENGL, MOUSEBUTTONDOWN, MOUSEBUTTONUP
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from utils.loader import load_current_data, load_tracklets_for_frames, load_point_vbo, simulate_tracklets, load_cube_vbo
from utils.drawing import render_text, draw_pygame_box
from utils.sceneManip import rotate_scene
from utils.openglSetup import init
from utils.settings import colors, basedir
#from utils.getGeo import get_maxspeed, get_location
import pygame
from OpenGL.GL import shaders
import numpy as np
from ctypes import sizeof, c_float, c_void_p
from glm import mat4, scale, translate, rotate, perspective, lookAt, vec3
from ultralytics import YOLO


# Vertex and Fragment Shader source code
vertex_shader = """
#version 330
in vec4 position;
out float height;

uniform mat4 modelviewprojection;

void main()
{
    gl_Position = modelviewprojection * position;
    height = position.z;
}
"""

fragment_shader = """
#version 330
in float height;
out vec4 FragColor;

void main()
{
    vec3 color = mix(vec3(1.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), height);
    color = mix(color, vec3(1.0, 1.0, 0.0), height);
    color = mix(color, vec3(.0, 1.0, 0.0), height);
    
    FragColor = vec4(color, 1.0);
}
"""

# Shader Program
shader_program = None

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise Exception("Shader compilation failed: {}".format(glGetShaderInfoLog(shader)))

    return shader

def link_program(vertex_shader, fragment_shader):
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise Exception("Program linking failed: {}".format(glGetProgramInfoLog(program)))

    return program

def init_shader_program():
    global shader_program
    vertex_shader_obj = compile_shader(vertex_shader, GL_VERTEX_SHADER)
    fragment_shader_obj = compile_shader(fragment_shader, GL_FRAGMENT_SHADER)
    shader_program = link_program(vertex_shader_obj, fragment_shader_obj)
    glDeleteShader(vertex_shader_obj)
    glDeleteShader(fragment_shader_obj)

def render_splash(image, position, scale, detections):
    x = position[0]
    y = position[1]
    width, height = image.get_width(), image.get_height()
    texture_data = pygame.image.tostring(image, "RGBA", True)

    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    tl = (x/20, y/20)
    tr = (x/20 + width*scale/20, y/20)

    br = (x/20 + width*scale/20, y/20 - height*scale/20)
    bl = (x/20, y/20 - height*scale/20)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(tl[0], tl[1])  # Oben links
    glTexCoord2f(1, 1)
    glVertex2f(tr[0], tr[1])  # Oben rechts
    glTexCoord2f(1, 0)
    glVertex2f(br[0], br[1])  # Unten rechts
    glTexCoord2f(0, 0)
    glVertex2f(bl[0], bl[1])  # Unten links
    glEnd()

    glDeleteTextures([texture_id])
    glDisable(GL_TEXTURE_2D)
    
    
def update_pygame(dataset_velo, tracklet_rects, tracklet_types, colors, points=1.0, rotation_angles=(0, 0, 0), dragging=False, initial_mouse_pos=None, zoom_factor=1.0):
    if dragging:
        rel_x, rel_y = pygame.mouse.get_pos()[0] - initial_mouse_pos[0], pygame.mouse.get_pos()[1] - initial_mouse_pos[1]
        rotation_angles = (
            rotation_angles[0] + rel_y * 0.1,
            rotation_angles[1],
            rotation_angles[2] + rel_x * 0.1
        )

    glPushMatrix()
    scale = 10
    glScalef(scale, scale, scale)  

    glTranslatef(0.0, 0.0, -70 * zoom_factor)
    rotate_scene(*rotation_angles)

    points_step = int(1. / points)
    velo_range = range(0, dataset_velo.shape[0], points_step)

    vbo = load_point_vbo(dataset_velo[velo_range, :-1])

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glUseProgram(shader_program)

    modelviewprojection_loc = glGetUniformLocation(shader_program, "modelviewprojection")
    modelviewprojection = glGetFloatv(GL_MODELVIEW_MATRIX)
    projection = glGetFloatv(GL_PROJECTION_MATRIX)
    glUniformMatrix4fv(modelviewprojection_loc, 1, GL_FALSE, np.dot(projection, modelviewprojection))

    glPointSize(1.5)
    num_points = len(velo_range)
    #print(num_points)
    if num_points > 0:
        glDrawArrays(GL_POINTS, 0, num_points)

    # Deaktiviere die Shader und Vertex-Attributarrays
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glUseProgram(0)

    if tracklet_rects is not None and tracklet_types is not None:
        for t_rects, t_type in zip(tracklet_rects, tracklet_types):
            draw_pygame_box(t_rects, axes=[0, 1, 2], color=colors.get(t_type, (1.0, 1.0, 1.0)))

    glPopMatrix()

    return rotation_angles, dragging, initial_mouse_pos, zoom_factor



def draw_3d_plots_pygame(points=1.0):
    model = YOLO("YoloWeights/yolov8n.pt")

    classNames = model.names
    pygame.init()
    display = (800, 1000)

    pygame.display.set_caption("KITTI Visualization")
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init()

    clock = pygame.time.Clock()
    last_update_time = 0

    tracklet_rects, tracklet_types = None, None    
    rotation_angles = (0, 0, 0)

    init_shader_program()

    dragging = False
    initial_mouse_pos = None
    zoom_factor = 1.0

    running = True
    next_frame = "0000000000"
    last_frame = next_frame
    first_frame = True

    #tracklet_rects, tracklet_types = simulate_tracklets()

    while running:

        clock.tick(20)  
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        data = load_current_data(basedir, next_frame, calibrated=False)

        data_velo, next_frame = data.velo
        latitude, longitude, height = data.oxts
        #cam00 = data.cam00
        #cam01 = data.cam01
        cam02 = data.cam02
        #cam03 = data.cam03

        imgdata = pygame.surfarray.array3d(cam02)
        imgdata = imgdata.swapaxes(0,1)
        results = model(imgdata, stream=True)

        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:

                #Boundingbox
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) #cv2
                w, h = x2-x1, y2-y1
                print(w, h)
                detections.append((x1, y1, w, h))




        #print(cam00)

        if last_frame != next_frame:
            #tracklet_rects, tracklet_types = simulate_tracklets()
            last_frame = next_frame


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # left mouse button
                    dragging = True
                    initial_mouse_pos = pygame.mouse.get_pos()
                if event.button == 4:  # scroll wheel up
                    zoom_factor /= 1.3
                if event.button == 5:  # scroll wheel down
                    zoom_factor *= 1.3

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:  # left mouse button
                    dragging = False
                    initial_mouse_pos = None

        #pygame.event.wait()  # Hier wird das Programm pausiert, bis eine Taste gedrückt wird
        fps = clock.get_fps()

        current_time = pygame.time.get_ticks()
        # if current_time - last_update_time >= 50000:
        #     location = str(get_location(latitude, longitude))
        #     speed_limit = str(get_maxspeed(str(latitude), str(longitude), str(100)))
        #     last_update_time = current_time

        # elif first_frame: 
        #     location = str(get_location(latitude, longitude))
        #     speed_limit = str(get_maxspeed(str(latitude), str(longitude), str(100)))
        #     first_frame = False


        render_text(-35, 25, str(round(fps, 2)))
        render_text(-35, 23, "Latitude: " + str(round(latitude, 6)))
        render_text(-35, 21, "Longitude: " + str(round(longitude, 6)))
        render_text(-35, 19, "Height: " + str(round(height, 2)))
        #render_text(-35, 17, "Ort: " + location)
        #render_text(-35, 15, "Speedlimit: " + speed_limit)

        glEnableClientState(GL_VERTEX_ARRAY)
        rotation_angles, dragging, initial_mouse_pos, zoom_factor = update_pygame(data_velo, tracklet_rects, tracklet_types, colors, 1.0, rotation_angles, dragging, initial_mouse_pos, zoom_factor)
        glDisableClientState(GL_VERTEX_ARRAY)
        # render_splash(cam00, (-800, display[1]/2+100), 0.5)
        # render_splash(cam01, (200, display[1]/2+100), 0.5)
        # render_splash(cam02, (-800, display[1]/2-100), 0.5)
        # render_splash(cam03, (200, display[1]/2-100), 0.5)
        render_splash(cam02, (200, display[1]/2+100), 0.5, detections)

        

        pygame.display.flip()


    pygame.quit()

if __name__ == "__main__":
    draw_3d_plots_pygame()