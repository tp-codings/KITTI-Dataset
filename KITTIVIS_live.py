from pygame.locals import DOUBLEBUF, OPENGL, MOUSEBUTTONDOWN, MOUSEBUTTONUP
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from utils.loader import load_current_data, load_tracklets_for_frames, load_point_vbo, simulate_tracklets, load_cube_vbo
from utils.drawing import render_text, draw_pygame_box
from utils.sceneManip import rotate_scene
from utils.openglSetup import init
from utils.settings import colors, basedir
from utils.getGeo import get_maxspeed, get_location
import pygame
from OpenGL.GL import shaders
import numpy as np
from ctypes import sizeof, c_float, c_void_p

from io import BytesIO


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

def render_splash(image):
       # using resources in open gl generally follows the form of generate, bind, modify

    # Generate: request a buffer for our vertices
    image_vbo = glGenBuffers(2)

    # Bind: set the newly requested buffer as the active GL_ARRAY_BUFFER. 
    #   All subsequent modifications of GL_ARRAY_BUFFER will affect our vbo
    glBindBuffer(GL_ARRAY_BUFFER, image_vbo)

    # Modify: Tell OpenGL to load data into the buffer. 

    # I've added two more coordinates to each vertex here for determining the position within the texture.
    # These two additional coordinates are typically refered to as uv coordinates.
    # Also there are now two triangles that cover the entire viewport.
    vertex_data = np.array([-1, 0.6, 0, 0,  -1, 1, 0, 1,  1, 1, 1, 1,  -1, 0.6, 0, 0,  1, 1, 1, 1,  1, 0.6, 1, 0], np.float32)
    glBufferData(GL_ARRAY_BUFFER, vertex_data, GL_STATIC_DRAW)

    vertex_position_attribute_location = 0
    uv_attribute_location = 1

    # glVertexAttribPointer basically works in the same way as glVertexPointer with two exceptions:
    #   First, it can be used to set the data source for any vertex attributes.
    #   Second, it has an option to normalize the data, which I have set to GL_FALSE.
    glVertexAttribPointer(vertex_position_attribute_location, 2, GL_FLOAT, GL_FALSE, sizeof(c_float)*4, c_void_p(0))
    # vertex attributes need to be enabled
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(uv_attribute_location, 2, GL_FLOAT, GL_FALSE, sizeof(c_float)*4, c_void_p(sizeof(c_float)*2))
    glEnableVertexAttribArray(1)

    # Generate: request a texture
    image_texture = glGenTextures(1)

    # Bind: set the newly requested texture as the active GL_TEXTURE_2D.
    #   All subsequent modifications of GL_TEXTURE_2D will affect our texture (or how it is used)
    glBindTexture(GL_TEXTURE_2D, image_texture)


    width = image.get_width()
    height = image.get_height()

    # retrieve a byte string representation of the image.
    # The 3rd parameter tells pygame to return a vertically flipped image, as the coordinate system used
    # by pygame differs from that used by OpenGL
    image_data = pygame.image.tostring(image, "RGBA", True)

    # Modify: Tell OpenGL to load data into the image
    mip_map_level = 0
    glTexImage2D(GL_TEXTURE_2D, mip_map_level, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)

    # set the filtering mode for the texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    image_vertex_shader = shaders.compileShader("""
        #version 330
        layout(location = 0) in vec2 pos;
        layout(location = 1) in vec2 uvIn;
        out vec2 uv;
        void main() {
            gl_Position = vec4(pos, 0, 1);
            uv = uvIn;
        }
        """, GL_VERTEX_SHADER)

    image_fragment_shader = shaders.compileShader("""
        #version 330
        out vec4 fragColor;
        in vec2 uv;
        uniform sampler2D tex;
        void main() {
            fragColor = texture(tex, uv);
        }
    """, GL_FRAGMENT_SHADER)

    shader_program_2 = shaders.compileProgram(image_vertex_shader, image_fragment_shader)


    #glEnableClientState(GL_VERTEX_ARRAY)

    # Enable alpha blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glUseProgram(shader_program_2)

    glDrawArrays(GL_TRIANGLES, 0, 6)

    #glDisableClientState(GL_VERTEX_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(0)
    
    
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

    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glUseProgram(shader_program)

    modelviewprojection_loc = glGetUniformLocation(shader_program, "modelviewprojection")
    modelviewprojection = glGetFloatv(GL_MODELVIEW_MATRIX)
    projection = glGetFloatv(GL_PROJECTION_MATRIX)
    glUniformMatrix4fv(modelviewprojection_loc, 1, GL_FALSE, np.dot(projection, modelviewprojection))

    glPointSize(1.5)

    num_points = len(velo_range)
    if num_points > 0:
        glDrawArrays(GL_POINTS, 0, num_points)
    glDisableClientState(GL_VERTEX_ARRAY)

    # Deaktiviere die Shader und Vertex-Attributarrays
    glUseProgram(0)

    if tracklet_rects is not None and tracklet_types is not None:
        for t_rects, t_type in zip(tracklet_rects, tracklet_types):
            draw_pygame_box(t_rects, axes=[0, 1, 2], color=colors.get(t_type, (1.0, 1.0, 1.0)))

    glPopMatrix()

    return rotation_angles, dragging, initial_mouse_pos, zoom_factor



def draw_3d_plots_pygame(points=1.0):
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
        clock.tick(30)  # Limit to 60 frames per second
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        data = load_current_data(basedir, next_frame, calibrated=False)

        data_velo, next_frame = data.velo
        latitude, longitude, height = data.oxts
        cam00 = data.cam00

        print(cam00)




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
        if current_time - last_update_time >= 50000:
            location = str(get_location(latitude, longitude))
            speed_limit = str(get_maxspeed(str(latitude), str(longitude), str(100)))
            last_update_time = current_time

        elif first_frame: 
            location = str(get_location(latitude, longitude))
            speed_limit = str(get_maxspeed(str(latitude), str(longitude), str(100)))
            first_frame = False


        render_text(-35, 25, str(round(fps, 2)))
        render_text(-35, 23, "Latitude: " + str(round(latitude, 6)))
        render_text(-35, 21, "Longitude: " + str(round(longitude, 6)))
        render_text(-35, 19, "Height: " + str(round(height, 2)))
        render_text(-35, 17, "Ort: " + location)
        render_text(-35, 15, "Speedlimit: " + speed_limit)


        rotation_angles, dragging, initial_mouse_pos, zoom_factor = update_pygame(data_velo, tracklet_rects, tracklet_types, colors, 1.0, rotation_angles, dragging, initial_mouse_pos, zoom_factor)
        render_splash(cam00)
        

        pygame.display.flip()


    pygame.quit()

if __name__ == "__main__":
    draw_3d_plots_pygame()