import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math

WIDTH, HEIGHT = 1080, 1080
FOV = np.pi / 3

MAX_STEPS = 100
MAX_DIST = 10
EPSILON = 0.001

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Ray Marching with PyOpenGL")

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    
    return shader

vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 position;
out vec2 uv;
void main() {
    gl_Position = vec4(position, 1.0);
    uv = (position.xy + 1.0) / 2.0;
}
"""

fragment_shader_code = """
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform vec3 lightPos;
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 cameraPos;

float sdf_sphere(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float sdf_cube(vec3 p, vec3 b )
{
  return length(max(abs(p)-b,0.0));
}

void main() {
    vec3 rayDir = normalize(vec3(uv - 0.5, 1.0));
    vec3 origin = cameraPos;
    
    float distance = 0.0;
    vec3 point = origin;
    
    for (int i = 0; i < 100; i++) {
        distance = sdf_sphere(point, vec3(0.0, 0.0, 3.0), 0.4);
        if (distance < 0.01) {
            break;
        }
        point += rayDir * distance;
    }

    vec3 normal = normalize(point - vec3(0.0, 0.0, 3.0));
    
    vec3 lightDir = normalize(lightPos - point);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * objectColor;
    
    fragColor = vec4(diffuse, 1.0);
}
"""

vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)

shader_program = glCreateProgram()
glAttachShader(shader_program, vertex_shader)
glAttachShader(shader_program, fragment_shader)
glLinkProgram(shader_program)
glUseProgram(shader_program)

light_position = np.array([3.0, 3.0, 3.0], dtype=np.float32)
light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
object_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
camera_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, light_position)
glUniform3fv(glGetUniformLocation(shader_program, "lightColor"), 1, light_color)
glUniform3fv(glGetUniformLocation(shader_program, "objectColor"), 1, object_color)
glUniform3fv(glGetUniformLocation(shader_program, "cameraPos"), 1, camera_position)

vertices = np.array([
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    1.0,  1.0, 0.0,
    -1.0,  1.0, 0.0,
], dtype=np.float32)

indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    theta = pygame.time.get_ticks() * 0.001
    light_position[0] = 3 * np.cos(theta)
    light_position[1] = 3 * np.sin(theta)
    light_position[2] = 3 * np.sin(theta)

    glUniform3fv(glGetUniformLocation(shader_program, "lightPos"), 1, light_position)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)

    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    pygame.display.flip()
    pygame.time.wait(10)

pygame.quit()