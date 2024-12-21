import numpy as np
import pygame 
def sdf_sphere(p, center, radius):
    return np.linalg.norm(p - center) - radius

def vec3(x, y, z):
    return np.array([x, y, z])

def lambert_illumination(normal, light_dir, object_color, light_color):
    intensity = max(np.dot(normal, light_dir), 0.0)
    return (object_color * light_color * intensity / 255).astype(int)

def ray_march(origin, direction):
    distance_traveled = 0.0
    for _ in range(MAX_STEPS):
        point = origin + direction * distance_traveled
        distance_to_surface = sdf_sphere(point, vec3(0, 0, 3), 1.0) 
        if distance_to_surface < EPSILON:
            return point 
        if distance_traveled > MAX_DIST:
            break
        distance_traveled += distance_to_surface
    return None

def render():
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i = (x - WIDTH / 2) / WIDTH * np.tan(FOV / 2) * 2
            j = (y - HEIGHT / 2) / HEIGHT * np.tan(FOV / 2) * 2
            direction = np.array([i, j, 1])
            direction = direction / np.linalg.norm(direction)
            hit = ray_march(np.array([0, 0, 0]), direction)
            if hit is not None:
                normal = (hit - vec3(0, 0, 3)) / np.linalg.norm(hit - vec3(0, 0, 3))

                shaded_color = lambert_illumination(normal, light_dir, object_color, light_color)

                pygame.draw.rect(screen, shaded_color, (x, y, 1, 1))


WIDTH, HEIGHT = 300, 300
FOV = np.pi / 3

MAX_STEPS = 10
MAX_DIST = 3
EPSILON = 0.01

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ray Marching with PyGame")
clock = pygame.time.Clock()

light_color = np.array([255, 255, 255])
object_color = np.array([255, 0, 0])
r = 3
theta = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    x = r * np.cos(theta)
    y = r * np.cos(theta)
    z = 3 + r * np.sin(theta)
    light_position = vec3(x, y, z)
    theta += 0.1

    light_dir = light_position - vec3(0, 0, 3)
    light_dir = light_dir / np.linalg.norm(light_dir)

    screen.fill((0, 0, 0))
    render()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()