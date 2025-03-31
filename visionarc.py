import pygame
import sys
import math

# Global constants
SCREEN_HEIGHT = 480
SCREEN_WIDTH = SCREEN_HEIGHT  # Square window for 2D view
MAP_SIZE = 8
TILE_SIZE = int(SCREEN_WIDTH / MAP_SIZE)
FOV = math.pi / 3
HALF_FOV = FOV / 2
CASTED_RAYS = 160
STEP_ANGLE = FOV / CASTED_RAYS
MAX_DEPTH = int(MAP_SIZE * TILE_SIZE)

# Global variables
player_x = SCREEN_WIDTH / 2
player_y = SCREEN_HEIGHT / 2
player_angle = math.pi

# Map
MAP = (
    '########'
    '# #    #'
    '# #  ###'
    '#      #'
    '##     #'
    '#  ### #'
    '#   #  #'
    '########'
)

# Init pygame
pygame.init()
win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('2D Ray-casting')
clock = pygame.time.Clock()

def draw_map():
    # Draw the map
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            square = i * MAP_SIZE + j
            color = (191, 191, 191) if MAP[square] == '#' else (65, 65, 65)
            pygame.draw.rect(
                win, color,
                (j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
            )
    
    # Draw player as a circle with direction indicator
    pygame.draw.circle(win, (162, 0, 255), (int(player_x), int(player_y)), 8)
    pygame.draw.line(
        win, (255, 255, 0), 
        (player_x, player_y),
        (player_x - math.sin(player_angle) * 20, player_y + math.cos(player_angle) * 20),
        3
    )

def ray_casting():
    start_angle = player_angle - HALF_FOV
    
    for ray in range(CASTED_RAYS):
        for depth in range(MAX_DEPTH):
            target_x = player_x - math.sin(start_angle) * depth
            target_y = player_y + math.cos(start_angle) * depth

            # Convert to map coordinates
            col = int(target_x / TILE_SIZE)
            row = int(target_y / TILE_SIZE)  
            square = row * MAP_SIZE + col
            
            # Ray hits a wall
            if MAP[square] == '#':
                # Highlight the wall tile that was hit
                pygame.draw.rect(
                    win, (195, 137, 38), 
                    (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                )
                
                # Draw the casted ray
                pygame.draw.line(
                    win, (233, 166, 49), 
                    (player_x, player_y), (target_x, target_y)
                )
                break

        start_angle += STEP_ANGLE

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    # Convert player position to map coordinates for collision detection
    col = int(player_x / TILE_SIZE)
    row = int(player_y / TILE_SIZE)  
    square = row * MAP_SIZE + col

    # Player hits the wall (collision detection)
    if MAP[square] == '#': 
        # Move player back if they collide with a wall
        if forward:
            player_x -= -math.sin(player_angle) * 5
            player_y -= math.cos(player_angle) * 5
        else:
            player_x += -math.sin(player_angle) * 5
            player_y += math.cos(player_angle) * 5
    
    # Clear screen
    win.fill((0, 0, 0))
    
    # Draw the map and rays
    draw_map()
    ray_casting()

    # Handle input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]: player_angle -= 0.1
    if keys[pygame.K_RIGHT]: player_angle += 0.1
    if keys[pygame.K_UP]:
        forward = True
        player_x += -math.sin(player_angle) * 5
        player_y += math.cos(player_angle) * 5
    if keys[pygame.K_DOWN]:
        forward = False
        player_x -= -math.sin(player_angle) * 5
        player_y -= math.cos(player_angle) * 5

    # Display FPS
    fps = str(int(clock.get_fps()))
    font = pygame.font.SysFont('Arial', 30)
    win.blit(font.render(fps, False, (255, 255, 255)), (10, 10))

    pygame.display.flip()
    clock.tick(30)