import pygame
import sys
import math
import random
from collections import defaultdict

class Agent:
    def __init__(self, x, y, type, cell_size):
        self.grid_x = x
        self.grid_y = y
        self.initial_x = x
        self.initial_y = y
        self.total_distance = 0
        self.type = type

        self.x = x * cell_size + cell_size // 2
        self.y = y * cell_size + cell_size // 2
        self.angle = 0
        self.cell_size = cell_size
        self.length = cell_size // 2
        self.lookahead_distance = cell_size
        self.move_step = 2
        self.radius = self.cell_size // 4
        self.vision_arc = defaultdict(list)  # Format: [("wall", distance), ("empty", max_range), ("object", distance)]
        self.fov = math.pi / 3  # Field of view (60 degrees)
        self.half_fov = self.fov / 2
        self.casted_rays = 20  # Number of rays
        self.step_angle = self.fov / self.casted_rays
        self.max_depth = cell_size * 8  # Maximum vision range
        self.trail_arc = defaultdict(list)

    def will_collide_with(self, other_agent, dx, dy):
        next_x = self.x + dx
        next_y = self.y + dy
        dist = math.hypot(other_agent.x - next_x, other_agent.y - next_y)
        return dist < self.radius + other_agent.radius

    def draw(self, screen):
        # Triangle (agent)
        tip = (self.x + math.cos(math.radians(self.angle)) * self.length,
               self.y + math.sin(math.radians(self.angle)) * self.length)
        left = (self.x + math.cos(math.radians(self.angle + 120)) * self.length // 1.5,
                self.y + math.sin(math.radians(self.angle + 120)) * self.length // 1.5)
        right = (self.x + math.cos(math.radians(self.angle - 120)) * self.length // 1.5,
                 self.y + math.sin(math.radians(self.angle - 120)) * self.length // 1.5)
        
        if getattr(self, "type", "") == "seeker":
            color = (255, 0, 0)  # Red
        else:
            color = (0, 255, 0)  # Green (hider or default)
        pygame.draw.polygon(screen, color, [tip, left, right])
        pygame.draw.circle(screen, (255, 0, 0), (int(tip[0]), int(tip[1])), 4)

        # Longer lookahead line (vision)
        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance
        pygame.draw.line(screen, (0, 255, 255), (self.x, self.y), (look_x, look_y), 1)

        # Draw vision arc lines
        start_angle = math.radians(self.angle) - self.half_fov
        for ray in range(self.casted_rays):
            depth = self.max_depth
            if self.vision_arc[str(ray+1)]:
                depth = max(pair[1] for pair in self.vision_arc[str(ray+1)])
            else:
                depth = self.max_depth
            end_x = self.x + math.cos(start_angle) * depth
            end_y = self.y + math.sin(start_angle) * depth
            pygame.draw.line(screen, (0, 255, 0), (self.x, self.y), (end_x, end_y), 1)
            start_angle += self.step_angle

    def move_forward(self, maze, screen, other_agents=[]):
        dx = math.cos(math.radians(self.angle)) * self.move_step
        dy = math.sin(math.radians(self.angle)) * self.move_step

        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance

        # Check wall
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'w':
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        # print("Penalty: Hit wall")
                        return  # Penalty on wall hit
                elif cell == 'o':
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    # if rect.clipline((self.x, self.y), (look_x, look_y)):
                    #     print("!: Open door")
                        #dont return go through in open door
                elif cell == 'd':
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        #print("Penalty: Hit Closed door")
                        return  # Penalty on wall hit

        # Check collision with another agent
        for other in other_agents:
            if self.will_collide_with(other, dx, dy):
                #print("Penalty: Collided with another agent")
                return

        # Apply movement and track distance
        prev_x, prev_y = self.x, self.y
        col = int(prev_x / self.cell_size)
        row = int(prev_y / self.cell_size)
        if self.type == "seeker" and maze[row][col] != 'o':
            maze[row][col] = '9'
        elif self.type == "hider" and maze[row][col] != 'o':
            maze[row][col] = '4'
        self.x += dx
        self.y += dy
        self.total_distance += math.hypot(self.x - prev_x, self.y - prev_y)
        self.update_vision_and_trail_arc(maze)  # Update vision arc after movement

    def open_door(self,maze):
        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'd':
                    #print('found door')
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        #print("Open door")
                        #print('cell' , x, y, maze[y][x])
                        maze[y][x] = 'o'
                        #print('cell' ,maze[y][x])

                        return maze# Penalty on wall hit
        return maze
    
    def close_door(self,maze):
        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'o':
                    #print('found door')
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        #print("Close door")
                        #print('cell' , x, y, maze[y][x])
                        maze[y][x] = 'd'
                        #print('cell' ,maze[y][x])

                        return maze# Penalty on wall hit
        return maze
       
    def rotate_left(self):
        self.angle = (self.angle - 30) % 360

    def rotate_right(self):
        self.angle = (self.angle + 30) % 360

    def update_vision_and_trail_arc(self, maze):
        self.vision_arc = defaultdict(list)
        self.trail_arc = defaultdict(list)
        start_angle = math.radians(self.angle) - self.half_fov

        for ray in range(self.casted_rays):
            for depth in range(self.max_depth):
                target_x = self.x + math.cos(start_angle) * depth
                target_y = self.y + math.sin(start_angle) * depth

                col = int(target_x / self.cell_size)
                row = int(target_y / self.cell_size)

                if 0 <= row < len(maze) and 0 <= col < len(maze[0]):
                    cell = maze[row][col]
                    if cell == 'w':  # Wall
                        self.vision_arc[str(ray+1)].append(("wall", depth))
                        break
                    elif cell == 'd':  # Closed door
                        self.vision_arc[str(ray+1)].append(("closed door", depth))
                        break
                    elif cell == 's':  # Seeker
                        self.vision_arc[str(ray+1)].append(("seeker", depth))
                        break
                    elif cell == 'h':
                        self.vision_arc[str(ray+1)].append(("hider", depth))
                        break
                    elif cell == 'o':  # Open door
                        try:
                            if self.vision_arc[str(ray+1)][-1][0] != "open door":
                                self.vision_arc[str(ray+1)].append(("open door", depth))
                        except IndexError:
                            self.vision_arc[str(ray+1)].append(("open door", depth))
                    elif depth == self.max_depth - 1:
                        self.vision_arc[str(ray+1)].append(("empty", depth))
                        break
                    if cell >= '0' and cell <= '4' and self.type == 'seeker':
                        self.trail_arc[str(ray+1)].append((cell, depth))
                    if cell >= '5' and cell <= '9' and self.type == 'hider':
                        self.trail_arc[str(ray+1)].append((cell, depth))
            start_angle += self.step_angle
        
