import pygame
import sys
import math
import random
from agent import Agent

class RandomAgent(Agent):
    def __init__(self, x, y, cell_size):
        super().__init__(x, y, cell_size)
        self.move_timer = 0
        self.state = "move"  # "move" or "rotate"
        self.rotation_pending = 0  # Degrees left to rotate

    def update(self, maze, screen, player_agents):
        player_agent = player_agents[0]
        self.move_timer += 1
        if self.move_timer >= 5:
            self.move_timer = 0

            if self.state == "move":
                if self._can_move_forward(maze, [player_agent]):
                    self.move_forward(maze, screen, [player_agent])
                else:
                    self.state = "rotate"
                    self.rotation_pending = 10
                    self.rotation_dir = random.choice([-1, 1])  # Left or right
            elif self.state == "rotate":
                self.angle = (self.angle + self.rotation_dir) % 360
                self.rotation_pending -= 1
                if self.rotation_pending <= 0:
                    self.state = "move"

    def _can_move_forward(self, maze, other_agents=[]):
        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance

        # Check wall
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'w':
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        return False

        # Check agent collision
        dx = math.cos(math.radians(self.angle)) * self.move_step
        dy = math.sin(math.radians(self.angle)) * self.move_step
        for other in other_agents:
            if self.will_collide_with(other, dx, dy):
                return False

        return True
