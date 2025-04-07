import pygame
import sys
import math
import random
from q_learning import QLearningAgent

#def read_maze(filename):
#     try:
#         with open(filename, 'r') as f:
#             return [list(line.strip()) for line in f.readlines()]
#     except FileNotFoundError:
#         return [list("w" * 20)] + [list("w" + " " * 18 + "w") for _ in range(18)] + [list("w" * 20)]
class Maze:
    def read_maze(self,filename):
        try:
            door_positions = []
            with open(filename, 'r') as f:
                maze = [list(line.strip()) for line in f.readlines()]
                maze = [['o' if cell == 'd' else cell for cell in row] for row in maze]
                door_positions = [[(row,cell) if cell == 'o' else cell for cell in row] for row in maze]

                # Ensure all rows are exactly 40 characters wide
                max_width = max(len(row) for row in maze)  # Find max row width
                for i in range(len(maze)):
                    if len(maze[i]) < max_width:
                        maze[i] += ['w'] * (max_width - len(maze[i]))  # Pad with walls

                if not maze or len(maze[0]) == 0:
                    raise ValueError("Maze file is empty or incorrectly formatted.")
                
                return maze, door_positions
        except FileNotFoundError:
            print("Maze file not found! Using a default 40x40 maze.")
            return [["w"] * 40] + [["w"] + ["1"] * 38 + ["w"] for _ in range(38)] + [["w"] * 40], []
        except ValueError as e:
            print(f"Error reading maze: {e}")
            return [["w"] * 40] + [["w"] + ["1"] * 38 + ["w"] for _ in range(38)] + [["w"] * 40], [] # Default fallback


    def draw_maze(self,screen, maze, cell_size):
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'w':
                    pygame.draw.rect(screen, (100, 100, 100), (x * cell_size, y * cell_size, cell_size, cell_size))
                elif cell == 'd':
                    pygame.draw.rect(screen, (255, 255, 0), (x * cell_size, y * cell_size, cell_size, cell_size))

    def get_free_position(self,maze):
        free_positions = [(x, y) for y in range(len(maze)) for x in range(len(maze[0])) if maze[y][x] != 'w']
        return random.choice(free_positions)

    def draw_agents(self,maze, cell_size):
        seeker = []
        hider = []
        unique_id = 0

        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 's':
                    unique_id += 1
                    agent = QLearningAgent(
                        x, y, cell_size,
                        id=unique_id,
                        type='seeker',
                        qtable_path=f"qtable_agent_seeker_{unique_id}.txt"
                    )
                    seeker.append(agent)

                elif cell == 'h':
                    unique_id += 1
                    agent = QLearningAgent(
                        x, y, cell_size,
                        id=unique_id,
                        type='hider',
                        qtable_path=f"qtable_agent_hider_{unique_id}.txt"
                    )
                    hider.append(agent)

        return seeker, hider