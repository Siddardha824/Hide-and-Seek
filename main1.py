import pygame
import sys
import math
import random
import os
from maze import Maze
from agent import Agent
from test_agent import RandomAgent

seeker = []

hider = []

hider_rank = []

reward_log_path = "agent_rewards.txt"
with open(reward_log_path, "w") as f:
    f.write("agent_id,type,total_reward,rank_point_hider,hider_rank\n")

def get_maze_file():
    # Check if a file name is passed via command-line
    maze_file_name = sys.argv[1] if len(sys.argv) > 1 else "maze.txt"

    # Check if file exists
    if not os.path.exists(maze_file_name):
        print(f"File '{maze_file_name}' not found in the repo.")
        sys.exit(1)

    return maze_file_name

def game_loop():
    maze_file = get_maze_file()
    ROUND_DURATION_SEC = 120  # <<< You can change this value
    pygame.init()

    font = pygame.font.SysFont(None, 36)
    cell_size = 20

    while True:  # Infinite round loop
        # Load maze
        maze = maze_object.read_maze(maze_file)
        width, height = len(maze[0]) * cell_size, len(maze) * cell_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze with Agent")
        clock = pygame.time.Clock()

        # Create agents
        seeker, hider = maze_object.draw_agents(maze, cell_size)

        #testing
        seeker[0].view_comments = True


        start_ticks = pygame.time.get_ticks()
        running = True

        while running:
            screen.fill((0, 0, 0))
            maze_object.draw_maze(screen, maze, cell_size)

            # Draw all agents
            for agent in seeker + hider:
                if not agent.destroyed:
                    agent.draw(screen)

            # === Timer ===
            seconds_passed = (pygame.time.get_ticks() - start_ticks) // 1000
            seconds_left = max(0, ROUND_DURATION_SEC - seconds_passed)
            timer_text = font.render(f"Time Left: {seconds_left}s", True, (255, 255, 255))
            text_rect = timer_text.get_rect(center=(width // 2, 20))
            screen.blit(timer_text, text_rect)

            pygame.display.flip()

            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Step agents
            for agent in seeker:
                maze = agent.step(maze, screen, hider)
            for random_agent in hider:
                maze = random_agent.step(maze, screen, seeker)

            # Save q-tables
            for agent in seeker + hider:
                agent.save_q_table()

            clock.tick(60)

            if seconds_left <= 0 or all(h.destroyed for h in hider):
                print("⏰ Round ended.")
                # Save rewards
                with open(reward_log_path, "a") as f:
                    for a in seeker:
                        f.write(f"{a.id},{a.type},{a.total_reward}, - , - \n")
                    sorted_hiders = sorted(hider, key=lambda x: x.rank_point, reverse=True)

                    # Assign rank and write
                    for i, a in enumerate(sorted_hiders):
                        rank = i + 1  # since highest points gets rank 1
                        f.write(f"{a.id},{a.type},{a.total_reward},{a.rank_point},{rank}\n")
                    f.write('--------------------------------------------------------------------\n')
                break  # End this round and restart loop

    

if __name__ == '__main__':
    maze_object = Maze()
    game_loop()