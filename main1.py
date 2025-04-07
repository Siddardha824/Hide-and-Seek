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
    maze_file_name = sys.argv[1] if len(sys.argv) > 1 else "maze3.txt"

    # Check if file exists
    if not os.path.exists(maze_file_name):
        print(f"File '{maze_file_name}' not found in the repo.")
        sys.exit(1)

    return maze_file_name


def draw_distance_visualizer(screen, agent1, agent2, max_distance=500):
    # Calculate distance between agents
    distance = math.hypot(agent1.x - agent2.x, agent1.y - agent2.y)

    # Much stronger exponential decay using e^(-x/20)
    decay_factor = 20  # Smaller value = faster decay
    alpha = min(255, int(255 * math.exp(-distance / decay_factor)))

    # Create a surface for the line with alpha channel
    line_surface = pygame.Surface(
        (screen.get_width(), screen.get_height()), pygame.SRCALPHA
    )

    # Draw the line with calculated alpha
    if agent1.type == "seeker" and agent2.type == "seeker":
        color = (255, 0, 0, alpha)  # Red for seeker-seeker
    elif agent1.type == "hider" and agent2.type == "hider":
        color = (0, 255, 0, alpha)  # Green for hider-hider
    else:
        color = (255, 255, 0, alpha)  # Yellow for seeker-hider

    pygame.draw.line(line_surface, color, (agent1.x, agent1.y), (agent2.x, agent2.y), 2)
    screen.blit(line_surface, (0, 0))


# Add after imports
import pygame.display


def create_distance_window():
    distance_window = pygame.display.set_mode((300, 400), pygame.RESIZABLE)
    pygame.display.set_caption("Distance Monitor")
    return distance_window


def game_loop():
    maze_file = get_maze_file()
    ROUND_DURATION_SEC = 60
    pygame.init()

    font = pygame.font.SysFont(None, 36)
    cell_size = 20

    # Create single window with space for both game and visualizer
    maze,door_positions = maze_object.read_maze(maze_file)
    width, height = len(maze[0]) * cell_size, len(maze) * cell_size

    # Combined window (maze width + 300px for visualizer)
    combined_window = pygame.display.set_mode((width + 300, height))
    pygame.display.set_caption("Hide and Seek with Distance Monitor")

    while True:  # Infinite round loop
        # Create agents
        seeker, hider = maze_object.draw_agents(maze, cell_size)
        clock = pygame.time.Clock()

        # testing
        hider[1].view_comments = True

        start_ticks = pygame.time.get_ticks()
        running = True

        while running:
            # Handle events first for better responsiveness
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Calculate time
            seconds_passed = (pygame.time.get_ticks() - start_ticks) // 1000
            seconds_left = max(0, ROUND_DURATION_SEC - seconds_passed)

            # Clear and draw everything
            combined_window.fill((0, 0, 0))
            maze_object.draw_maze(combined_window, maze, cell_size)

            # Draw agents
            for agent in seeker + hider:
                if not agent.destroyed:
                    agent.draw(combined_window)

            # Draw visualizer background
            pygame.draw.rect(combined_window, (50, 50, 50), (width, 0, 300, height))

            # Draw distance bars
            all_agents = [a for a in seeker + hider if not a.destroyed]
            bar_height = 30
            spacing = 10
            y_pos = 20

            for i in range(len(all_agents)):
                for j in range(i + 1, len(all_agents)):
                    agent1, agent2 = all_agents[i], all_agents[j]
                    distance = math.hypot(agent1.x - agent2.x, agent1.y - agent2.y)

                    # Calculate bar length with exponential decay
                    max_bar_width = 280
                    decay_factor = 20  # Faster decay
                    bar_width = max_bar_width * math.exp(-distance / decay_factor)

                    # Draw bar background
                    pygame.draw.rect(
                        combined_window,
                        (100, 100, 100),
                        (width + 10, y_pos, max_bar_width, bar_height),
                    )

                    # Draw actual bar
                    if agent1.type == agent2.type == "seeker":
                        color = (255, 0, 0)  # Red
                    elif agent1.type == agent2.type == "hider":
                        color = (0, 255, 0)  # Green
                    else:
                        color = (255, 255, 0)  # Yellow

                    pygame.draw.rect(
                        combined_window,
                        color,
                        (width + 10, y_pos, int(bar_width), bar_height),
                    )

                    # Draw label
                    label = f"{agent1.type[:1]}{agent1.id}-{agent2.type[:1]}{agent2.id}"
                    text = pygame.font.SysFont(None, 24).render(
                        label, True, (255, 255, 255)
                    )
                    combined_window.blit(text, (width + 15, y_pos + 8))

                    y_pos += bar_height + spacing

            # Draw connecting lines between agents
            for i in range(len(all_agents)):
                for j in range(i + 1, len(all_agents)):
                    draw_distance_visualizer(
                        combined_window, all_agents[i], all_agents[j]
                    )

            # Draw timer and update display
            timer_text = font.render(
                f"Time Left: {seconds_left}s", True, (255, 255, 255)
            )
            text_rect = timer_text.get_rect(center=(width // 2, 20))
            combined_window.blit(timer_text, text_rect)

            # Single display update
            pygame.display.flip()

            # Step agents
            for agent in seeker:
                maze = agent.step(maze, combined_window, hider, door_positions)
            for random_agent in hider:
                maze = random_agent.step(maze, combined_window, seeker, door_positions)

            # Control frame rate
            clock.tick(60)

            if seconds_left <= 0 or all(h.destroyed for h in hider):
                print("⏰ Round ended.")
                # Save rewards
                with open(reward_log_path, "a") as f:
                    for a in seeker:
                        f.write(f"{a.id},{a.type},{a.total_reward}, - , - \n")
                    sorted_hiders = sorted(
                        hider, key=lambda x: x.rank_point, reverse=True
                    )

                    # Assign rank and write
                    for i, a in enumerate(sorted_hiders):
                        rank = i + 1  # since highest points gets rank 1
                        f.write(
                            f"{a.id},{a.type},{a.total_reward},{a.rank_point},{rank}\n"
                        )
                    f.write(
                        "--------------------------------------------------------------------\n"
                    )
                break  # End this round and restart loop


if __name__ == "__main__":
    maze_object = Maze()
    game_loop()
