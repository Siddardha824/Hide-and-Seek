import pygame
import random
import time
import numpy as np
from collections import defaultdict

# --- Constants ---
GRID_SIZE = 10
CELL_SIZE = 60
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
SCREEN_SIZE = (WIDTH, HEIGHT)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)  # Empty cell floor
GRAY = (100, 100, 100)  # Wall color
BLUE = (0, 0, 255)  # Agent color
RED = (255, 0, 0)  # Color for emphasizing enclosed area (optional)
ENCLOSED_FLOOR_COLOR = (200, 200, 250)  # Lighter blue for the target area floor

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.999  # Decay rate for exploration
EPSILON_MIN = 0.01  # Minimum exploration rate

# Rewards/Penalties
WALL_PENALTY = -50.0
STEP_COST = -0.1
ENCLOSEDNESS_FACTOR = 10.0  # Multiplier for enclosedness reward

# Simulation Parameters
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100
FPS = 5000  # Control visualization speed (higher means faster simulation)


# --- Environment Class ---
class GridEnvironment:
    def __init__(self, size):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]  # 0: empty, 1: wall
        self.walls = set()
        self.enclosed_area_coords = set()
        self._place_walls_and_enclosure()

    def _place_walls_and_enclosure(self, num_random_walls=1):
        # 1. Create an obvious enclosed area (e.g., bottom right corner)
        enclosure_start_row, enclosure_start_col = 6, 6
        enclosure_end_row, enclosure_end_col = 8, 8
        # Walls around the enclosure
        for r in range(enclosure_start_row - 1, enclosure_end_row + 2):
            for c in range(enclosure_start_col - 1, enclosure_end_col + 2):
                if (
                    r == enclosure_start_row - 1
                    or r == enclosure_end_row + 1
                    or c == enclosure_start_col - 1
                    or c == enclosure_end_col + 1
                ):
                    # Make sure it's within bounds
                    if 0 <= r < self.size and 0 <= c < self.size:
                        self.walls.add((r, c))

        # Remove one wall piece to create an entrance
        entrance_pos = (
            enclosure_start_row - 1,
            enclosure_start_col + 1,
        )  # Example entrance
        if entrance_pos in self.walls:
            self.walls.remove(entrance_pos)

        # Mark the floor cells inside the enclosure
        for r in range(enclosure_start_row, enclosure_end_row + 1):
            for c in range(enclosure_start_col, enclosure_end_col + 1):
                if (
                    0 <= r < self.size
                    and 0 <= c < self.size
                    and (r, c) not in self.walls
                ):
                    self.enclosed_area_coords.add((r, c))

        # 2. Add some random walls elsewhere
        added_walls = 0
        while added_walls < num_random_walls:
            r, c = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            # Avoid placing walls inside the main enclosure area or blocking the entrance too closely
            is_near_enclosure = False
            for er in range(enclosure_start_row - 1, enclosure_end_row + 2):
                for ec in range(enclosure_start_col - 1, enclosure_end_col + 2):
                    if r == er and c == ec:
                        is_near_enclosure = True
                        break
                if is_near_enclosure:
                    break

            if (r, c) not in self.walls and not is_near_enclosure:
                self.walls.add((r, c))
                added_walls += 1

    def is_wall(self, pos):
        return pos in self.walls

    def is_valid_pos(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def get_start_pos(self):
        while True:
            r, c = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (
                not self.is_wall((r, c)) and (r, c) not in self.enclosed_area_coords
            ):  # Start outside enclosure
                return (r, c)

    def draw(self, screen, cell_size):
        for r in range(self.size):
            for c in range(self.size):
                rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                if (r, c) in self.walls:
                    color = GRAY
                elif (r, c) in self.enclosed_area_coords:
                    color = ENCLOSED_FLOOR_COLOR  # Special floor for enclosed area
                else:
                    color = GREEN  # Regular floor
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)  # Grid lines


# --- Agent Class ---
class QLearningAgent:
    def __init__(self, environment, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        self.env = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)

        # Use defaultdict for Q-table for easy handling of new states
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        self.pos = self.env.get_start_pos()

    def get_state(self):
        # State is simply the agent's position tuple
        return self.pos

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.num_actions))  # Explore
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            q_values = self.q_table[state]
            # Handle ties randomly if multiple max values exist
            max_q = np.max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def calculate_enclosedness(self, pos):
        r, c = pos
        enclosed_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check N, S, W, E neighbors
            nr, nc = r + dr, c + dc
            neighbor_pos = (nr, nc)
            # Treat out-of-bounds as walls for enclosedness calculation
            if not self.env.is_valid_pos(neighbor_pos) or self.env.is_wall(
                neighbor_pos
            ):
                enclosed_count += 1
        return enclosed_count  # Returns a value from 0 to 4

    def update(self, state, action_index, reward, next_state):
        old_q_value = self.q_table[state][action_index]
        next_max_q = np.max(
            self.q_table[next_state]
        )  # Q-value of the best action from next state

        # Q-learning formula
        new_q_value = old_q_value + self.alpha * (
            reward + self.gamma * next_max_q - old_q_value
        )
        self.q_table[state][action_index] = new_q_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(
            self.epsilon_min, self.epsilon
        )  # Ensure it doesn't go below min

    def move(self, action_index):
        dr, dc = self.actions[action_index]
        next_pos = (self.pos[0] + dr, self.pos[1] + dc)

        # Check boundaries and walls
        if not self.env.is_valid_pos(next_pos) or self.env.is_wall(next_pos):
            # Hit wall or boundary
            reward = WALL_PENALTY
            next_state = self.pos  # Stay in the same state
        else:
            # Valid move
            enclosedness = self.calculate_enclosedness(next_pos)
            reward = (
                ENCLOSEDNESS_FACTOR * enclosedness
            ) + STEP_COST  # Reward based on new state's enclosedness + step cost
            self.pos = next_pos  # Update agent position
            next_state = self.pos

        return reward, next_state

    def draw(self, screen, cell_size):
        r, c = self.pos
        center_x = c * cell_size + cell_size // 2
        center_y = r * cell_size + cell_size // 2
        pygame.draw.circle(screen, BLUE, (center_x, center_y), cell_size // 3)


# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(
        f"Q-Learning Bot Finding Enclosed Space ({GRID_SIZE}x{GRID_SIZE})"
    )
    clock = pygame.time.Clock()

    environment = GridEnvironment(GRID_SIZE)
    agent = QLearningAgent(
        environment, ALPHA, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN
    )

    running = True
    episode = 0
    paused = False

    print(f"Starting training for {NUM_EPISODES} episodes...")
    print(f"Target Enclosed Area Floor Cells: {environment.enclosed_area_coords}")

    while running and episode < NUM_EPISODES:
        # Reset for new episode
        agent.pos = environment.get_start_pos()
        state = agent.get_state()
        total_episode_reward = 0
        step = 0

        while step < MAX_STEPS_PER_EPISODE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    global FPS  # Move global declaration to the top of the event block
                    if event.key == pygame.K_SPACE:  # Pause/Resume
                        paused = not paused
                    if event.key == pygame.K_s:  # Speed up visualization
                        FPS = min(5000, FPS + 500)
                        print(f"FPS set to {FPS}")
                    if event.key == pygame.K_a:  # Slow down visualization
                        FPS = max(30, FPS - 500)
                        print(f"FPS set to {FPS}")

            if not running:
                break
            if paused:
                # Keep drawing while paused
                screen.fill(WHITE)
                environment.draw(screen, CELL_SIZE)
                agent.draw(screen, CELL_SIZE)
                # Display pause text
                font = pygame.font.Font(None, 36)
                pause_text = font.render("PAUSED", True, RED)
                screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, 10))
                pygame.display.flip()
                clock.tick(10)  # Lower FPS during pause
                continue

            # --- Q-Learning Step ---
            action_index = agent.choose_action(state)
            reward, next_state = agent.move(action_index)
            agent.update(state, action_index, reward, next_state)

            state = next_state
            total_episode_reward += reward
            step += 1
            # --- End Q-Learning Step ---

            # --- Drawing ---
            screen.fill(WHITE)  # Clear screen
            environment.draw(screen, CELL_SIZE)
            agent.draw(screen, CELL_SIZE)

            # Display Info
            font = pygame.font.Font(None, 28)
            ep_text = font.render(f"Episode: {episode+1}/{NUM_EPISODES}", True, BLACK)
            step_text = font.render(
                f"Step: {step}/{MAX_STEPS_PER_EPISODE}", True, BLACK
            )
            eps_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, BLACK)
            screen.blit(ep_text, (5, 5))
            screen.blit(step_text, (5, 30))
            screen.blit(eps_text, (5, 55))

            pygame.display.flip()  # Update the display
            clock.tick(FPS)  # Control simulation speed

            # Check if episode should end (e.g., agent reached a specific goal, but here we just use max steps)

        # --- End of Episode ---
        agent.decay_epsilon()
        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{NUM_EPISODES} finished. Epsilon: {agent.epsilon:.3f}"
            )
            # Optional: Add evaluation runs here without exploration to see learned policy

        episode += 1

    print("Training finished.")

    # Keep window open after training to observe final state or learned path (optional)
    # You could add code here to run the agent greedily (epsilon=0)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        screen.fill(WHITE)
        environment.draw(screen, CELL_SIZE)
        agent.draw(screen, CELL_SIZE)
        font = pygame.font.Font(None, 36)
        final_text = font.render("Training Complete. Press Quit.", True, RED)
        screen.blit(final_text, (WIDTH // 2 - final_text.get_width() // 2, 10))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    main()
