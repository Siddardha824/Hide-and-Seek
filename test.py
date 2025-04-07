import pygame
import numpy as np
import random
from collections import defaultdict
import time

# --- Constants ---
GRID_WIDTH = 20
GRID_HEIGHT = 15
CELL_SIZE = 30
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Bot
GREEN = (0, 255, 0)  # Goal
BLUE = (0, 0, 255)  # Walls
GRAY = (200, 200, 200)  # Grid lines
DARK_BLUE = (0, 0, 139)  # Room Walls

# Environment Elements
EMPTY = 0
WALL = 1
GOAL = 2
BOT = 3  # Used for drawing, not grid state usually
ROOM_WALL = 4  # Differentiate room walls visually if needed

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 200

# Actions (Up, Down, Left, Right)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_COUNT = len(ACTIONS)


# --- Environment Class ---
class GridEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.bot_pos = (0, 0)
        self.goal_pos = (0, 0)
        self._place_elements()

    def _place_elements(self):
        # Reset grid
        self.grid.fill(EMPTY)

        # --- Place the Room ---
        room_w = random.randint(4, 7)
        room_h = random.randint(4, 7)
        # Ensure room fits
        room_x = random.randint(1, self.width - room_w - 1)
        room_y = random.randint(1, self.height - room_h - 1)

        # Create walls
        for y in range(room_y, room_y + room_h):
            self.grid[y, room_x] = ROOM_WALL  # Left wall
            self.grid[y, room_x + room_w - 1] = ROOM_WALL  # Right wall
        for x in range(room_x, room_x + room_w):
            self.grid[room_y, x] = ROOM_WALL  # Top wall
            self.grid[room_y + room_h - 1, x] = ROOM_WALL  # Bottom wall

        # --- Create a small discontinuity (gap) ---
        gap_side = random.choice(["top", "bottom", "left", "right"])
        if gap_side == "top" and room_w > 1:
            gap_x = random.randint(room_x + 1, room_x + room_w - 2)
            self.grid[room_y, gap_x] = EMPTY
        elif gap_side == "bottom" and room_w > 1:
            gap_x = random.randint(room_x + 1, room_x + room_w - 2)
            self.grid[room_y + room_h - 1, gap_x] = EMPTY
        elif gap_side == "left" and room_h > 1:
            gap_y = random.randint(room_y + 1, room_y + room_h - 2)
            self.grid[gap_y, room_x] = EMPTY
        elif gap_side == "right" and room_h > 1:
            gap_y = random.randint(room_y + 1, room_y + room_h - 2)
            self.grid[gap_y, room_x + room_w - 1] = EMPTY

        # --- Place Goal inside the room ---
        while True:
            gx = random.randint(room_x + 1, room_x + room_w - 2)
            gy = random.randint(room_y + 1, room_y + room_h - 2)
            if self.grid[gy, gx] == EMPTY:
                self.goal_pos = (gx, gy)
                self.grid[gy, gx] = GOAL
                break

        # --- Place Random Walls outside the room ---
        num_random_walls = int(0.1 * self.width * self.height)  # 10% walls
        for _ in range(num_random_walls):
            while True:
                wx = random.randint(0, self.width - 1)
                wy = random.randint(0, self.height - 1)
                # Avoid placing on goal, inside room structure, or start
                is_in_room_area = (room_x <= wx < room_x + room_w) and (
                    room_y <= wy < room_y + room_h
                )
                if (
                    self.grid[wy, wx] == EMPTY
                    and not is_in_room_area
                    and (wx, wy) != (0, 0)
                ):
                    self.grid[wy, wx] = WALL
                    break

        # --- Place Bot ---
        # Start outside the room, ensure it's not on a wall
        while True:
            bx = random.randint(0, self.width - 1)
            by = random.randint(0, self.height - 1)
            is_in_room_area = (room_x <= bx < room_x + room_w) and (
                room_y <= by < room_y + room_h
            )
            if self.grid[by, bx] == EMPTY and not is_in_room_area:
                self.bot_pos = (bx, by)
                break

    def reset(self):
        # Could re-randomize everything or just reset bot position
        # For now, just reset bot position to a valid random spot
        # self._place_elements() # Uncomment to fully randomize each episode

        # Find a valid starting position (not wall, not goal, not inside room)
        while True:
            bx = random.randint(0, self.width - 1)
            by = random.randint(0, self.height - 1)
            is_wall = self.grid[by, bx] == WALL or self.grid[by, bx] == ROOM_WALL
            is_goal = (bx, by) == self.goal_pos

            # Check if inside the physical room boundary for starting position
            room_coords = np.where(self.grid == ROOM_WALL)
            if len(room_coords[0]) > 0:
                min_ry, max_ry = np.min(room_coords[0]), np.max(room_coords[0])
                min_rx, max_rx = np.min(room_coords[1]), np.max(room_coords[1])
                is_inside_room = (min_rx <= bx <= max_rx) and (min_ry <= by <= max_ry)
            else:  # Should not happen if room is placed
                is_inside_room = False

            if not is_wall and not is_goal and not is_inside_room:
                self.bot_pos = (bx, by)
                break

        return self.get_state()

    def get_state(self):
        bx, by = self.bot_pos
        # State includes position AND immediate wall presence (gazing)
        wall_up = 1 if by == 0 or self.grid[by - 1, bx] in [WALL, ROOM_WALL] else 0
        wall_down = (
            1
            if by == self.height - 1 or self.grid[by + 1, bx] in [WALL, ROOM_WALL]
            else 0
        )
        wall_left = 1 if bx == 0 or self.grid[by, bx - 1] in [WALL, ROOM_WALL] else 0
        wall_right = (
            1
            if bx == self.width - 1 or self.grid[by, bx + 1] in [WALL, ROOM_WALL]
            else 0
        )

        # Return state as a tuple - Q-table keys must be hashable
        return (bx, by, wall_up, wall_down, wall_left, wall_right)

    def step(self, action_index):
        action = ACTIONS[action_index]
        bx, by = self.bot_pos
        next_bx, next_by = bx + action[0], by + action[1]

        reward = -0.1  # Small penalty for each step to encourage efficiency
        done = False

        # Check boundaries
        if 0 <= next_bx < self.width and 0 <= next_by < self.height:
            # Check for walls
            if self.grid[next_by, next_bx] not in [WALL, ROOM_WALL]:
                self.bot_pos = (next_bx, next_by)
                # Check if goal reached
                if self.bot_pos == self.goal_pos:
                    reward = 100  # Big reward for reaching the goal
                    done = True
            else:
                reward = -1  # Penalty for hitting a wall
        else:
            reward = -1  # Penalty for hitting boundary

        next_state = self.get_state()
        return next_state, reward, done

    def render(self, screen):
        screen.fill(WHITE)
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell_type = self.grid[y, x]

                if cell_type == WALL:
                    pygame.draw.rect(screen, BLUE, rect)
                elif cell_type == ROOM_WALL:
                    pygame.draw.rect(screen, DARK_BLUE, rect)
                elif cell_type == GOAL:
                    pygame.draw.rect(screen, GREEN, rect)

                # Draw grid lines
                pygame.draw.rect(screen, GRAY, rect, 1)

        # Draw Bot
        bot_rect = pygame.Rect(
            self.bot_pos[0] * CELL_SIZE,
            self.bot_pos[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(screen, RED, bot_rect)


# --- Q-Learning Agent Class ---
class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = defaultdict(
            lambda: np.zeros(len(actions))
        )  # state -> [q_val_action1, q_val_action2, ...]
        self.epsilon = EPSILON

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.actions)))  # Explore
        else:
            # Check if state exists, else choose random (or could return default 0 action)
            if state not in self.q_table or np.all(self.q_table[state] == 0):
                return random.choice(range(len(self.actions)))
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action_index, reward, next_state, done):
        current_q = self.q_table[state][action_index]

        # Get max Q value for the next state (handle terminal state)
        next_max_q = np.max(self.q_table[next_state]) if not done else 0

        # Q-learning formula
        new_q = current_q + ALPHA * (reward + GAMMA * next_max_q - current_q)
        self.q_table[state][action_index] = new_q

    def decay_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(
                MIN_EPSILON, self.epsilon
            )  # Ensure it doesn't go below min


# --- Main Training Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Q-Learning Bot in Room Environment")
    clock = pygame.time.Clock()

    env = GridEnvironment(GRID_WIDTH, GRID_HEIGHT)
    agent = QLearningAgent(ACTIONS)

    total_rewards = []
    print("Starting Training...")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # --- Run Episode ---
        while not done and step < MAX_STEPS_PER_EPISODE:
            # --- Pygame visualization and event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Training interrupted.")
                    return

            action_index = agent.choose_action(state)
            next_state, reward, done = env.step(action_index)
            agent.learn(state, action_index, reward, next_state, done)

            state = next_state
            episode_reward += reward
            step += 1

            # --- Render the environment ---
            env.render(screen)
            pygame.display.flip()
            clock.tick(15)  # Control simulation speed (FPS) - increase for faster sim

        # --- End of Episode ---
        agent.decay_epsilon()
        total_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(
                f"Episode {episode + 1}/{NUM_EPISODES} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}"
            )
            # Optional: Speed up rendering significantly after some learning
            # if episode > NUM_EPISODES // 2:
            #     clock.tick(60) # Render faster
            # elif episode > NUM_EPISODES // 4:
            #      clock.tick(30)

    print("Training finished.")
    # Keep window open briefly after training
    time.sleep(5)
    pygame.quit()

    # You could save the Q-table here if needed
    # import pickle
    # with open('q_table.pkl', 'wb') as f:
    #     pickle.dump(dict(agent.q_table), f) # Convert defaultdict for saving


if __name__ == "__main__":
    main()
