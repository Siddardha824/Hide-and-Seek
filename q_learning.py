import math
import random
import pygame
import os
from agent import Agent

class QLearningAgent(Agent):
    def __init__(self, x, y, cell_size, id=0, type='none', qtable_path=None):
        super().__init__(x, y, type, cell_size)
        self.initial_pos = (self.x, self.y)
        self.id = id
        self.qtable_path = qtable_path or f"qtable_agent_{id}.json"
        self.q_table = self.load_q_table()
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.9
        self.total_reward = 0
        self.prev_state = None
        self.prev_action = None
        self.type = type
        self.destroyed = False
        self.view_comments = False
        self.rank_point = 0

    def save_q_table(self):
        with open(self.qtable_path, "w") as f:
            for state, actions in self.q_table.items():
                state_str = f"{state[0]},{state[1]},{state[2]}"  # x,y,angle,vision_arc
                actions_str = ",".join(f"{a}:{q:.4f}" for a, q in actions.items())
                f.write(f"{state_str}|{actions_str}\n")
                #print(f"{state_str}|{actions_str}\n")

    def load_q_table(self):
        q_table = {}
        if not os.path.exists(self.qtable_path):
            return q_table
        with open(self.qtable_path, "r") as f:
            for line in f:
                if "|" not in line:
                    continue
                state_str, actions_str = line.strip().split("|")
                x, y, angle = map(float, state_str.split(","))
                action_pairs = actions_str.split(",")
                actions = {}
                for pair in action_pairs:
                    if ":" in pair:
                        a, q = pair.split(":")
                        actions[a] = float(q)
                q_table[(x, y, angle)] = actions
        return q_table
    
    def get_state(self):
        #print(self.vision_arc)
        return (round(self.x), round(self.y), round(self.angle))

    def get_action(self):
        state = self.get_state()
        actions = ['move', 'left', 'right', 'open', 'close']

        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in actions}

        if random.random() < self.epsilon:
            action = random.choice(actions)  # Exploration
        else:
            max_q = max(self.q_table[state].values())
            best_actions = [a for a, q in self.q_table[state].items() if q == max_q]
            action = random.choice(best_actions)  # Exploitation

        self.prev_state = state
        self.prev_action = action
        return action

    def update_q_value(self, reward, next_state):
        self.total_reward += reward
        old_q = self.q_table[self.prev_state][self.prev_action]
        next_max = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.prev_state][self.prev_action] = new_q

    def step(self, maze, screen, other_agents):
        action = self.get_action()
        reward = 0

        prev_pos = (self.x, self.y)

        if action == 'move':
            self.move_forward(maze, screen, other_agents)
        elif action == 'left':
            self.rotate_left()
        elif action == 'right':
            self.rotate_right()
        elif action == 'open':
            maze = self.open_door(maze)
        elif action == 'close':
            maze = self.close_door(maze)

        # === Reward Calculation ===
        distance_from_start = math.hypot(self.x - self.initial_pos[0], self.y - self.initial_pos[1])
        if distance_from_start > 500:
            reward += 25
            if self.view_comments == True:
                print(f"[{self.id}] Reward: explored area well.")
        elif distance_from_start < 500:
            reward -= 15
            if self.view_comments == True:
                print(f"[{self.id}] Penalty: not exploring much.")

        # for other in other_agents:
        #     dist_to_other = math.hypot(self.x - other.x, self.y - other.y)
        #     if dist_to_other < 50:
        #         reward -= 2
        #         print(f"[{self.id}] Penalty: too close to other agent.")
        #     else:
        #         reward += 1
        #         print(f"[{self.id}] Reward: keeping distance from other agent.")

        # Convert agent's pixel position to cell value (like your wall check style)
        my_region = None
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if rect.collidepoint(self.x, self.y):
                    my_region = maze[y][x]
                    if self.type == 'hider':
                        if my_region == '2':
                            self.rank_point += 3
                        if my_region == '1':
                            self.rank_point += 2
                        else:
                            self.rank_point += 1 #higher for who stays longest
                    break
            if my_region is not None:
                break

        for other in other_agents:
            other_region = None
            for y, row in enumerate(maze):
                for x, cell in enumerate(row):
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.collidepoint(other.x, other.y):
                        other_region = maze[y][x]
                        break
                if other_region is not None:
                    break

            if my_region == other_region and my_region not in ['w', 'o', 'h', 's']:
                dist_to_other = math.hypot(self.x - other.x, self.y - other.y)
                if dist_to_other < 200:
                    if self.type == 'hider' and other.type == 'seeker':
                        reward -= 10
                    elif self.type == 'seeker' and other.type == 'hider':
                        reward += 5
                    if self.view_comments == True:
                        print(f"[{self.id}] Too close to other agent in same room.")
                elif dist_to_other < 500:
                    if self.type == 'hider' and other.type == 'seeker':
                        reward += 10
                    elif self.type == 'seeker' and other.type == 'hider':
                        reward -= 5
                    if self.view_comments == True:
                        print(f"[{self.id}] Keeping distance from other agent in same room.")


        if self.type == 'hider':
            # === Penalty: Seeker detected in hider's vision ===
            look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
            look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance

            for other in other_agents:
                if other.type != 'seeker':
                    continue
                seeker_rect = pygame.Rect(
                    other.x - other.radius,
                    other.y - other.radius,
                    other.radius * 2,
                    other.radius * 2
                )
                if seeker_rect.clipline((self.x, self.y), (look_x, look_y)):
                    reward -= 100
                    if self.view_comments == True:
                        print(f"[{self.id}] HEAVY PENALTY: Seeker in hider’s vision! Must escape!")
                    
        elif self.type == 'seeker':
            # === Reward: Hider detected in seeker's vision ===
            look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
            look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance

            for other in other_agents:
                if other.type != 'hider':
                    continue
                hider_rect = pygame.Rect(
                    other.x - other.radius,
                    other.y - other.radius,
                    other.radius * 2,
                    other.radius * 2
                )
                if hider_rect.clipline((self.x, self.y), (look_x, look_y)):
                    reward += 1000
                    if self.view_comments == True:
                        print(f"[{self.id}] REWARD: Found a hider in vision! DESTROYED.")
                    other.destroyed = True  # Flag the hider as destroyed
                    

        # Check if it hit a wall
        look_x = self.x + math.cos(math.radians(self.angle)) * self.lookahead_distance
        look_y = self.y + math.sin(math.radians(self.angle)) * self.lookahead_distance
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                if cell == 'w':
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    if rect.clipline((self.x, self.y), (look_x, look_y)):
                        reward -= 10
                        if self.view_comments == True:
                            print(f"[{self.id}] Penalty: hit wall.")
                        break

        next_state = self.get_state()
        self.update_q_value(reward, next_state)
        return maze
