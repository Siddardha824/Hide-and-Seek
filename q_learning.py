# q_learning.py

import math
import random
import pygame
import os
from agent import Agent  # Make sure agent.py is accessible
from collections import defaultdict  # Ensure defaultdict is imported


class QLearningAgent(Agent):
    def __init__(self, x, y, cell_size, id=0, type="none", qtable_path=None):
        # Initialize base Agent class
        super().__init__(x, y, type, cell_size)

        # QLearning specific attributes
        self.initial_pos = (self.x, self.y)
        self.id = id
        # Use .txt extension based on previous implementation
        self.qtable_path = qtable_path or f"qtable_agent_{self.id}.txt"
        self.q_table = self.load_q_table()
        self.epsilon = 0.2  # Exploration rate
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.total_reward = 0  # Accumulated reward over an episode for logging
        self.prev_state = None
        self.prev_action = None
        # self.type is inherited from Agent init
        self.destroyed = False
        self.view_comments = False  # Set to True for debug prints
        self.rank_point = 0  # For hider ranking

        # Constants for vision-based rewards/actions
        self.VISION_PROXIMITY_MAX_BONUS = (
            200  # Max reward/penalty magnitude for seeing opponent
        )
        self.VISION_PROXIMITY_DECAY_RATE = (
            self.cell_size * 3.5
        )  # Controls how fast effect drops with distance
        self.VISION_CATCH_THRESHOLD = (
            self.cell_size * 1.5
        )  # Distance within which a seeker catches (based on vision)
        self.CATCH_BONUS = 1000  # Reward for seeker catching hider
        self.SEEN_PENALTY_MULTIPLIER = (
            1.5  # How much stronger penalty is for hider being seen vs seeker reward
        )
        self.WALL_PENALTY = 10  # Penalty for trying to move into wall/door

    def save_q_table(self):
        """Saves the Q-table to a file."""
        qtable_path = self.qtable_path  # Use path defined in init
        try:
            with open(qtable_path, "w") as f:
                for state, actions in self.q_table.items():
                    # Ensure state components are strings for joining
                    state_str = ",".join(map(str, state))
                    actions_str = ",".join(f"{a}:{q:.4f}" for a, q in actions.items())
                    f.write(f"{state_str}|{actions_str}\n")
        except Exception as e:
            print(f"Error saving Q-table for agent {self.id} to {qtable_path}: {e}")

    def load_q_table(self):
        """Loads the Q-table from a file."""
        q_table = {}
        qtable_path = self.qtable_path  # Use path defined in init
        if not os.path.exists(qtable_path):
            print(
                f"Q-table file not found for agent {self.id}: {qtable_path}. Starting fresh."
            )
            return q_table
        try:
            with open(qtable_path, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if "|" not in line:
                        continue
                    try:
                        state_str, actions_str = line.split("|")
                        # Convert state components back to floats/numbers
                        state_components = list(map(float, state_str.split(",")))
                        state = tuple(state_components)

                        action_pairs = actions_str.split(",")
                        actions = {}
                        for pair in action_pairs:
                            if ":" in pair:
                                a, q_str = pair.split(":")
                                try:
                                    actions[a] = float(q_str)
                                except ValueError:
                                    print(
                                        f"Warning: L{line_num+1} Could not parse Q-value '{q_str}' for state {state}, action {a} in {qtable_path}. Setting to 0.0"
                                    )
                                    actions[a] = 0.0
                        q_table[state] = actions
                    except Exception as parse_e:
                        print(
                            f"Error parsing line {line_num+1} in {qtable_path}: '{line}'. Error: {parse_e}"
                        )
        except Exception as e:
            print(f"Error loading Q-table for agent {self.id} from {qtable_path}: {e}")
        print(
            f"Loaded Q-table for agent {self.id} from {qtable_path} with {len(q_table)} states."
        )
        return q_table

    def get_state(self):
        """Gets the current state representation (rounded position and angle)."""
        # Consider rounding to larger increments (e.g., // 5 * 5) or using grid coords
        # to reduce state space size if performance is an issue.
        return (round(self.x), round(self.y), round(self.angle))

    def get_action(self, maze=None):
        """Chooses an action using epsilon-greedy strategy."""
        state = self.get_state()
        actions = ["move", "left", "right", "open", "close"]

        # Initialize Q-values for new state if not seen before
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in actions}
            if self.view_comments:
                print(f"[{self.id}] New state encountered: {state}")

        # Get current region (door or not)
        current_region = self.get_current_region(maze) if maze is not None else None

        # Remove door actions if not at a door location
        if current_region not in ["d", "o"]:  # 'd' for door, 'o' for open door
            actions = [a for a in actions if a not in ["open", "close"]]

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            action = random.choice(actions)  # Explore
            if self.view_comments:
                print(f"[{self.id}] Exploring: chose {action}")
        else:
            # Exploit: Choose best known action from remaining options
            q_values = {a: q for a, q in self.q_table[state].items() if a in actions}
            if not q_values:  # Should not happen if initialized, but safety check
                action = random.choice(actions)
            else:
                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if abs(q - max_q) < 1e-6]
                action = random.choice(best_actions)  # Random selection among best

            if self.view_comments:
                q_str = ", ".join(f"{a}:{q:.1f}" for a, q in q_values.items())
                print(
                    f"[{self.id}] Exploiting: Qs={{{q_str}}}, chose {action} (maxQ {max_q:.1f})"
                )

        # Store state and action for Q-update later
        self.prev_state = state
        self.prev_action = action
        return action

    def update_q_value(self, reward, next_state):
        """Updates the Q-value for the previous state-action pair."""
        # Ensure we have a previous state/action to update
        if self.prev_state is None or self.prev_action is None:
            if self.view_comments:
                print(f"[{self.id}] Skipping Q-update: No previous state/action.")
            return

        # Ensure Q-table entries exist for calculation
        if self.prev_state not in self.q_table:
            self.q_table[self.prev_state] = {
                a: 0.0 for a in ["move", "left", "right", "open", "close"]
            }
        if self.prev_action not in self.q_table[self.prev_state]:
            self.q_table[self.prev_state][self.prev_action] = 0.0
        if next_state not in self.q_table:
            self.q_table[next_state] = {
                a: 0.0 for a in ["move", "left", "right", "open", "close"]
            }

        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_q(s') - Q(s,a))
        old_q = self.q_table[self.prev_state][self.prev_action]

        # Find max Q-value for the next state
        next_q_values = self.q_table[next_state]
        next_max_q = 0.0
        if next_q_values:  # Check if dictionary is not empty
            next_max_q = max(next_q_values.values())

        # Calculate new Q-value
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[self.prev_state][self.prev_action] = new_q

        # Accumulate reward for episode logging
        self.total_reward += reward

        if self.view_comments:
            print(
                f"[{self.id}] QUpdate: State={self.prev_state}, Action={self.prev_action}, Reward={reward:.1f}, NextState={next_state}, OldQ={old_q:.2f}, NewQ={new_q:.2f}"
            )

    # --- Main Step Function ---
    def step(self, maze, screen, other_agents, door_positions):
        """Performs one step of action, reward calculation, and learning."""
        # 1. Choose action based on current state
        action = self.get_action(maze)  # Pass maze here
        current_reward = 0.0

        # 2. Perform Action in environment
        if action == "move":
            self.move_forward(maze, screen, other_agents)
        elif action == "left":
            self.rotate_left()
        elif action == "right":
            self.rotate_right()
        elif action == "open":
            maze = self.open_door(maze)  # Returns updated maze
        elif action == "close":
            maze = self.close_door(maze)  # Returns updated maze

        # 3. Update Vision Arc based on NEW state
        # Assumes Agent.py has the updated method that detects agents
        try:
            active_others = [
                a
                for a in other_agents
                if not getattr(a, "destroyed", False) and a is not self
            ]
            self.update_vision_arc(maze, active_others)
        except (AttributeError, TypeError) as e:
            print(
                f"CRITICAL WARNING: Agent {self.id} 'update_vision_arc' failed or missing/wrong arguments! Vision rewards will not work. Error: {e}"
            )
            # Ensure vision_arc exists even if update fails, to prevent later errors
            if not hasattr(self, "vision_arc"):
                self.vision_arc = defaultdict(list)

        # 4. Calculate Reward based on outcome of action and new state/vision

        # --- Exploration Reward ---
        distance_from_start = math.hypot(
            self.x - self.initial_pos[0], self.y - self.initial_pos[1]
        )
        explore_reward = 0
        if distance_from_start > 2000:
            explore_reward = 60
        elif distance_from_start > 1000:
            explore_reward = 50
        elif distance_from_start > 500:
            explore_reward = 25
        elif distance_from_start < self.cell_size * 10:
            explore_reward = -5  # Penalize staying too close?
        # You had a large penalty (-150) for dist < 500, keeping it similar:
        if distance_from_start < 500:
            explore_reward = -150
        current_reward += explore_reward
        # Add view_comments logic if needed

        # --- Region/Door Interaction Rewards ---
        my_region = self.get_current_region(maze)
        region_reward = 0
        if self.type == "hider":
            base_region_reward = 300
            if my_region == "b":
                region_reward += base_region_reward + 300
                if action == "close":
                    region_reward += 500
                elif action == "open":
                    region_reward -= 500
                self.rank_point += 3
            elif my_region == "c":
                region_reward += base_region_reward
                if action == "close":
                    region_reward += 300
                elif action == "open":
                    region_reward -= 300
                self.rank_point += 2
            elif my_region == "a":  # Add reward for region 1
                if action == "open":
                    region_reward += 200  # Reward for opening doors in region 1
                self.rank_point += 1
            else:
                self.rank_point += 1
        elif self.type == "seeker":
            if (
                action == "open" and my_region != "w"
            ):  # Reward opening doors (except walls)
                region_reward += 500
        current_reward += region_reward
        # Add view_comments logic if needed

        # --- Same Room Interaction Rewards ---
        interaction_reward = 0
        # Note: This iterates through ALL agents, use active_others?
        for other in active_others:  # Use active agents only
            other_region = self.get_agent_region(maze, other)  # Need another helper
            if (
                my_region is not None
                and my_region == other_region
                and my_region not in ["w", "o", "d", "h", "s"]
            ):  # Check grid cells, not start/door cells
                dist_to_other = math.hypot(self.x - other.x, self.y - other.y)
                if self.type == "hider" and other.type == "hider":
                    # Reward hiders sticking together in safe rooms?
                    if my_region in ["b", "c"]:
                        interaction_reward += 500  # Make sure this aligns with goals
                elif self.type == "hider" and other.type == "seeker":
                    if dist_to_other < 200:
                        interaction_reward -= 100  # Penalty increases closer
                    elif dist_to_other < 500:
                        interaction_reward -= 50  # Smaller penalty further away
                elif self.type == "seeker" and other.type == "hider":
                    if dist_to_other < 200:
                        interaction_reward += 200  # Reward increases closer
                    elif dist_to_other < 500:
                        interaction_reward += 100  # Smaller reward further away
        current_reward += interaction_reward
        # Add view_comments logic if needed

        # --- Vision Arc Based Opponent Detection Reward/Penalty/Catch ---
        opponent_type = "hider" if self.type == "seeker" else "seeker"
        opponent_detected_in_fov = False
        min_opponent_dist_in_fov = float("inf")
        vision_reward = 0

        for ray_idx, items in self.vision_arc.items():
            for item_tuple in items:
                if len(item_tuple) >= 3 and item_tuple[0] == "agent":
                    item_depth = item_tuple[1]
                    detected_agent_type = item_tuple[2]
                    if detected_agent_type == opponent_type:
                        opponent_detected_in_fov = True
                        min_opponent_dist_in_fov = min(
                            min_opponent_dist_in_fov, item_depth
                        )

        if opponent_detected_in_fov:
            proximity_effect = self.VISION_PROXIMITY_MAX_BONUS * math.exp(
                -min_opponent_dist_in_fov / self.VISION_PROXIMITY_DECAY_RATE
            )

            if self.type == "seeker":
                vision_reward += proximity_effect
                if self.view_comments:
                    print(
                        f"[{self.id} Saw Hider]: VisRew= +{proximity_effect:.1f} (dist {min_opponent_dist_in_fov:.1f})"
                    )

                # Catching Logic
                if min_opponent_dist_in_fov < self.VISION_CATCH_THRESHOLD:
                    closest_hider_obj, min_actual_dist_sq = self.find_closest_opponent(
                        active_others, "hider"
                    )
                    if (
                        closest_hider_obj
                        and min_actual_dist_sq
                        < (self.VISION_CATCH_THRESHOLD * 1.1) ** 2
                    ):
                        if not closest_hider_obj.destroyed:
                            closest_hider_obj.destroyed = True
                            vision_reward += self.CATCH_BONUS  # Add catch bonus
                            if self.view_comments:
                                print(
                                    f"💥 [{self.id}] Caught hider {closest_hider_obj.id}! (Actual dist {math.sqrt(min_actual_dist_sq):.1f}) Bonus= +{self.CATCH_BONUS}"
                                )

            elif self.type == "hider":
                penalty = proximity_effect * self.SEEN_PENALTY_MULTIPLIER
                vision_reward -= penalty
                if self.view_comments:
                    print(
                        f"[{self.id} Saw Seeker]: VisPen= -{penalty:.1f} (dist {min_opponent_dist_in_fov:.1f})"
                    )

        current_reward += vision_reward

        # --- Vision Arc Based Wall Collision Penalty ---
        wall_penalty = 0
        hit_wall_close = False
        min_wall_dist = float("inf")
        for ray_idx, items in self.vision_arc.items():
            for item_tuple in items:
                # Check structure before unpacking
                if len(item_tuple) >= 2:
                    item_type = item_tuple[0]
                    item_depth = item_tuple[1]
                    if item_type in ["wall", "closed_door"]:
                        min_wall_dist = min(min_wall_dist, item_depth)
                        # Check if obstacle is very close
                        if item_depth < self.move_step * 1.5:
                            hit_wall_close = True
                            # No need to check further items on this ray if close wall found
                            break
            # No need to check further rays if close wall found on any ray
            if hit_wall_close:
                break

        # Apply penalty only if the action was 'move' and hit obstacle
        if hit_wall_close and action == "move":
            wall_penalty = -self.WALL_PENALTY  # Use the defined constant
            if self.view_comments:
                print(
                    f"[{self.id}] Penalty: Tried moving into close obstacle (dist {min_wall_dist:.1f}). Pen= {wall_penalty}"
                )
        current_reward += wall_penalty

        # 5. Learn from Experience (Update Q-value)
        next_state = self.get_state()
        self.update_q_value(current_reward, next_state)

        # 6. Return updated maze state
        return maze

    # --- Helper Functions ---
    def get_current_region(self, maze):
        """Determines the maze grid cell type at the agent's current location."""
        grid_x = int(self.x / self.cell_size)
        grid_y = int(self.y / self.cell_size)
        if 0 <= grid_y < len(maze) and 0 <= grid_x < len(maze[0]):
            return maze[grid_y][grid_x]
        return None  # Indicate agent is outside maze bounds

    def get_agent_region(self, maze, agent):
        """Determines the maze grid cell type for a given agent object."""
        grid_x = int(agent.x / self.cell_size)
        grid_y = int(agent.y / self.cell_size)
        if 0 <= grid_y < len(maze) and 0 <= grid_x < len(maze[0]):
            return maze[grid_y][grid_x]
        return None

    def find_closest_opponent(self, opponent_list, opponent_type_target):
        """Finds the closest non-destroyed opponent of a specific type."""
        closest_opponent_obj = None
        min_dist_sq = float("inf")
        for other in opponent_list:
            if other.type == opponent_type_target and not getattr(
                other, "destroyed", False
            ):
                dist_sq = (self.x - other.x) ** 2 + (self.y - other.y) ** 2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_opponent_obj = other
        return closest_opponent_obj, min_dist_sq
