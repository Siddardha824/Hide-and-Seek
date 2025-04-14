# Hide and Seek Reinforcement Learning Simulation

A Python implementation of a hide-and-seek game using Q-learning reinforcement learning, built with Pygame.

## 📁 Project Structure

Hide-and-Seek-main/
├── main1.py # Main game loop and visualization
├── q_learning.py # Q-learning agent implementation
├── agent.py # Base agent class with movement/vision
├── maze.py # Maze loading and agent placement
├── maze3.txt # Sample maze configuration
└── agent_rewards.txt # Log file for agent performance
plaintext

## ✨ Key Features

- **Q-learning Agents**: Both seekers and hiders learn through reinforcement.
- **Vision System**: Agents have a limited field-of-view using ray casting.
- **Maze Navigation**: Supports walls, open/closed doors, and reward-based regions.
- **Reward System**: Rich reward structure guiding agent behavior dynamically.
- **Visualization**: Real-time display of agent movements and interactions.

## 🖥️ Requirements

- Python 3.6+
- [Pygame](https://www.pygame.org/) (`pip install pygame`)

## 🛠️ Setup, Installation, and Running

Follow these steps to set up the environment and run the simulation:

### Setup and Installation

1.  **Extract the Project:**
    Extract the provided zip file to your desired location.

    ```bash
    unzip hide-and-seek.zip
    cd hide-and-seek
    ```

2.  **Run the Setup Script:**
    Execute the setup script to prepare the environment and install dependencies.

    ```bash
    ./setup.sh
    ```

3.  **Activate the Virtual Environment:**
    Before running the simulation, activate the Python environment.
    ```bash
    source venv/bin/activate
    ```
    _(Note: For Windows users, use `.\venv\Scripts\activate`)_

### How to Run

1.  **Run the Simulation:**
    With the virtual environment activated, start the simulation:
    ```bash
    python main1.py
    ```

---

_Make sure you are in the project's root directory when running these commands._

## 🧱 Maze File Format

- w : Wall
- a , b , c : Different regions (affects rewards)
- d : Closed door (can be opened)
- o : Open door
- s : Seeker starting position
- h : Hider starting position
  Example maze line: waaaaaaawdwaaaaaaaw

## Agent Types

### 🔴 Seeker (Red)

- Learns to find and catch hiders.
- **Rewards:**
  - Seeing hiders (with proximity bonus)
  - Catching hiders: `+1000`
  - Opening doors: `+500`

### 🟢 Hider (Green)

- Learns to avoid seekers and hide effectively.
- **Rewards:**
  - Staying in safe regions: `+300 to +600`
  - Closing doors: `+300 to +500`
  - Avoiding seekers (penalty when seen): `-100`

## ⚙️ Learning Parameters

| Parameter            | Value   |
| -------------------- | ------- |
| Exploration rate (ε) | 0.2     |
| Learning rate (α)    | 0.1     |
| Discount factor (γ)  | 0.9     |
| Vision range         | 8 cells |
| Field of view        | 60°     |

## 📄 Output Files

- agent_rewards.txt : Logs each agent's performance per round
- qtable*agent*[type]\_[id].txt : Saved Q-tables for each agent

## 🛠️ Customization

Modify these parameters in q_learning.py :

- Reward values in **init**
- Learning rates
- Vision parameters

## 🚀 Future Improvements

- Add more complex maze generation
- Implement multi-agent cooperation
- Add more sophisticated vision/reward systems
- Include neural network based learning
- Include dual q-network learning for collaboration
