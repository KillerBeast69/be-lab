"""
====================================================================
ASSIGNMENT: Reinforcement Learning (Maze Environment Exploration)
====================================================================

1. ASSIGNMENT DETAILS
---------------------
Objective: Implement Reinforcement Learning (specifically Q-Learning) 
using a maze environment. The agent needs to explore the maze, avoid 
walls, and find the optimal path from the Start state to the Goal state 
while maximizing its cumulative reward.

2. VIVA THEORY
--------------
- What is Reinforcement Learning (RL)?
  RL is a type of machine learning where an "Agent" learns to make 
  decisions by taking "Actions" in an "Environment" to maximize a 
  numerical "Reward" over time.
- What is Q-Learning?
  It is a model-free, value-based RL algorithm. The agent maintains a 
  Q-table, which evaluates how good it is to take a specific action in a 
  specific state.
- What are States, Actions, and Rewards?
  * States (S): The agent's current position in the maze.
  * Actions (A): Moves the agent can make (Up, Down, Left, Right).
  * Rewards (R): Feedback from the environment (e.g., +100 for reaching 
    the goal, -1 for each step to encourage speed, -10 for hitting walls).
- What is the Bellman Equation?
  It updates the Q-value based on the immediate reward plus the discounted 
  maximum future reward. 
  Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max(Q(s',a')) - Q(s,a)]
- What is the Exploration vs. Exploitation trade-off (Epsilon-Greedy)?
  * Exploration: Trying random actions to discover new paths (Epsilon).
  * Exploitation: Using the known Q-table to take the best action.
  The agent explores a lot initially, then slowly shifts to exploitation.

3. CODE DETAILS
---------------
- maze (numpy array): Represents the grid (0: path, 1: wall, 2: start, 3: goal).
- Q-Table: A 3D numpy array initialized with zeros. Shape is (rows, cols, 4) 
  because there are 4 possible actions for every cell.
- choose_action(): Implements the epsilon-greedy strategy.
- step(): Takes the action, computes the next state, and returns the reward.
- train_agent(): Runs thousands of episodes where the agent repeatedly attempts 
  the maze, updating its Q-values using the Bellman Equation.
- matplotlib.pyplot (plt): Used to visualize the maze and draw the optimal 
  learned path after training is complete.

4. IMPLEMENTATION & EXECUTION
-----------------------------
Prerequisites (Necessary Packages):
- numpy
- matplotlib

How to run:
1. Open your terminal or command prompt.
2. Install the required packages by running:
   pip install numpy matplotlib
3. Save this code in a file named `rl_maze.py`.
4. Execute the script by running:
   python rl_maze.py
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. Environment Definition
# ==========================================
# 0 = Empty path
# 1 = Wall / Obstacle
# 2 = Start
# 3 = Goal
MAZE = np.array([
    [2, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 3]
])

ROWS, COLS = MAZE.shape
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = len(ACTIONS)

# Define starting and goal positions
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

# ==========================================
# 2. Reinforcement Learning Parameters
# ==========================================
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor (values future rewards)
EPSILON = 1.0     # Initial exploration rate
EPSILON_MIN = 0.05 # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay rate per episode
EPISODES = 500    # Total number of training attempts

# Initialize Q-Table with zeros: dimensions [rows, cols, actions]
q_table = np.zeros((ROWS, COLS, NUM_ACTIONS))

# ==========================================
# 3. Helper Functions
# ==========================================
def get_next_state(state, action_idx):
    """Calculates the next state based on the current state and chosen action."""
    r, c = state
    action = ACTIONS[action_idx]
    
    if action == 'UP':    r = max(r - 1, 0)
    elif action == 'DOWN':  r = min(r + 1, ROWS - 1)
    elif action == 'LEFT':  c = max(c - 1, 0)
    elif action == 'RIGHT': c = min(c + 1, COLS - 1)
    
    return (r, c)

def step(state, action_idx):
    """Executes the action and returns next_state, reward, and done flag."""
    next_state = get_next_state(state, action_idx)
    r, c = next_state
    
    # Check what is in the next state cell
    if MAZE[r, c] == 1:
        # Hit a wall: return to original state, give negative reward
        return state, -10, False
    elif next_state == GOAL_STATE:
        # Reached the goal
        return next_state, 100, True
    else:
        # Valid move on empty path
        return next_state, -1, False

def choose_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1) # Explore
    else:
        return np.argmax(q_table[state[0], state[1]]) # Exploit

# ==========================================
# 4. Training the Agent (Q-Learning)
# ==========================================
def train_agent():
    global EPSILON
    print("Starting Q-Learning Training...")
    
    for episode in range(EPISODES):
        state = START_STATE
        done = False
        steps = 0
        
        while not done and steps < 100: # Max 100 steps per episode to prevent infinite loops
            # 1. Choose action
            action_idx = choose_action(state, EPSILON)
            
            # 2. Take action
            next_state, reward, done = step(state, action_idx)
            
            # 3. Update Q-Table using Bellman Equation
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            td_target = reward + GAMMA * q_table[next_state[0], next_state[1], best_next_action]
            td_error = td_target - q_table[state[0], state[1], action_idx]
            q_table[state[0], state[1], action_idx] += ALPHA * td_error
            
            state = next_state
            steps += 1
            
        # Decay Epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{EPISODES} completed. Current Epsilon: {EPSILON:.2f}")

    print("Training Complete!")

# ==========================================
# 5. Extract and Visualize the Policy
# ==========================================
def get_optimal_path():
    """Follows the greedy policy to find the optimal path from Start to Goal."""
    path = [START_STATE]
    state = START_STATE
    steps = 0
    
    while state != GOAL_STATE and steps < 30:
        action_idx = np.argmax(q_table[state[0], state[1]])
        next_state, _, _ = step(state, action_idx)
        
        # If the agent gets stuck or loops, break
        if next_state == state or next_state in path:
            print("Warning: Agent got stuck, path incomplete.")
            break
            
        path.append(next_state)
        state = next_state
        steps += 1
        
    return path

def plot_maze_and_path(path):
    """Draws the maze grid and overlays the learned optimal path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Colors: 0: white (path), 1: black (wall), 2: green (start), 3: red (goal)
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'lightgreen', 'salmon'])
    
    ax.imshow(MAZE, cmap=cmap)
    
    # Add gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='gray', linewidth=2)
    ax.set_xticks(np.arange(-0.5, COLS, 1))
    ax.set_yticks(np.arange(-0.5, ROWS, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Plot the optimal path
    if path:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax.plot(path_x, path_y, marker='o', color='blue', markersize=10, linewidth=3, label='Optimal Path')
    
    # Mark Start and Goal text
    ax.text(0, 0, 'Start', ha='center', va='center', color='black', fontweight='bold')
    ax.text(4, 4, 'Goal', ha='center', va='center', color='black', fontweight='bold')
    
    plt.title('Reinforcement Learning: Maze Solution (Q-Learning)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('rl_maze_solution.png')
    print("-> Saved maze solution plot as 'rl_maze_solution.png'")

def main():
    train_agent()
    optimal_path = get_optimal_path()
    print(f"\nOptimal Path Found: {optimal_path}")
    print("Generating Visualization...")
    plot_maze_and_path(optimal_path)
    print("\nExecution Complete. Please check the directory for the generated image plot.")

if __name__ == "__main__":
    main()
