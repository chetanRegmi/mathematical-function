# Initialize Q-table
num_states = 16  # 4x4 grid
num_actions = 4  # Up, Down, Left, Right
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Simulate one step of Q-learning
state = 0  # Starting state
action = 1  # Move down
reward = -1  # Reward for moving
next_state = 4  # New state after action

# Update Q-value
q_learning_update(Q, state, action, reward, next_state, alpha, gamma)

print("Updated Q-table:")
print(Q)