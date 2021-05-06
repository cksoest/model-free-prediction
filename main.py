from grid import Grid
import numpy as np


rewards = np.array([
            [-1, -1, -1, 40],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [10, -2, -1, -1]
        ])
policy = [
    ["D", "L", "R", "X"],
    ["D", "R", "D", "U"],
    ["R", "U", "D", "U"],
    ["X", "L", "R", "U"]
]
terminal_states = [(0, 3), (3, 0)]
gamma = 1

grid = Grid(rewards, policy, terminal_states, gamma)
# grid.generate_episode()
# grid.monte_carlo_policy_evaluation()
# grid.temporal_difference_learning(0.01)
# grid.run_td(500, 0.01, verbose=True)
# grid.run_sarsa(500, 0.01)
# grid.run_sarsa(1000, 0.01)
grid.run_mc(500)
print(grid.q)
