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
grid.run_q_learning(500, 0.05, verbose=True)
