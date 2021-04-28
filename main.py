from grid import Grid
import numpy as np


rewards = np.array([
            [-1, -1, -1, 40],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [10, -2, -1, -1]
        ])
terminal_states = [(0, 3), (3, 0)]
gamma = 1

grid = Grid(rewards, terminal_states, gamma, False)
grid.monte_carlo_policy_evaluation()
grid.run(500, verbose=True)
