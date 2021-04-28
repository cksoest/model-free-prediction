import numpy as np
import copy
from itertools import product
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple


class Grid:
    def __init__(self, rewards, terminal_states, gamma, exploring_starts):
        self.rewards = rewards
        self.terminal_states = terminal_states
        self.gamma = gamma
        self.values = np.zeros(self.rewards.shape)
        self.greedy_policy = np.zeros(self.rewards.shape)
        self.greedy_policy = self.greedy_policy.tolist()
        self.start_points = self.generate_start_points()
        self.returns = {(i, j): [] for i, j in product(range(rewards.shape[0]), range(rewards.shape[1]))}
        self.exploring_starts = exploring_starts

    def generate_start_points(self):
        start_points = []
        for i, j in product(range(self.rewards.shape[0]), range(self.rewards.shape[1])):
            if (i, j) in self.terminal_states:
                continue
            else:
                start_points.append((i, j))
        return start_points

    def generate_episode(self):
        episode = []
        actions = ["U", "R", "D", "L"]

        if self.exploring_starts:
            current_state = start_state
        else:
            current_state = random.choice(self.start_points)

        episode.append(current_state)
        while current_state not in self.terminal_states:
            action = random.choice(actions)
            u = (current_state[0] - 1, current_state[1])
            r = (current_state[0], current_state[1] + 1)
            d = (current_state[0] + 1, current_state[1])
            l = (current_state[0], current_state[1] - 1)

            if action == "U":
                chance = random.random()
                if chance > 0.3:
                    new_state = u
                elif chance < 0.3 and chance > 0.2:
                    new_state = r
                elif chance < 0.2 and chance > 0.1:
                    new_state = d
                else:
                    new_state = r

            elif action == "R":
                chance = random.random()
                if chance > 0.3:
                    new_state = r
                elif chance < 0.3 and chance > 0.2:
                    new_state = d
                elif chance < 0.2 and chance > 0.1:
                    new_state = l
                else:
                    new_state = u

            elif action == "D":
                chance = random.random()
                if chance > 0.3:
                    new_state = d
                elif chance < 0.3 and chance > 0.2:
                    new_state = l
                elif chance < 0.2 and chance > 0.1:
                    new_state = u
                else:
                    new_state = r

            else:
                chance = random.random()
                if chance > 0.3:
                    new_state = l
                elif chance < 0.3 and chance > 0.2:
                    new_state = u
                elif chance < 0.2 and chance > 0.1:
                    new_state = r
                else:
                    new_state = d

            if new_state[0] in np.arange(self.rewards.shape[0]) and new_state[1] in np.arange(
                    self.rewards.shape[1]):
                current_state = new_state

            episode[-1] = (episode[-1], action, self.rewards[current_state[0]][current_state[1]])
            if current_state not in self.terminal_states:
                episode.append(current_state)
        return episode

    def monte_carlo_policy_evaluation(self):
        episode = self.generate_episode()
        episode_copy = copy.deepcopy(episode)
        g = 0
        for step in reversed(episode):
            episode_copy.remove(step)
            g = self.gamma*g + step[2]
            if step[0] not in [step[0] for step in episode_copy]:
                self.returns[step[0]].append(g)
                self.values[step[0][0]][step[0][1]] = np.mean(self.returns[step[0]])

    def print_values(self):
        # z = self.values
        # z = z.tolist()
        # z.reverse()
        # # plt.pcolormesh(z, cmap="Reds")
        # # plt.title("Values of grid")
        # # plt.show()
        for row in self.values:
            values = "|"
            for v in row:
                values += str(round(v, 1))
                values += "|"
            print(values)
        print("\n")

    def run(self, amount_iterations, verbose=False):

        for i in range(amount_iterations):
            print("iteration {}".format(i))
            self.monte_carlo_policy_evaluation()
            if verbose:
                self.print_values()
