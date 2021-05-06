import numpy as np
import copy
from itertools import product
import random
import matplotlib.pyplot as plt


class Grid:
    def __init__(self, rewards, policy, terminal_states, gamma):
        self.rewards = rewards
        self.policy = policy
        self.soft_policy = {(i, j, a): 0 for i, j, a in product(range(rewards.shape[0]), range(rewards.shape[1]), ["U", "R", "D", "L"])}
        self.terminal_states = terminal_states
        self.start_points = self.generate_start_points()
        self.gamma = gamma
        self.values = np.zeros(self.rewards.shape)
        self.returns = {(i, j): [] for i, j in product(range(rewards.shape[0]), range(rewards.shape[1]))}
        self.q = {(i, j, a): 0 for i, j, a in product(range(rewards.shape[0]), range(rewards.shape[1]), ["U", "R", "D", "L"])}

    def generate_start_points(self):
        start_points = []
        for i, j in product(range(self.rewards.shape[0]), range(self.rewards.shape[1])):
            if (i, j) in self.terminal_states:
                continue
            else:
                start_points.append((i, j))
        return start_points

    @staticmethod
    def new_state_based_on_action(current_state, action):
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
        return new_state

    def generate_episode(self, soft=False):
        episode = []
        current_state = random.choice(self.start_points)
        episode.append(current_state)

        while current_state not in self.terminal_states:
            if soft:
                u = self.soft_policy[(current_state[0], current_state[1], "U")]
                r = self.soft_policy[(current_state[0], current_state[1], "R")]
                d = self.soft_policy[(current_state[0], current_state[1], "D")]
                l = self.soft_policy[(current_state[0], current_state[1], "L")]
                actions = ["U", "R", "D", "L"]
                action_weights = [u, r, d, l]
                action = random.choices(actions, weights=action_weights, k=1)[0]
                new_state = self.new_state_based_on_action(current_state, action)
            else:
                action = self.policy[current_state[0]][current_state[1]]
                new_state = self.new_state_based_on_action(current_state, action)

            if new_state[0] in np.arange(self.rewards.shape[0]) and new_state[1] in np.arange(self.rewards.shape[1]):
                current_state = new_state

            episode[-1] = (episode[-1], action, self.rewards[current_state[0]][current_state[1]])
            episode.append(current_state)
        return episode

    def monte_carlo_policy_evaluation(self):
        episode = self.generate_episode()
        episode_copy = copy.deepcopy(episode)
        g = 0
        for step in reversed(episode[:-1]):
            episode_copy.remove(step)
            g = self.gamma*g + step[2]
            if step[0] not in [step_2[0] for step_2 in episode_copy]:
                self.returns[step[0]].append(g)
                self.values[step[0][0]][step[0][1]] = np.mean(self.returns[step[0]])

    def temporal_difference_learning(self, step_size):
        current_state = random.choice(self.start_points)
        while current_state not in self.terminal_states:
            action = self.policy[current_state[0]][current_state[1]]
            new_state = self.new_state_based_on_action(current_state, action)
            if new_state[0] in np.arange(self.rewards.shape[0]) and new_state[1] in np.arange(self.rewards.shape[1]):
                reward = self.rewards[new_state[0]][new_state[1]]
                self.values[current_state[0]][current_state[1]] += step_size * (reward + self.gamma * self.values[new_state[0]][new_state[1]] - self.values[current_state[0]][current_state[1]])
                current_state = new_state

            else:
                reward = self.rewards[current_state[0]][current_state[1]]
                self.values[current_state[0]][current_state[1]] += step_size * (reward + self.gamma * self.values[current_state[0]][current_state[1]] - self.values[current_state[0]][current_state[1]])

    def sarsa(self, step_size, max_q=False):
        current_state = random.choice(self.start_points)
        qs = {}
        for k in self.q:
            if k[0] == current_state[0] and k[1] == current_state[1]:
                qs[k] = self.q[k]
        max_q = max(qs, key=qs.get)
        action = max_q[2]

        while current_state not in self.terminal_states:
            new_state = self.new_state_based_on_action(current_state, action)

            if new_state[0] in np.arange(self.rewards.shape[0]) and new_state[1] in np.arange(self.rewards.shape[1]):
                reward = self.rewards[new_state[0]][new_state[1]]

                qs_1 = {}
                for k in self.q:
                    if k[0] == new_state[0] and k[1] == new_state[1]:
                        qs_1[k] = self.q[k]
                max_q_1 = max(qs, key=qs.get)
                action_1 = max_q_1[2]

                self.q[(current_state[0], current_state[1], action)] += step_size * (reward+self.gamma*self.q[(new_state[0], new_state[1], action_1)] - self.q[(current_state[0], current_state[1], action)])
                current_state = new_state
                action = action_1
            else:
                reward = self.rewards[current_state[0]][current_state[1]]
                self.q[(current_state[0], current_state[1], action)] += step_size * (reward+self.gamma*self.q[(current_state[0], current_state[1], action)] - self.q[(current_state[0], current_state[1], action)])

    def q_learning(self, step_size):
        current_state = random.choice(self.start_points)
        while current_state not in self.terminal_states:
            qs = {}
            for k in self.q:
                if k[0] == current_state[0] and k[1] == current_state[1]:
                    qs[k] = self.q[k]
            max_q = max(qs, key=qs.get)
            action = max_q[2]
            max_q = qs[max_q]

            new_state = self.new_state_based_on_action(current_state, action)
            if new_state[0] in np.arange(self.rewards.shape[0]) and new_state[1] in np.arange(self.rewards.shape[1]):
                qs_1 = {}
                for k in self.q:
                    if k[0] == new_state[0] and k[1] == new_state[1]:
                        qs_1[k] = self.q[k]
                max_q1 = max(qs_1, key=qs_1.get)
                action = max_q1[2]
                max_q1 = qs_1[max_q1]

                reward = self.rewards[new_state[0]][new_state[1]]
                self.q[(current_state[0], current_state[1], action)] += step_size * (reward + self.gamma * max_q1 - self.q[(current_state[0], current_state[1], action)])
                current_state = new_state
            else:
                reward = self.rewards[current_state[0]][current_state[1]]
                self.q[(current_state[0], current_state[1], action)] += step_size * (reward + self.gamma * max_q - self.q[(current_state[0], current_state[1], action)])


    # def on_policy_first_visit_monte_carlo_control(self, epsilon):


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

    def run_mc(self, amount_iterations, verbose=False):
        for i in range(amount_iterations):
            print("iteration {}".format(i))
            self.monte_carlo_policy_evaluation()
            if verbose:
                self.print_values()

    def run_td(self, amount_iterations, step_size, verbose=False):
        for i in range(amount_iterations):
            print("iteration {}".format(i))
            self.temporal_difference_learning(step_size)
            if verbose:
                self.print_values()

    def run_sarsa(self, amount_iterations, step_size, verbose=False):
        for i in range(amount_iterations):
            print("iteration {}".format(i))
            self.sarsa(step_size)
            if verbose:
                self.print_values()

    def run_q_learning(self, amount_iterations, step_size, verbose=False):
        for i in range(amount_iterations):
            print("iteration {}".format(i))
            self.q_learning(step_size)
            if verbose:
                self.print_values()
