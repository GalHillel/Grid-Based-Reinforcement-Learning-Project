import ast
import numpy as np
import random


# Define the gridworld environment
class GridWorldFree:
    def __init__(self, width, height, terminal_states, action_prob, reward, discount_factor=0.5):
        self.width = width
        self.height = height
        self.terminal_states = {(x, y): reward for x, y, reward in terminal_states}
        self.action_prob = action_prob
        self.reward = reward
        self.discount_factor = discount_factor
        self.actions = ['U', 'D', 'L', 'R']
        self.q_values = np.zeros((height, width, len(self.actions)))
        self.epsilon = 0.1
        self.learning_rate = 0.1

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_state(self, state, action):
        if self.is_terminal(state):
            return state

        x, y = state
        if action == 'U':
            next_state = (x, y + 1)
        elif action == 'D':
            next_state = (x, y - 1)
        elif action == 'L':
            next_state = (x - 1, y)
        elif action == 'R':
            next_state = (x + 1, y)

        # Check for walls or out-of-bounds
        if next_state[0] < 0 or next_state[0] >= self.width or next_state[1] < 0 or next_state[
            1] >= self.height or next_state in self.terminal_states and self.terminal_states[next_state] == 0:
            return state

        return next_state

    def get_reward(self, state):
        return self.terminal_states.get(state, self.reward)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_values[state[1], state[0]])]

    def q_learning(self, episodes=1000):
        for episode in range(episodes):
            state = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            while self.is_terminal(state):
                state = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

            while not self.is_terminal(state):
                action = self.choose_action(state)
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(next_state)

                best_next_action = np.argmax(self.q_values[next_state[1], next_state[0]])
                td_target = reward + self.discount_factor * self.q_values[
                    next_state[1], next_state[0], best_next_action]
                td_error = td_target - self.q_values[state[1], state[0], self.actions.index(action)]
                self.q_values[state[1], state[0], self.actions.index(action)] += self.learning_rate * td_error

                state = next_state

    def extract_policy(self):
        policy = np.empty((self.height, self.width), dtype=str)
        for y in range(self.height):
            for x in range(self.width):
                state = (x, y)
                if self.is_terminal(state):
                    policy[y, x] = 'T' if self.terminal_states[state] != 0 else 'W'
                else:
                    best_action = np.argmax(self.q_values[y, x])
                    policy[y, x] = self.actions[best_action]
        return policy

    def get_state_values(self):
        state_values = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                if self.is_terminal((x, y)):
                    state_values[y, x] = self.terminal_states[(x, y)]
                else:
                    state_values[y, x] = np.max(self.q_values[y, x])
        return state_values


def main():
    with open("instances.txt", "r") as file:
        data = file.readlines()

    for i in range(10):
        W = int(data[1].split("=")[1])
        H = int(data[2].split("=")[1])
        L = ast.literal_eval(data[3].split("=")[1].strip())
        p = float(data[4].split("=")[1])
        r = float(data[5].split("=")[1])

        # get next lines
        data = data[7:]

        # Create gridworld instance
        gridworld = GridWorldFree(W, H, L, p, r)

        # Train using Q-learning
        gridworld.q_learning(episodes=10000)

        print(f"---------------------- instance={i + 1} ----------------------")
        print(f"W={W} | H={H} | p={p} | r={r} | L={L}\n")

        # Print the state values
        values = gridworld.get_state_values()
        print(values)
        print()

        # Extract and print the policy
        optimal_policy = gridworld.extract_policy()
        for row in np.flip(optimal_policy, 0):
            print(row)


if __name__ == '__main__':
    main()

