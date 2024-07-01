import numpy as np
import ast


class GridWorldBased:
    """
    Grid World Environment for Value Iteration.

    Attributes:
        w (int): Width of the grid.
        h (int): Height of the grid.
        p (float): Probability of moving in the intended direction.
        r (float): Default reward value for non-terminal states.
        discount (float): Discount factor for future rewards.
        grid (numpy.ndarray): Grid representation with rewards and terminal states.
        policy (numpy.ndarray): Policy grid indicating the optimal action for each state.
        value (numpy.ndarray): Value grid storing the expected cumulative rewards for each state.
    """

    def __init__(self, w, h, L, p, r, discount=0.5):
        """
        Initializes the grid world environment.

        Args:
            w (int): Width of the grid.
            h (int): Height of the grid.
            L (list): List of terminal states and their rewards in the format [(x, y, reward), ...].
            p (float): Probability of moving in the intended direction.
            r (float): Default reward value for non-terminal states.
            discount (float, optional): Discount factor for future rewards (default is 0.5).
        """
        self.w = w
        self.h = h
        self.p = p
        self.r = r
        self.discount = discount
        self.grid = np.full((h, w), r)
        self.policy = np.full((h, w), " ")
        self.value = np.zeros((h, w))

        for x, y, reward in L:
            if reward == 0:
                self.grid[y, x] = None  # Wall
            else:
                self.grid[y, x] = reward  # Terminal state
                self.value[y, x] = reward  # Terminal state value
                self.policy[y, x] = "T"

        self.actions = ["U", "D", "L", "R"]
        self.action_prob = {
            "U": {"U": p, "L": (1 - p) / 2, "R": (1 - p) / 2},
            "D": {"D": p, "L": (1 - p) / 2, "R": (1 - p) / 2},
            "L": {"L": p, "U": (1 - p) / 2, "D": (1 - p) / 2},
            "R": {"R": p, "U": (1 - p) / 2, "D": (1 - p) / 2},
        }

    def is_terminal(self, x, y):
        """
        Checks if a specific state is a terminal state.

        Args:
            x (int): X-coordinate of the state.
            y (int): Y-coordinate of the state.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return self.grid[y, x] is not None and self.grid[y, x] != self.r

    def step(self, x, y, action):
        """
        Computes the next state based on the current state and action.

        Args:
            x (int): Current X-coordinate.
            y (int): Current Y-coordinate.
            action (str): Action to take ('U', 'D', 'L', 'R').

        Returns:
            tuple: Next state coordinates (x, y).
        """
        if action == "U":
            return x, min(y + 1, self.h - 1)
        elif action == "D":
            return x, max(y - 1, 0)
        elif action == "L":
            return max(x - 1, 0), y
        elif action == "R":
            return min(x + 1, self.w - 1), y

    def expected_value(self, x, y, action):
        """
        Calculates the expected value for taking a specific action from a given state.

        Args:
            x (int): Current X-coordinate.
            y (int): Current Y-coordinate.
            action (str): Action to take ('U', 'D', 'L', 'R').

        Returns:
            float: Expected value for the action.
        """
        expected_val = 0
        for act in self.action_prob[action]:
            new_x, new_y = self.step(x, y, act)
            if self.grid[new_y, new_x] is None:
                new_x, new_y = x, y
            expected_val += self.action_prob[action][act] * self.value[new_y, new_x]
        return expected_val

    def value_iteration(self, iterations=1000):
        """
        Performs value iteration to compute the optimal policy and state values.

        Args:
            iterations (int, optional): Number of iterations for value iteration (default is 1000).
        """
        for _ in range(iterations):
            new_value = np.copy(self.value)
            for y in range(self.h):
                for x in range(self.w):
                    if self.is_terminal(x, y) or self.grid[y, x] is None:
                        continue
                    max_val = float("-inf")
                    best_action = None
                    for action in self.actions:
                        val = self.expected_value(x, y, action)
                        if val > max_val:
                            max_val = val
                            best_action = action
                    new_value[y, x] = self.grid[y, x] + self.discount * max_val
                    self.policy[y, x] = best_action
            self.value = new_value

    def get_policy(self):
        """
        Returns the computed optimal policy.

        Returns:
            numpy.ndarray: Policy grid with optimal actions for each state.
        """
        return self.policy


def main():
    """
    Main function to run the grid world instances from file and perform value iteration.
    """
    with open("data/tests/instances.txt", "r") as file:
        data = file.readlines()

    for i in range(10):
        W = int(data[1].split("=")[1])
        H = int(data[2].split("=")[1])
        L = ast.literal_eval(data[3].split("=")[1].strip())
        p = float(data[4].split("=")[1])
        r = float(data[5].split("=")[1])

        # Get next lines
        data = data[7:]

        # Create GridWorldBased instance
        gridworld = GridWorldBased(W, H, L, p, r)

        # Perform value iteration to find optimal policy
        gridworld.value_iteration()

        print(f"---------------------- Instance={i + 1} ----------------------")
        print(f"W={W} | H={H} | p={p} | r={r} | L={L}\n")

        # Print values
        for row in np.flip(gridworld.value, 0):
            print(row)
        print()

        # Retrieve the policy
        optimal_policy = gridworld.get_policy()
        for row in np.flip(optimal_policy, 0):
            print(row)


if __name__ == "__main__":
    main()
