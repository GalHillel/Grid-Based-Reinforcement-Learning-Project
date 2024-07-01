import ast
import numpy as np
import pandas as pd

from GridWorld import GridWorldAdditive
from ValueIteration import ValueIteration
from ModelFree import GridWorldFree
from ModelBased import GridWorldBased

discount = 0.5


def main():
    """
    Main function to run experiments on different Markov Decision Process (MDP) models.

    Reads instances from a file, computes values using Value Iteration, Model-Based RL, and Model-Free RL,
    and calculates differences between their state values.

    Outputs average differences and differences per cell to a CSV file.

    Example:
        To run:
        python Results.py
    """

    # Open the instances file
    with open("data/tests/instances.txt", "r") as file:
        data = file.readlines()

    # Lists to store average differences and differences per cell
    average_differences = []
    cells_difference = []

    # Iterate over each instance in the file
    for i in range(10):
        # Parse parameters for the current instance
        W = int(data[1].split("=")[1])
        H = int(data[2].split("=")[1])
        L = ast.literal_eval(data[3].split("=")[1].strip())
        p = float(data[4].split("=")[1])
        r = float(data[5].split("=")[1])

        # Move to the next set of parameters
        data = data[7:]

        print(f"---------------------- instance={i + 1} ----------------------")
        print(f"W={W} | H={H} | p={p} | r={r} | L={L}\n")

        # Value Iteration ----------------------------------------------------

        # Extract terminal states and walls for GridWorldAdditive
        terminals = {(-i[1] + (H - 1), i[0]): i[2] for i in L if i[2] != 0}
        walls = [(-i[1] + (H - 1), i[0]) for i in L if i[2] == 0]

        # Initialize GridWorldAdditive and perform value iteration
        gwa = GridWorldAdditive((H, W), p, walls, terminals, r)
        vi = ValueIteration()
        values_MDP = np.zeros((H, W))
        temp = vi.valueIteration(gwa, discount, 100)
        for x in range(H):
            for y in range(W):
                values_MDP[-x + (H - 1)][y] = temp[(x, y)]

        # Model Based ------------------------------------------------------

        # Initialize GridWorldBased and perform value iteration
        gridworld_b = GridWorldBased(W, H, L, p, r)
        gridworld_b.value_iteration()
        values_MBRL = gridworld_b.value

        # Model Free ---------------------------------------------------------

        # Initialize GridWorldFree and perform Q-learning
        gridworld = GridWorldFree(W, H, L, p, r)
        gridworld.q_learning(episodes=10000)
        values_MFRL = gridworld.get_state_values()

        # Compute and print average differences
        diff_mdp_mbrl = (values_MDP - values_MBRL).mean()
        diff_mdp_mfrl = (values_MDP - values_MFRL).mean()
        diff_mbrl_mfrl = (values_MBRL - values_MFRL).mean()

        print(f"average(d(MDP, MBRL))= {diff_mdp_mbrl}")
        print(f"average(d(MDP, MFRL))= {diff_mdp_mfrl}")
        print(f"average(d(MBRL, MFRL))= {diff_mbrl_mfrl}\n")

        # Compute differences per cell and create a DataFrame
        d = {}
        for x in range(H):
            for y in range(W):
                d[f"({x}, {y})"] = [
                    values_MDP[x][y] - values_MBRL[x][y],
                    values_MDP[x][y] - values_MFRL[x][y],
                    values_MBRL[x][y] - values_MFRL[x][y],
                ]
        df = pd.DataFrame(d)
        print(df)

        # Append DataFrame to cells_difference list
        cells_difference.append(df)

    # Write differences per cell to a CSV file
    with open("data/results/results.csv", "a") as f:
        i = 1
        for df in cells_difference:
            f.write(f"Test {i}\n")
            df.to_csv(f, index=False)
            f.write("\n")
            i += 1


if __name__ == "__main__":
    main()
