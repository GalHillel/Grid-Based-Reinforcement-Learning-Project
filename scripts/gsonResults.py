import json
import numpy as np
import pandas as pd
from GridWorld import GridWorldAdditive
from ValueIteration import ValueIteration
from ModelFree import GridWorldFree
from ModelBased import GridWorldBased

discount = 0.5


def load_grid_from_json(filename):
    with open(filename, "r") as f:
        grid = json.load(f)
    return grid


def main():
    """
    Main function to run experiments on different Markov Decision Process (MDP) models.

    Reads instances from generated JSON files, computes values using Value Iteration, Model-Based RL, and Model-Free RL,
    and calculates differences between their state values.

    Outputs average differences and differences per cell to a CSV file.
    """

    grid_files = [
        "data/tests/grid_t1_r-0.04.json",
        "data/tests/grid_t2_r0.04.json",
        "data/tests/grid_t3_r-1.json",
        "data/tests/grid_t4_r-0.5.json",
        "data/tests/grid_t5_r-0.25.json",
        "data/tests/grid_t6_r-0.5.json",
        "data/tests/grid_t7_r-0.25.json",
        "data/tests/grid_t8_r-0.1.json",
        "data/tests/grid_t8_r-0.3.json",
        "data/tests/grid_t8_r-0.5.json",
        "data/tests/grid_t9_r-0.2.json",
        "data/tests/grid_t9_r-0.4.json",
        "data/tests/grid_t9_r-0.6.json",
        "data/tests/grid_t10_r-0.2.json",
        "data/tests/grid_t10_r-0.4.json",
        "data/tests/grid_t10_r-0.6.json",
        "data/tests/grid_t11_r-0.1.json",
        "data/tests/grid_t11_r-0.3.json",
        "data/tests/grid_t11_r-0.5.json",
        "data/tests/grid_t12_r-0.2.json",
        "data/tests/grid_t12_r-0.4.json",
        "data/tests/grid_t12_r-0.6.json",
        "data/tests/grid_t13_r-0.1.json",
        "data/tests/grid_t13_r-0.3.json",
        "data/tests/grid_t13_r-0.5.json",
        "data/tests/grid_t14_r-0.2.json",
        "data/tests/grid_t14_r-0.4.json",
        "data/tests/grid_t14_r-0.6.json",
        "data/tests/grid_t15_r-0.2.json",
        "data/tests/grid_t15_r-0.3.json",
        "data/tests/grid_t15_r-0.4.json",
    ]

    cells_difference = []

    for idx, filename in enumerate(grid_files):
        grid = load_grid_from_json(filename)
        H, W = len(grid), len(grid[0])

        print(f"---------------------- instance={idx + 1} ----------------------")
        print(f"W={W} | H={H} | grid={grid}\n")

        # Value Iteration ----------------------------------------------------

        terminals = {}
        walls = []
        for y in range(H):
            for x in range(W):
                if grid[y][x] != 0:
                    terminals[(y, x)] = grid[y][x]
                else:
                    walls.append((y, x))

        shape = (H, W)
        gwa = GridWorldAdditive(
            shape, prob=0.8, walls=walls, terminals=terminals, reward=0.0
        )
        vi = ValueIteration()
        values_MDP = vi.valueIteration(gwa, discount, 100)

        # Reshape values_MDP properly
        values_MDP_array = np.zeros((H, W))
        for x in range(H):
            for y in range(W):
                values_MDP_array[x, y] = values_MDP[(x, y)]

        # Model Based ------------------------------------------------------

        gridworld_b = GridWorldBased(W, H, L=[], p=0.8, r=-0.04)
        gridworld_b.value_iteration()
        values_MBRL = gridworld_b.value

        # Model Free -------------------------------------------------------

        gridworld = GridWorldFree(W, H, L=[], p=0.8, r=-0.04)
        gridworld.q_learning(episodes=10000)
        values_MFRL = gridworld.get_state_values()

        # Compute and print average differences
        diff_mdp_mbrl = (values_MDP_array - values_MBRL).mean()
        diff_mdp_mfrl = (values_MDP_array - values_MFRL).mean()
        diff_mbrl_mfrl = (values_MBRL - values_MFRL).mean()

        print(f"average(d(MDP, MBRL))= {diff_mdp_mbrl}")
        print(f"average(d(MDP, MFRL))= {diff_mdp_mfrl}")
        print(f"average(d(MBRL, MFRL))= {diff_mbrl_mfrl}\n")

        # Compute differences per cell and create a DataFrame
        d = {}
        for x in range(H):
            for y in range(W):
                d[f"({x}, {y})"] = [
                    values_MDP_array[x, y] - values_MBRL[x][y],
                    values_MDP_array[x, y] - values_MFRL[x][y],
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
