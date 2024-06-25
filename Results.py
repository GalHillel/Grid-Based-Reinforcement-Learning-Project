import ast
import numpy as np
import pandas as pd

from GridWorld import GridWorldAdditive
from ValueIteration import ValueIteration
from ModelFree import GridWorldFree
from ModelBased import GridWorldBased

discount = 0.5


def main():
    with open("instances.txt", "r") as file:
        data = file.readlines()

    average_differences = []
    cells_difference = []

    for i in range(10):
        W = int(data[1].split("=")[1])
        H = int(data[2].split("=")[1])
        L = ast.literal_eval(data[3].split("=")[1].strip())
        p = float(data[4].split("=")[1])
        r = float(data[5].split("=")[1])

        # get next lines
        data = data[7:]

        print(f"---------------------- instance={i + 1} ----------------------")
        print(f"W={W} | H={H} | p={p} | r={r} | L={L}\n")

        # Value iteration ----------------------------------------------------

        terminals = {(-i[1] + (H - 1), i[0]): i[2] for i in L if i[2] != 0}
        walls = [(-i[1] + (H - 1), i[0]) for i in L if i[2] == 0]
        gwa = GridWorldAdditive((H, W), p, walls, terminals, r)
        vi = ValueIteration()
        temp = vi.valueIteration(gwa, discount, 100)
        values_MDP = np.zeros((H, W))
        for x in range(H):
            for y in range(W):
                values_MDP[-x + (H-1)][y] = temp[(x, y)]
        # print(values_MDP)
        # print()

        # Model Based ------------------------------------------------------

        gridworld_b = GridWorldBased(W, H, L, p, r)
        gridworld_b.value_iteration()

        # print values
        values_MBRL = gridworld_b.value
        # print(values_MBRL)
        # print()

        # Model Free ---------------------------------------------------------

        gridworld = GridWorldFree(W, H, L, p, r)
        gridworld.q_learning(episodes=10000)

        # Print the state values
        values_MFRL = gridworld.get_state_values()
        # print(values_MFRL)
        # print()

        # Difference average
        print(f"average(d(MDP, MBRL))= {(values_MDP-values_MBRL).mean()}")
        print(f"average(d(MDP, MFRL))= {(values_MDP - values_MFRL).mean()}")
        print(f"average(d(MBRL, MFRL))= {(values_MBRL - values_MFRL).mean()}\n")

        # Difference per cell
        d = {}
        for x in range(H):
            for y in range(W):
                d[f"({x}, {y})"] = [values_MDP[x][y] - values_MBRL[x][y],
                             values_MDP[x][y] - values_MFRL[x][y],
                             values_MBRL[x][y] - values_MFRL[x][y]]
        df = pd.DataFrame(d)
        print(df)

        cells_difference.append(df)

    with open('results.csv', 'a') as f:
        i = 1
        for df in cells_difference:
            f.write(f"Test {i}")
            df.to_csv(f)
            f.write("\n")
            i += 1


if __name__ == '__main__':
    main()
