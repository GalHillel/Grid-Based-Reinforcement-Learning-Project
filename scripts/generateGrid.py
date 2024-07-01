import os
import json

# Define the test cases with different configurations
test_cases = [
    {
        "w": 4,
        "h": 3,
        "L": [(1, 1, 0), (3, 2, 1), (3, 1, -1)],
        "p": 0.8,
        "r": [-0.04, 0.04, -1],
    },
    {
        "w": 12,
        "h": 4,
        "L": [
            (1, 0, -100),
            (2, 0, -100),
            (3, 0, -100),
            (4, 0, -100),
            (5, 0, -100),
            (6, 0, -100),
            (7, 0, -100),
            (8, 0, -100),
            (9, 0, -100),
            (10, 0, -100),
            (11, 0, 1),
        ],
        "p": 1,
        "r": [-1],
    },
    {
        "w": 12,
        "h": 6,
        "L": [
            (1, 0, -100),
            (2, 0, -100),
            (3, 0, -100),
            (4, 0, -100),
            (5, 0, -100),
            (6, 0, -100),
            (7, 0, -100),
            (8, 0, -100),
            (9, 0, -100),
            (10, 0, -100),
            (11, 0, 1),
        ],
        "p": 0.9,
        "r": [-1],
    },
    {
        "w": 5,
        "h": 5,
        "L": [(4, 0, -10), (0, 4, -10), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.5],
    },
    {
        "w": 5,
        "h": 5,
        "L": [(2, 2, -2), (4, 4, -1), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.25],
    },
    {
        "w": 7,
        "h": 7,
        "L": [(1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)],
        "p": 0.8,
        "r": [-0.5],
    },
    {
        "w": 7,
        "h": 7,
        "L": [(3, 1, 0), (3, 5, 0), (1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)],
        "p": 0.8,
        "r": [-0.25],
    },
    {
        "w": 6,
        "h": 6,
        "L": [(2, 2, -2), (4, 4, -1), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.1, -0.3, -0.5],
    },
    {
        "w": 8,
        "h": 8,
        "L": [(1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)],
        "p": 0.8,
        "r": [-0.2, -0.4, -0.6],
    },
    {
        "w": 10,
        "h": 5,
        "L": [(4, 0, -10), (0, 4, -10), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.2, -0.4, -0.6],
    },
    {
        "w": 8,
        "h": 6,
        "L": [(1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)],
        "p": 0.8,
        "r": [-0.1, -0.3, -0.5],
    },
    {
        "w": 6,
        "h": 8,
        "L": [(2, 2, -2), (4, 4, -1), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.2, -0.4, -0.6],
    },
    {
        "w": 7,
        "h": 10,
        "L": [(1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)],
        "p": 0.8,
        "r": [-0.1, -0.3, -0.5],
    },
    {
        "w": 5,
        "h": 12,
        "L": [(4, 0, -10), (0, 4, -10), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.2, -0.4, -0.6],
    },
    {
        "w": 6,
        "h": 6,
        "L": [(2, 2, -2), (4, 4, -1), (1, 1, 1), (3, 3, 2)],
        "p": 0.9,
        "r": [-0.2, -0.3, -0.4],
    },
]


def generate_grid(w, h, L, p, r):
    grid = [[0 for _ in range(w)] for _ in range(h)]
    for x, y, reward in L:
        grid[y][x] = reward
    return grid


if __name__ == "__main__":
    # Print a debug statement to verify script execution
    print("Script started.")

    # Create data/tests directory if it doesn't exist
    if not os.path.exists("../data/tests"):
        os.makedirs("../data/tests")
        print("Created directory: ../data/tests")

    # Iterate through test cases
    for idx, case in enumerate(test_cases):
        w, h = case["w"], case["h"]
        L, p, rewards = case["L"], case["p"], case["r"]
        for reward in rewards:
            grid = generate_grid(w, h, L, p, reward)
            filename = f"data/tests/grid_t{idx + 1}_r{reward}.json"
            print(f"Generating file: {filename}")
            with open(filename, "w") as f:
                json.dump(grid, f)

    print("Script finished.")
