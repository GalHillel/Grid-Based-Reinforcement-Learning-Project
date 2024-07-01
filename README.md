
# Grid-Based Reinforcement Learning Project

## Overview
This project involves implementing and comparing various reinforcement learning techniques for solving grid-based environments. The goal is to find the optimal policy using different methods such as Value Iteration, Model-Based RL, and Model-Free RL.

## Directory Structure
The project is organized as follows:
```
root
├── data/                  # Directory for storing grid data and results
│   ├── grid.json          # JSON file containing grid information
│   ├── results/           # Directory for storing result files
├── scripts/               # Directory for Python scripts
│   ├── generate_grid.py   # Script for generating grid data
│   ├── value_iteration.py # Implementation of Value Iteration algorithm
│   ├── model_based_rl.py  # Implementation of Model-Based RL algorithm
│   ├── model_free_rl.py   # Implementation of Model-Free RL algorithm
├── notebooks/             # Directory for Jupyter notebooks
│   ├── Grid_Environment.ipynb    # Notebook for grid environment setup and exploration
│   ├── Value_Iteration.ipynb     # Notebook for Value Iteration algorithm
│   ├── Model_Based_RL.ipynb      # Notebook for Model-Based RL algorithm
│   ├── Model_Free_RL.ipynb       # Notebook for Model-Free RL algorithm
├── README.md              # This README file
└── requirements.txt       # List of Python packages required
```

## Installation
To set up the environment and install necessary packages, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grid-rl-project.git
   cd grid-rl-project
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Generate Grid Data
To generate grid data, run:
```bash
python scripts/generate_grid.py
```
This will create a `grid.json` file in the `data/` directory containing information about the grid environment.

### Run Value Iteration
To perform Value Iteration, run:
```bash
python scripts/value_iteration.py
```
Results will be saved in the `data/results/` directory.

### Run Model-Based RL
To perform Model-Based Reinforcement Learning, run:
```bash
python scripts/model_based_rl.py
```
Results will be saved in the `data/results/` directory.

### Run Model-Free RL
To perform Model-Free Reinforcement Learning, run:
```bash
python scripts/model_free_rl.py
```
Results will be saved in the `data/results/` directory.

## Jupyter Notebooks
For an interactive exploration of the grid environment and the algorithms, use the provided Jupyter notebooks in the `notebooks/` directory. You can start Jupyter Notebook by running:
```bash
jupyter notebook
```

## Project Details
### Grid Environment
The grid environment is represented as a 2D grid with different states, including rewards and obstacles. The agent's goal is to navigate through the grid to maximize the cumulative reward.

### Value Iteration
Value Iteration is a dynamic programming algorithm used to find the optimal policy by iteratively updating the value of each state based on the expected utility of future states.

### Model-Based RL
Model-Based Reinforcement Learning involves building a model of the environment's dynamics and using it to plan the optimal policy. This method estimates the transition probabilities and rewards from the agent's experiences.

### Model-Free RL
Model-Free Reinforcement Learning methods, such as Q-Learning, directly learn the optimal policy from the agent's experiences without explicitly modeling the environment. This approach updates the value of actions based on the observed rewards and transitions.

## Results
The results of each algorithm, including the optimal policies and value functions, are saved in the `data/results/` directory. These results can be visualized and analyzed to compare the performance of different reinforcement learning techniques.
