class GridWorld:
    """
    Grid Environment for MDP Value Iteration.

    Attributes:
        EXIT (tuple): Represents an exit coordinate indicating the end of the game.
        NORTH (tuple): Movement direction for north.
        EAST (tuple): Movement direction for east.
        SOUTH (tuple): Movement direction for south.
        WEST (tuple): Movement direction for west.
        DIRCS (list): List of all movement directions.
        index (dict): Mapping of directions to indices.
        GAMEOVER (tuple): Represents a coordinate indicating the end of the game.
    """

    EXIT = (float("inf"), float("inf"))
    NORTH = (-1, 0)
    EAST = (0, +1)
    SOUTH = (+1, 0)
    WEST = (0, -1)
    DIRCS = [NORTH, EAST, SOUTH, WEST]
    index = {NORTH: 0, EAST: 1, SOUTH: 2, WEST: 3}
    GAMEOVER = (-1, -1)

    def __init__(self, shape, prob, walls, terminals):
        """
        Initializes the grid environment.

        Args:
            shape (tuple): Shape of the grid (rows, columns).
            prob (float): Probability of moving in the intended direction.
            walls (set): Set of wall coordinates in the grid.
            terminals (dict): Dictionary of terminal states and their rewards.
        """
        self.rows, self.cols = shape
        accident = (1 - prob) / 2
        self.turns = {-1: accident, 0: prob, +1: accident}
        self.walls = set(walls)
        self.terms = terminals

    def getStates(self):
        """
        Returns a list of all valid states in the grid.

        Returns:
            list: List of valid states (coordinates).
        """
        return [
            (i, j)
            for i in range(self.rows)
            for j in range(self.cols)
            if (i, j) not in self.walls
        ]

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns possible transition states and their probabilities for a given state and action.

        Args:
            state (tuple): Current state coordinate.
            action (tuple): Action direction tuple.

        Returns:
            list: List of (next_state, probability) tuples.
        """
        if state in self.terms:
            return [(GridWorld.GAMEOVER, 1.0)]

        result = []
        for turn in self.turns:
            dirc = GridWorld.DIRCS[
                (GridWorld.index[action] + turn) % len(GridWorld.DIRCS)
            ]
            row = state[0] + dirc[0]
            col = state[1] + dirc[1]
            landing = (
                row if 0 <= row < self.rows else state[0],
                col if 0 <= col < self.cols else state[1],
            )
            if landing in self.walls:
                landing = state
            prob = self.turns[turn]
            result.append((landing, prob))
        return result

    def getReward(self, state, action, nextState):
        """
        Returns the reward for transitioning from a state to the next state with an action.

        Args:
            state (tuple): Current state coordinate.
            action (tuple): Action direction tuple.
            nextState (tuple): Next state coordinate.

        Returns:
            float: Reward value.
        """
        if state in self.terms:
            return self.terms[state]
        else:
            return 0

    def isTerminal(self, state):
        """
        Checks if a given state is a terminal state.

        Args:
            state (tuple): State coordinate.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return state == GridWorld.GAMEOVER

    def getLegalActions(self, state):
        """
        Returns legal actions available from a given state.

        Args:
            state (tuple): Current state coordinate.

        Returns:
            list: List of legal action direction tuples.
        """
        if state in self.terms:
            return [GridWorld.EXIT]
        else:
            return GridWorld.DIRCS

    def printValues(self, values):
        """
        Prints the current values of states in a grid format.

        Args:
            values (dict): Dictionary mapping state coordinates to values.
        """
        output = str()
        divide = "\n" + "----------- " * self.cols + "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                output += "   %+.2f   |" % values[(i, j)]
            output += divide
        print(output)

    def printPolicy(self, policy):
        """
        Prints the current policy actions in a grid format.

        Args:
            policy (dict): Dictionary mapping state coordinates to action tuples.
        """
        actmap = {
            GridWorld.NORTH: "^",
            GridWorld.EAST: ">",
            GridWorld.SOUTH: "v",
            GridWorld.WEST: "<",
        }

        output = str()
        divide = "\n" + "------- " * self.cols + "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                try:
                    output += "   %s   |" % actmap[policy[(i, j)]]
                except KeyError:
                    if (i, j) in self.terms:
                        if self.terms[(i, j)] > 0:
                            output += "   $   |"
                        else:
                            output += "   !   |"
                    else:
                        output += "   W   |"

            output += divide
        print(output)


class GridWorldAdditive(GridWorld):
    """
    Additive Grid Environment for MDP Value Iteration.

    Inherits from GridWorld and modifies reward calculation.

    Attributes:
        reward (float): Constant reward value for transitions.

    Methods:
        __init__(self, shape, prob, walls, terminals, reward):
            Initializes the additive grid environment.

        getReward(self, state, action, nextState):
            Returns the reward for transitioning from a state to the next state with an action, considering the additive reward.
    """

    def __init__(self, shape, prob, walls, terminals, reward):
        """
        Initializes the additive grid environment.

        Args:
            shape (tuple): Shape of the grid (rows, columns).
            prob (float): Probability of moving in the intended direction.
            walls (set): Set of wall coordinates in the grid.
            terminals (dict): Dictionary of terminal states and their rewards.
            reward (float): Constant reward value for transitions.
        """
        super(GridWorldAdditive, self).__init__(shape, prob, walls, terminals)
        self.reward = reward

    def getReward(self, state, action, nextState):
        """
        Returns the reward for transitioning from a state to the next state with an action, considering the additive reward.

        Args:
            state (tuple): Current state coordinate.
            action (tuple): Action direction tuple.
            nextState (tuple): Next state coordinate.

        Returns:
            float: Reward value.
        """
        if state in self.terms:
            return self.terms[state]
        else:
            return self.reward
