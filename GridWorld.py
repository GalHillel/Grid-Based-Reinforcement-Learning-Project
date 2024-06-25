'''
 c 0 1 2 3
r  _ _ _ _
0 |  _   $|
1 | |_|  !|
2 |_ _ _ _|
'''


# Grid Environment for MDP Value Iteration
class GridWorld:
    EXIT = (float("inf"), float("inf"))
    NORTH = (-1, 0)
    EAST = (0, +1)
    SOUTH = (+1, 0)
    WEST = (0, -1)
    DIRCS = [NORTH, EAST, SOUTH, WEST]
    index = {NORTH: 0, EAST: 1, SOUTH: 2, WEST: 3}
    GAMEOVER = (-1, -1)

    def __init__(self, shape, prob, walls, terminals):
        self.rows, self.cols = shape
        accident = (1 - prob) / 2
        self.turns = {-1: accident, 0: prob, +1: accident}
        self.walls = set(walls)
        self.terms = terminals

    def getStates(self):
        return [(i, j) for i in range(self.rows)
                for j in range(self.cols) if (i, j) not in self.walls]

    def getTransitionStatesAndProbs(self, state, action):
        if state in self.terms:
            return [(GridWorld.GAMEOVER, 1.0)]

        result = []
        for turn in self.turns:
            dirc = GridWorld.DIRCS[(GridWorld.index[action] + turn) %
                                   len(GridWorld.DIRCS)]
            row = state[0] + dirc[0]
            col = state[1] + dirc[1]
            landing = (row if 0 <= row < self.rows else state[0],
                       col if 0 <= col < self.cols else state[1])
            if landing in self.walls:
                landing = state
            prob = self.turns[turn]
            result.append((landing, prob))
        return result

    def getReward(self, state, action, nextState):
        if state in self.terms:
            return self.terms[state]
        else:
            return 0

    def isTerminal(self, state):
        return state == GridWorld.GAMEOVER

    def getLegalActions(self, state):
        if state in self.terms:
            return [GridWorld.EXIT]
        else:
            return GridWorld.DIRCS

    def printValues(self, values):
        output = str()
        divide = "\n" + "----------- " * self.cols + "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                output += "   %+.2f   |" % values[(i, j)]
            output += divide
        print(output)

    def printPolicy(self, policy):
        actmap = {GridWorld.NORTH: '^', GridWorld.EAST: '>',
                  GridWorld.SOUTH: 'v', GridWorld.WEST: '<'}

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


# Additive Grid Environment for MDP Value Iteration
class GridWorldAdditive(GridWorld):
    def __init__(self, shape, prob, walls, terminals, reward):
        super(GridWorldAdditive, self).__init__(shape, prob, walls, terminals)
        self.reward = reward

    def getReward(self, state, action, nextState):
        if state in self.terms:
            return self.terms[state]
        else:
            return self.reward
