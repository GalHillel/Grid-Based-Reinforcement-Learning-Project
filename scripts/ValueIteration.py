from collections import defaultdict


class ValueIteration:
    """
    Value Iteration algorithm for solving Markov Decision Processes (MDPs).
    """

    def getQValueFromValues(self, mdp, state, action, values, discount):
        """
        Computes the Q-value for a given state-action pair using current value estimates.

        Args:
            mdp (object): The Markov Decision Process (MDP) instance.
            state (object): The current state in the MDP.
            action (object): The action taken in the current state.
            values (defaultdict): Dictionary mapping states to their value estimates.
            discount (float): Discount factor for future rewards.

        Returns:
            float: The computed Q-value for the state-action pair.
        """
        q_value = 0
        for landing_state, prob in mdp.getTransitionStatesAndProbs(state, action):
            reward = mdp.getReward(state, action, landing_state)
            q_value += prob * (reward + discount * values[landing_state])
        return q_value

    def valueIteration(self, mdp, discount, iterations=100):
        """
        Performs the value iteration algorithm to compute optimal state values.

        Args:
            mdp (object): The Markov Decision Process (MDP) instance.
            discount (float): Discount factor for future rewards.
            iterations (int, optional): Number of iterations for the algorithm (default is 100).

        Returns:
            defaultdict: Dictionary mapping states to their optimal value estimates.
        """
        values = defaultdict(lambda: 0)
        for _ in range(iterations):
            next_values = defaultdict(lambda: 0)
            for state in mdp.getStates():
                if not mdp.isTerminal(state):
                    max_q_value = float("-inf")
                    for action in mdp.getLegalActions(state):
                        q_value = self.getQValueFromValues(
                            mdp, state, action, values, discount
                        )
                        max_q_value = max(max_q_value, q_value)
                    next_values[state] = max_q_value
            values = next_values
        return values

    def getQValues(self, mdp, values, discount):
        """
        Computes Q-values for all state-action pairs using current value estimates.

        Args:
            mdp (object): The Markov Decision Process (MDP) instance.
            values (defaultdict): Dictionary mapping states to their value estimates.
            discount (float): Discount factor for future rewards.

        Returns:
            dict: Dictionary mapping (state, action) pairs to their computed Q-values.
        """
        q_values = {}
        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                for action in mdp.getLegalActions(state):
                    q_values[state, action] = self.getQValueFromValues(
                        mdp, state, action, values, discount
                    )
        return q_values

    def getPolicy(self, mdp, values, discount):
        """
        Extracts the optimal policy based on computed values and Q-values.

        Args:
            mdp (object): The Markov Decision Process (MDP) instance.
            values (defaultdict): Dictionary mapping states to their value estimates.
            discount (float): Discount factor for future rewards.

        Returns:
            dict: Dictionary mapping states to their optimal actions according to the policy.
        """
        policy = {}
        for state in mdp.getStates():
            if not mdp.isTerminal(state):
                max_q_value = -float("inf")
                best_action = None
                for action in mdp.getLegalActions(state):
                    q_value = self.getQValueFromValues(
                        mdp, state, action, values, discount
                    )
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = action
                policy[state] = best_action
        return policy
