from agent import Agent
import numpy as np

class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        The value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)

        # Policy initialization
        self.V = {}
        for s in states:
            self.V[s] = 0

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)

                if len(actions)<1:
                    pass
                else:
                    r = {}
                    for action in actions:
                        reward = mdp.getReward(s, action, None)
                        successors = self.mdp.getTransitionStatesAndProbs(s, action)
                        for nextState, prob in successors:
                            if action in r.keys():
                                r[action] = r[action] + prob * (reward + discount * self.V[nextState])
                            else:
                                r[action] = prob * (reward + discount * self.V[nextState])

                    newV[s] = r[max(r, key=r.get)]

            # Update value function with new estimate
            self.V.update(newV)


    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        return self.V[state]

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  """
        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        reward = self.mdp.getReward(state, action, None)
        Q = 0
        for nextState, prob in successors:
            Q = Q + prob * (reward + self.discount * self.V[nextState])
        return Q

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:
            Q = {}
            for action in actions:
                Q[state,action] = self.getQValue(state,action)
            state_action_pair = max(Q, key=Q.get)
            return state_action_pair[1]

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        pass
