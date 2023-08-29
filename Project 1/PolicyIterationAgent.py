import numpy as np
from agent import Agent

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        The policy iteration agent take an mdp on
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

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}
        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                for s in states:
                    a = self.pi[s]
                    if a is None:
                        pass
                    else:
                        successors = self.mdp.getTransitionStatesAndProbs(s, a)
                        for nextState, prob in successors:
                            reward = mdp.getReward(s, a, None)
                            if s in newV.keys():
                                newV[s] = newV[s] + prob * (reward + discount * self.V[nextState])
                            else:
                                newV[s] = prob * (reward + discount * self.V[nextState])

                # update value estimate
                self.V.update(newV)


            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    action_id = 0
                    r = np.zeros(len(actions))
                    for action in actions:
                        successors= self.mdp.getTransitionStatesAndProbs(s, action)
                        reward = mdp.getReward(s, action, None)
                        for nextState,prob in successors:
                            r[action_id] += prob*(reward + discount * self.V[nextState])
                        action_id = action_id + 1
                    self.pi[s] = actions[np.argmax(r)]

                    if old_action != self.pi[s]:
                        policy_stable = False

            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        # Look up the value of the state (after the policy converged).
        return self.V[state]

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note: policy iteration does not
        necessarily create this quantity and hence may have
        to derive it on the fly.
        """
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
        return self.pi[state]

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        pass
