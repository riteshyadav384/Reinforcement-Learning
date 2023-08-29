import numpy as np

import util
from agent import Agent

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        if state in self.Q.keys():
            best_action = max(self.Q[state], key=self.Q[state].get)
            return self.Q[state][best_action]
        else:
            return 0

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        if state in self.Q.keys():
            return self.Q[state][action]
        else:
            return 0

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        if state in self.Q.keys():
            return max(self.Q[state], key=self.Q[state].get)
        else:
            return self.getRandomAction(state)

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            return np.random.choice(all_actions)
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that the agent balance exploration and exploitation as appropriate. """
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        terminal_state = (-1,-1)
        if state in self.Q.keys():
            if action in self.Q[state].keys():
                if nextState in self.Q.keys():
                    best_action = max(self.Q[nextState], key=self.Q[nextState].get)
                    self.Q[state][action] = (1 - self.learningRate) * self.Q[state][action] + self.learningRate * (
                                reward + self.discount * self.Q[nextState][best_action])
                else:
                    if nextState == terminal_state:
                        self.Q[state][action] = (1 - self.learningRate) * self.Q[state][action] + self.learningRate * (reward)
                    else:
                        self.Q[nextState] = {}
                        for actions in self.actionFunction(nextState):
                            self.Q[nextState][actions]=0.0
            else:
                self.Q[state][action] = 0.0

        else:
            self.Q[state] = {}
            for actions in self.actionFunction(state):
                self.Q[state][actions] = 0.0