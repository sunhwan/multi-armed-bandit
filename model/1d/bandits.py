from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            return 1
        else:
            return 0

class MarkovBandit(Bandit):
    def __init__(self, n, M=None, reward_state=0, nstep=100):
        assert M is None or len(M) == n
        self.n = n
        self.reward_state = reward_state
        if M is None:
            np.random.seed(int(time.time()))
            self.M = np.random.random((n, n))
            # normalize markov transition matrix
            for i in range(n):
                self.M[i, :] = self.M[i, :] / np.sum(self.M[i, :])
        else:
            self.M = M
        # transition count matrix
        self.T = np.zeros_like(self.M)

        # markov chain for giving reward
        self.markov_chain = []

        mask = [False if i == self.reward_state else True for i in range(n)]
        self.best_proba = max(self.M[mask, self.reward_state])
        self.nstep = nstep
        
    def evolve_markov_chain(self, i):
        # the player selected the i-th machine
        # evolve the transition matrix
        states = list(range(self.n))
        for _ in range(self.nstep):
            j = np.random.choice(states, p=self.M[i, :])
            self.T[i, j] += 1
            self.markov_chain.append(j)
    
    def generate_reward(self, i):
        self.evolve_markov_chain(i)
        if i == self.reward_state and self.reward_state == self.markov_chain[-1]:
            return 0
        
        if self.reward_state == self.markov_chain[-1]:
            return 1
        else:
            return 0
