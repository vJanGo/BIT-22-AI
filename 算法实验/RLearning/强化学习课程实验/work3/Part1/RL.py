import math

import numpy as np
import MDP

class RL:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class
        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''
        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)
        Inputs:
        state -- current state
        action -- action to be executed
        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''
        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def boltzmann_action_selection(self, Q, state, temperature):
        '''Boltzmann exploration strategy to choose an action based on Q-values and temperature'''
        q_values = Q[:, state]
        exp_q = np.exp(q_values / temperature) # 计算选取到每个action的概率
        probabilities = exp_q / np.sum(exp_q) # 按概率选取
        return np.random.choice(len(q_values), p=probabilities)

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon=0, temperature=0):
        '''
        Q-Learning algorithm with epsilon-greedy and Boltzmann exploration.
        Inputs:
        s0 -- initial state
        initialQ -- initial Q-function (|A|x|S| array)
        nEpisodes -- number of episodes
        nSteps -- number of steps per episode
        epsilon -- probability of selecting a random action
        temperature -- parameter for Boltzmann exploration
        Outputs: 
        Q -- final Q-function (|A|x|S| array)
        policy -- final policy
        rewardList -- cumulative reward for each episode
        '''
        Q = np.copy(initialQ)
        rewardList = []
        nActions, nStates = Q.shape

        for episode in range(nEpisodes):
            state = s0
            cumulativeReward = 0
            for step in range(nSteps):
                # Action selection (epsilon-greedy with Boltzmann exploration)
                if np.random.rand() < epsilon:
                    action = np.random.choice(nActions)
                elif temperature > 0:
                    action = self.boltzmann_action_selection(Q, state, temperature)
                else:
                    action = np.argmax(Q[:, state])

                # Sample reward and next state
                reward, nextState = self.sampleRewardAndNextState(state, action)

                # Q-value update
                # bestNextAction = np.argmax(Q[:, nextState])
                Q[action, state] += self.mdp.discount * (reward + self.mdp.discount * max(Q[:,nextState]) - Q[action, state])

                # Accumulate reward for this episode
                cumulativeReward += reward

                # Move to next state
                state = nextState

            rewardList.append(cumulativeReward)

        # Derive policy from Q values
        policy = np.argmax(Q, axis=0)

        return [Q, policy, rewardList]
