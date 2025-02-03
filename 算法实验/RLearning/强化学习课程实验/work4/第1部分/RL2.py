import numpy as np
import MDP
from sympy import *

class RL2:
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

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''从随机策略中采样单个动作的程序，通过以下概率公式采样
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        本函数将被reinforce()调用来选取动作

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action

        提示：计算出概率后，可以用np.random.choice()，来进行采样
        '''
        # temporary value to ensure that the code compiles until this
        # function is coded
        action = 0
        P = np.exp(policyParams[:,state]/np.sum(exp(policyParams[:,state])))
        action = np.random.choice(len(P),p=P)
        return action



    def epsilonGreedyBandit(self, nIterations):
        '''Epsilon greedy 算法 for bandits (假设没有折扣因子).
        Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        counts = np.zeros(self.mdp.nActions) 
        reward_list = []
        s=0
        epsilon = 1/nIterations
        for t in range(nIterations):
            if np.random.rand() < epsilon:
                # 随机选择一个臂
                action = np.random.choice(self.mdp.nActions)
            else:
                # 选择当前经验平均奖励最高的臂
                action = np.argmax(empiricalMeans)
            
            # 拉取臂，观察奖励
            print(epsilon)
            reward = self.sampleReward(self.mdp.R[action,:])
            
            reward_list.append(reward)
            
            # 更新经验平均奖励
            counts[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / counts[action]
        return empiricalMeans,reward_list

    def thompsonSamplingBandit(self, prior, nIterations, k=1):
        '''Thompson sampling 算法 for Bernoulli bandits (假设没有折扣因子)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards


        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)

        提示：根据beta分布的参数，可以采用np.random.beta()进行采样
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        counts = np.zeros(self.mdp.nActions)
        reward_list = []

        for t in range(nIterations):
            # 根据 beta 分布采样每个臂的奖励
            sampled_rewards = [np.random.beta(prior[a, 0], prior[a, 1]) for a in range(self.mdp.nActions)]
            
            # 选择最大采样奖励的臂
            action = np.argmax(sampled_rewards)
            
            # 拉取臂，观察奖励
            reward = self.sampleReward(self.mdp.R[action,:])
            reward_list.append(reward)
            
            # 更新 Beta 分布的参数
            if reward == 1:
                prior[action, 0] += 1  # 更新 alpha
            else:
                prior[action, 1] += 1  # 更新 beta

            # 更新经验平均奖励
            counts[action] += 1
            empiricalMeans[action] += (reward - empiricalMeans[action]) / counts[action]

        return empiricalMeans,reward_list

    def UCBbandit(self, nIterations):
        '''Upper confidence bound 算法 for bandits (假设没有折扣因子)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs:
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        reward_list -- 用于记录每次获得的奖励(array of |nIterations| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        empiricalMeans = np.zeros(self.mdp.nActions)
        reward_list = []

        return empiricalMeans,reward_list

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        '''reinforce 算法，学习到一个随机策略，建模为：
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        上面的sampleSoftmaxPolicy()实现该方法，通过调用sampleSoftmaxPolicy(policyParams,state)来选择动作
        并且同学们需要根据上课讲述的REINFORCE算法，计算梯度，根据更新公式，完成策略参数的更新。
        其中，超参数：折扣因子gamma=0.95，学习率alpha=0.01

        Inputs:
        s0 -- 初始状态
        initialPolicyParams -- 初始策略的参数 (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- 最终策略的参数 (array of |A|x|S| entries)
        rewardList --用于记录每个episodes的累计折扣奖励 (array of |nEpisodes| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
        rewardList = []

        return [policyParams,rewardList] 