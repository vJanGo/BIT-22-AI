import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState
import matplotlib.pyplot as plt

from dqnnetwork import NeuralNetwork
from dqnnetwork import *
import csv

def train(main_network, target_network,start):
    # define Adam optimizer
    optimizer = optim.Adam(main_network.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([main_network.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = main_network.initial_epsilon
    iteration = 0
    ###
    episode_count = 0
    episode_reward = 0
    reward_list = []
    # 创建CSV文件并写入表头
    csv_file = "train_reward/target/target_reward_per_episode.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward"])  # 写入表头

    epsilon_decrements = np.linspace(main_network.initial_epsilon, main_network.final_epsilon, main_network.number_of_iterations)

    # main infinite loop
    while iteration < main_network.number_of_iterations:
        # get output from the neural network
        output = main_network(state)[0]

        # initialize action
        action = torch.zeros([main_network.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(main_network.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # Accumulate rewards
        episode_reward += reward.item()
        
        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > main_network.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), main_network.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state from target network
        output_1_batch = target_network(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + main_network.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(main_network(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        ### 每 100 次迭代同步目标网络参数
        if iteration % 100 == 0:
            target_network.load_state_dict(main_network.state_dict())

        if iteration % 25000 == 0:
            torch.save(main_network, "pretrained_model/1120_2_target/current_target_network_" + str(iteration) + ".pth")
            

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

        # End of an episode
        if terminal:
            episode_count += 1
            reward_list.append(episode_reward)
            # 将当前episode和reward写入CSV文件
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([episode_count, episode_reward])  # 写入数据

            episode_reward = 0

            # Print episode-level log
            print(f"Episode: {episode_count}, Iteration: {iteration}, Reward: {reward_list[-1]}")

        
        # Generate reward-episode plot after 6000 episodes
        # if episode_count == 6000:
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(reward_list, label="Episode Reward")
        #     plt.xlabel("Episode")
        #     plt.ylabel("Total Reward")
        #     plt.title("Reward per Episode")
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig("reward_episode_plot.png")
        #     print("Reward-Episode plot saved as 'reward_episode_plot.png'")
        #     break
        
if __name__ == "__main__":
    cuda_is_available = torch.cuda.is_available()
    
    if not os.path.exists('pretrained_model/'):
        os.mkdir('pretrained_model/')

    main_network = NeuralNetwork()
    target_network = NeuralNetwork()
    if cuda_is_available:  # put on GPU if CUDA is available
        main_networkl = main_network.cuda()
        target_network = target_network.cuda()
    main_network.apply(init_weights)
    target_network.load_state_dict(main_network.state_dict())
    start = time.time()

    train(main_network, target_network,start)