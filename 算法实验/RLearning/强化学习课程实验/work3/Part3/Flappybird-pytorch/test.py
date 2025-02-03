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
from dqnnetwork import NeuralNetwork
from dqnnetwork import *
import matplotlib.pyplot as plt

def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    ###
    episode_count = 0
    episode_reward = 0
    reward_list = []
    
    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        # Accumulate rewards
        episode_reward += reward.item()
        # set state to be state_1
        state = state_1
        
        # End of an episode
        if terminal:
            episode_count += 1

            # Print episode-level log
            print(f"Episode: {episode_count}, Reward: {episode_reward}")
            episode_reward = 0

if __name__ == "__main__":
    cuda_is_available = torch.cuda.is_available()
    
    
    model = torch.load(
            'pretrained_model/1120_2_target/current_target_network_825000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

    if cuda_is_available:  # put on GPU if CUDA is available
        model = model.cuda()

    test(model)