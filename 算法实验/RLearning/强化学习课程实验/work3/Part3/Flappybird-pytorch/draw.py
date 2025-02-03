import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from game.flappy_bird import GameState
from dqnnetwork import NeuralNetwork, resize_and_bgr2gray, image_to_tensor


def test(model, num_games=10):
    game_state = GameState()
    total_reward = 0
    max_reward = -1
    for _ in range(num_games):
        # Initial action is to do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        episode_reward = 0

        while not terminal:
            # Get output from the neural network
            output = model(state)[0]

            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.cuda()

            # Get action
            action_index = torch.argmax(output).item()
            action[action_index] = 1

            # Get next state
            image_data_1, reward, terminal = game_state.frame_step(action)
            image_data_1 = resize_and_bgr2gray(image_data_1)
            image_data_1 = image_to_tensor(image_data_1)
            state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

            # Accumulate rewards
            episode_reward += reward
            state = state_1

        total_reward += episode_reward
        max_reward = episode_reward if episode_reward > max_reward else max_reward
    # Return max reward
    return max_reward


if __name__ == "__main__":
    cuda_is_available = torch.cuda.is_available()
    model_directory = 'pretrained_model/1119_2_original/'
    weight_files = sorted(
        [os.path.join(model_directory, f) for f in os.listdir(model_directory) if f.endswith('.pth')]
    )

    training_steps = []
    average_rewards = []

    for weight_file in weight_files:
        # Extract training step from file name (assuming format '..._XXXXXX.pth')
        training_step = int(weight_file.split('_')[-1].split('.')[0])
        training_steps.append(training_step)

        # Load the model
        model = torch.load(weight_file, map_location='cpu' if not cuda_is_available else None).eval()
        if cuda_is_available:
            model = model.cuda()

        # Test the model
        avg_reward = test(model, num_games=10)
        average_rewards.append(avg_reward)

        print(f"Weight: {weight_file}, Training Step: {training_step}, Avg Reward: {avg_reward}")

    # Plot training steps vs. average rewards
    plt.figure(figsize=(10, 6))
    plt.bar(training_steps, average_rewards, width=10, align='center', color='skyblue', edgecolor='black')  
    plt.title("original network")
    plt.xlabel("Training Steps")
    plt.ylabel("Rewards")
    plt.grid(axis='y')
    plt.show()

