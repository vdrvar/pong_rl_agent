# pong_rl_agent/rl/model.py

"""
Neural network model for the RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    """
    Deep Q-Network model for the RL agent.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes the neural network layers.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
        """
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output Q-values for each action
        return x
