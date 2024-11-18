# pong_rl_agent/rl/agent.py

"""
Reinforcement Learning agent module.
"""

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from rl.model import DQNModel


class Agent:
    """
    RL agent that interacts with the environment and selects actions.
    """

    def __init__(self, action_space_size: int, state_size: int) -> None:
        """
        Initializes the agent.

        Args:
            action_space_size (int): The number of possible actions.
            state_size (int): The size of the state representation.
        """
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.memory = []  # Experience replay memory
        self.last_action = None  # Initialize last_action

        # Initialize the neural network model
        self.policy_net = DQNModel(input_size=state_size, output_size=action_space_size)
        self.target_net = DQNModel(input_size=state_size, output_size=action_space_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor

    def select_action(self, state: List[float], epsilon: float) -> int:
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (List[float]): The current state of the environment.
            epsilon (float): The probability of selecting a random action.

        Returns:
            int: The action to take.
        """
        if random.random() < epsilon:
            # Explore: select a random action
            action = random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: select the action with the highest predicted Q-value
            state_tensor = torch.tensor([state], dtype=torch.float32)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                print(f"Q-values: {q_values}")
                action = q_values.argmax().item()
        self.last_action = action

        print(f"Selected action: {action}")

        return action

    def optimize_model(self) -> None:
        """
        Performs a single optimization step on the model.
        """
        pass  # To be implemented

    def save_model(self, path: str) -> None:
        """
        Saves the model state to a file.

        Args:
            path (str): The file path to save the model.
        """
        pass  # To be implemented

    def load_model(self, path: str) -> None:
        """
        Loads the model state from a file.

        Args:
            path (str): The file path to load the model from.
        """
        pass  # To be implemented

    def store_experience(self, experience: Tuple) -> None:
        """
        Stores an experience in memory.

        Args:
            experience (Tuple): A tuple of (state, action, reward, next_state, done).
        """
        self.memory.append(experience)

    def update_target_network(self) -> None:
        """
        Updates the target network with the policy network's weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
