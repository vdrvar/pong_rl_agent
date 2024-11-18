# pong_rl_agent/rl/agent.py

"""
Reinforcement Learning agent module.
"""

import random
from typing import List, Tuple


class Agent:
    """
    RL agent that interacts with the environment and selects actions.
    """

    def __init__(self, action_space_size: int) -> None:
        """
        Initializes the agent.

        Args:
            action_space_size (int): The number of possible actions.
        """
        self.action_space_size = action_space_size
        self.memory = []  # Experience replay memory

    def select_action(self, state: List[float]) -> int:
        """
        Selects an action based on the current state.

        Args:
            state (List[float]): The current state of the environment.

        Returns:
            int: The action to take.
        """
        # For now, select a random action
        return random.randint(0, self.action_space_size - 1)

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
