# pong_rl_agent/rl/replay_buffer.py

"""
Replay buffer for experience replay.
"""

import random
from collections import deque
from typing import Tuple, Any

class ReplayBuffer:
    """
    Stores and samples experiences for training the agent.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args: Any) -> None:
        """
        Adds a new experience to the buffer.

        Args:
            *args: Experience tuple (state, action, reward, next_state, done).
        """
        self.memory.append(args)

    def sample(self, batch_size: int) -> Tuple:
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple: Batch of sampled experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Returns the current size of internal memory.

        Returns:
            int: Number of stored experiences.
        """
        return len(self.memory)
