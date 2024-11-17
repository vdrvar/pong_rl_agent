# pong_rl_agent/main.py

"""
Main module to run the Pong RL Agent application.
"""

from game.game_env import GameEnv

def main() -> None:
    """
    Initializes the game environment and starts the game loop.
    """
    # Initialize game environment
    game_env = GameEnv()

    # Run the game loop
    game_env.run()

if __name__ == "__main__":
    main()
