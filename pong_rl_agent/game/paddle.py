# pong_rl_agent/game/paddle.py

"""
Paddle module for Pong.
"""

import pygame

class Paddle:
    """
    Represents a paddle in the Pong game.
    """

    WIDTH = 10
    HEIGHT = 100
    COLOR = (255, 255, 255)  # White color

    def __init__(self, x: int, y: int, screen_height: int, speed: int = 7) -> None:
        """
        Initializes the paddle at a specific position.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.
            screen_height (int): The height of the game screen for boundary checks.
            speed (int): The speed at which the paddle moves.
        """
        self.x = x
        self.y = y
        self.screen_height = screen_height
        self.rect = pygame.Rect(self.x, self.y, self.WIDTH, self.HEIGHT)
        self.velocity = 0
        self.speed = speed  # Use the speed parameter

    def move(self, dy: int) -> None:
        """
        Moves the paddle vertically.

        Args:
            dy (int): The amount to move in the y-direction.
        """
        self.velocity = dy * self.speed
        self.y += self.velocity

        # Keep paddle within the screen boundaries
        if self.y < 0:
            self.y = 0
        elif self.y + self.HEIGHT > self.screen_height:
            self.y = self.screen_height - self.HEIGHT

        # Update the rectangle position
        self.rect.y = self.y

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the paddle on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw the paddle on.
        """
        pygame.draw.rect(surface, self.COLOR, self.rect)
