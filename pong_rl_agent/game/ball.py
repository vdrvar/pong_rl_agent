# pong_rl_agent/game/ball.py

"""
Ball module for Pong.
"""

import random

import pygame
from game.paddle import Paddle


class Ball:
    """
    Represents the ball in the Pong game.
    """

    RADIUS = 7
    COLOR = (255, 255, 255)  # White color

    def __init__(self, x: int, y: int, screen_width: int, screen_height: int) -> None:
        """
        Initializes the ball at a specific position.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.
            screen_width (int): Width of the game screen for boundary checks.
            screen_height (int): Height of the game screen for boundary checks.
        """
        self.x = x
        self.y = y
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Ball's rectangle for collision detection
        self.rect = pygame.Rect(
            self.x - self.RADIUS, self.y - self.RADIUS, self.RADIUS * 2, self.RADIUS * 2
        )

        # Velocity of the ball (reduced speed)
        self.x_vel = random.choice([-4, 4])  # Changed from [-5, 5] to [-3, 3]
        self.y_vel = random.choice([-4, 4])  # Changed from [-5, 5] to [-3, 3]

        self.max_vel = 3  # Maximum velocity for normalization

    def update(self, paddles: list) -> None:
        """
        Updates the ball's position based on its velocity and handles collisions.

        Args:
            paddles (list): List of Paddle instances to check for collisions.
        """
        self.x += self.x_vel
        self.y += self.y_vel

        # Update the rectangle position
        self.rect.x = self.x - self.RADIUS
        self.rect.y = self.y - self.RADIUS

        # Bounce off the top and bottom walls
        if self.y - self.RADIUS <= 0 or self.y + self.RADIUS >= self.screen_height:
            self.y_vel *= -1

        # Check for collision with paddles
        self.handle_collision(paddles)

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the ball on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw the ball on.
        """
        pygame.draw.circle(surface, self.COLOR, (int(self.x), int(self.y)), self.RADIUS)

    def reset(self) -> None:
        """
        Resets the ball to the center of the screen with a new random velocity.
        """
        self.x = self.screen_width // 2
        self.y = self.screen_height // 2
        self.x_vel = random.choice([-3.5, 3.5])  # Changed from [-5, 5] to [-3, 3]
        self.y_vel = random.choice([-3.5, 3.5])  # Changed from [-5, 5] to [-3, 3]

    def handle_collision(self, paddles: list) -> None:
        for paddle in paddles:
            if self.rect.colliderect(paddle.rect):
                # Calculate collision point
                paddle_center = paddle.y + Paddle.HEIGHT / 2
                collision_diff = self.y - paddle_center
                collision_ratio = collision_diff / (Paddle.HEIGHT / 2)
                angle = collision_ratio * (5 * (3.14 / 12))  # Max angle of 75 degrees

                # Adjust velocities
                speed = (self.x_vel**2 + self.y_vel**2) ** 0.5
                self.x_vel = -self.x_vel
                self.y_vel = speed * collision_ratio

                # Prevent sticking
                if self.x_vel > 0:
                    self.x = paddle.rect.right + self.RADIUS
                else:
                    self.x = paddle.rect.left - self.RADIUS
                break
