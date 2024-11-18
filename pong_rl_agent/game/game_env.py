# pong_rl_agent/game/game_env.py

"""
Game environment module for Pong.
"""

import pygame
from typing import Tuple, Optional
from game.paddle import Paddle
from game.ball import Ball
import random


class GameEnv:
    """
    Manages the game environment, updates game state, and handles interactions with the RL agent.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        """
        Initializes the game environment.

        Args:
            width (int): Width of the game window.
            height (int): Height of the game window.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pong RL Agent")
        self.clock = pygame.time.Clock()
        self.running = True

        # Initialize paddles
        self.human_paddle = Paddle(
            x=50, y=(self.height - Paddle.HEIGHT) // 2, screen_height=self.height
        )

        self.ai_paddle = Paddle(
            x=self.width - 50 - Paddle.WIDTH,
            y=(self.height - Paddle.HEIGHT) // 2,
            screen_height=self.height,
            speed=5,  # Reduced speed for AI paddle
        )

        # Initialize ball
        self.ball = Ball(
            x=self.width // 2,
            y=self.height // 2,
            screen_width=self.width,
            screen_height=self.height,
        )

        self.ai_reaction_time = 200  # AI reacts every 200 milliseconds
        self.last_ai_update = pygame.time.get_ticks()

        self.max_score = 3  # Winning score
        self.game_over = False  # Flag to indicate if the game is over

        self.human_score = 0
        self.ai_score = 0
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 30)

    def reset(self) -> None:
        """
        Resets the game to the initial state after a point is scored.
        """
        self.ball.reset()
        self.human_paddle.y = (self.height - Paddle.HEIGHT) // 2
        self.human_paddle.rect.y = self.human_paddle.y
        self.ai_paddle.y = (self.height - Paddle.HEIGHT) // 2
        self.ai_paddle.rect.y = self.ai_paddle.y

    def handle_events(self) -> None:
        """
        Handles user input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def step(self) -> None:
        """
        Updates the game state based on user input and ball movement.
        """

        if self.game_over:
            return  # Skip updates if the game is over

        # Handle human paddle movement
        keys = pygame.key.get_pressed()
        dy = 0
        if keys[pygame.K_UP]:
            dy = -1
        elif keys[pygame.K_DOWN]:
            dy = 1
        self.human_paddle.move(dy)

        # Update AI paddle movement
        self.ai_paddle_move()

        # Update ball position and handle collisions
        self.ball.update(paddles=[self.human_paddle, self.ai_paddle])

        # Update ball position and handle collisions
        self.ball.update(paddles=[self.human_paddle, self.ai_paddle])

        # Check for scoring
        if self.ball.x - self.ball.RADIUS <= 0:
            # AI scores
            self.ai_score += 1
            if self.ai_score >= self.max_score:
                self.game_over = True
            else:
                self.reset()
        elif self.ball.x + self.ball.RADIUS >= self.width:
            # Human scores
            self.human_score += 1
            if self.human_score >= self.max_score:
                self.game_over = True
            else:
                self.reset()

    def ai_paddle_move(self) -> None:
        """
        AI logic with randomness to make mistakes.
        """
        error_chance = 0.2  # 20% chance to make a mistake

        if random.random() < error_chance:
            # AI makes a mistake
            move_direction = random.choice([-1, 0, 1])
        else:
            # AI tries to follow the ball
            if self.ball.y < self.ai_paddle.y + Paddle.HEIGHT / 2:
                move_direction = -1
            elif self.ball.y > self.ai_paddle.y + Paddle.HEIGHT / 2:
                move_direction = 1
            else:
                move_direction = 0

        self.ai_paddle.move(dy=move_direction)

    def render(self) -> None:
        """
        Renders the current game state to the display.
        """
        self.screen.fill((0, 0, 0))  # Clear screen with black
        self.human_paddle.draw(self.screen)
        self.ai_paddle.draw(self.screen)
        self.ball.draw(self.screen)

        # Render the scores
        human_score_surface = self.font.render(
            f"{self.human_score}", True, (255, 255, 255)
        )
        ai_score_surface = self.font.render(f"{self.ai_score}", True, (255, 255, 255))

        # Position the scores
        self.screen.blit(human_score_surface, (self.width // 4, 20))
        self.screen.blit(ai_score_surface, (self.width * 3 // 4, 20))

        pygame.display.flip()

    def run(self) -> None:
        """
        Main loop of the game environment.
        """
        while self.running:
            self.clock.tick(60)  # Limit to 60 FPS

            if not self.game_over:
                self.handle_events()
                self.step()
                self.render()
            else:
                self.display_game_over()
                self.handle_game_over_events()

        self.close()

    def display_game_over(self) -> None:
        """
        Displays the Game Over screen.
        """
        self.screen.fill((0, 0, 0))  # Clear screen with black

        # Display "Game Over" text
        game_over_text = self.font.render("Game Over", True, (255, 255, 255))
        self.screen.blit(
            game_over_text,
            (
                self.width // 2 - game_over_text.get_width() // 2,
                self.height // 2 - 60,
            ),
        )

        # Display winner
        if self.human_score >= self.max_score:
            winner_text = self.font.render("You Win!", True, (255, 255, 255))
        else:
            winner_text = self.font.render("AI Wins!", True, (255, 255, 255))
        self.screen.blit(
            winner_text,
            (
                self.width // 2 - winner_text.get_width() // 2,
                self.height // 2 - 20,
            ),
        )

        # Display instructions to restart or quit
        restart_text = self.font.render(
            "Press R to Restart or Q to Quit", True, (255, 255, 255)
        )
        self.screen.blit(
            restart_text,
            (
                self.width // 2 - restart_text.get_width() // 2,
                self.height // 2 + 20,
            ),
        )

        pygame.display.flip()

    def handle_game_over_events(self) -> None:
        """
        Handles events during the game over state.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Restart the game
                    self.restart_game()
                elif event.key == pygame.K_q:
                    # Quit the game
                    self.running = False

    def restart_game(self) -> None:
        """
        Restarts the game by resetting scores and game state.
        """
        self.human_score = 0
        self.ai_score = 0
        self.game_over = False
        self.reset()

    def close(self) -> None:
        """
        Closes the game environment.
        """
        pygame.quit()
