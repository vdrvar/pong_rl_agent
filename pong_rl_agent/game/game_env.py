# pong_rl_agent/game/game_env.py

"""
Game environment module for Pong.
"""

import random

import pygame
import torch
from game.ball import Ball
from game.paddle import Paddle
from rl.agent import Agent


class GameEnv:
    """
    Manages the game environment, updates game state, and handles interactions with the RL agent.
    """

    def __init__(self, width: int = 1000, height: int = 800) -> None:
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

        self.max_score = 5  # Winning score
        self.game_over = False  # Flag to indicate if the game is over

        self.human_score = 0
        self.ai_score = 0
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 30)

        # Initialize the RL agent with state size
        state_size = 6  # As per our state representation
        self.agent = Agent(action_space_size=3, state_size=state_size)

        # Initialize epsilon for epsilon-greedy policy
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.99  # Slower decay
        self.epsilon_min = 0.01

        self.state = self.get_state()
        self.total_reward = 0  # Initialize total reward

        self.target_update_interval = 1000  # Update target network every 1000 steps
        self.steps_done = 0  # Keep track of the number of steps

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
            return

        # Store the current state
        current_state = self.state

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
        previous_ball_x = self.ball.x  # Store previous ball position
        self.ball.update(paddles=[self.human_paddle, self.ai_paddle])

        # Initialize reward
        reward = 0

        # Check if the AI paddle hit the ball
        if self.ball.rect.colliderect(self.ai_paddle.rect):
            reward += 1  # Positive reward for hitting the ball

        # Check if the ball passed the AI paddle (missed by AI)
        if previous_ball_x > self.ai_paddle.x and self.ball.x < self.ai_paddle.x:
            reward -= 1  # Negative reward for missing the ball

        # Check for scoring
        done = False
        if self.ball.x - self.ball.RADIUS <= 0:
            # AI scores
            self.ai_score += 1
            reward += 10  # Positive reward for scoring
            done = True
            if self.ai_score >= self.max_score:
                self.game_over = True
            else:
                self.reset()
        elif self.ball.x + self.ball.RADIUS >= self.width:
            # Human scores
            self.human_score += 1
            reward -= 10  # Negative reward for opponent scoring
            done = True
            if self.human_score >= self.max_score:
                self.game_over = True
            else:
                self.reset()

        # Small penalty for each step to encourage efficiency
        reward -= 0.01

        # Calculate the vertical distance penalty
        ai_paddle_center_y = self.ai_paddle.y + self.ai_paddle.HEIGHT / 2
        distance_y = abs(self.ball.y - ai_paddle_center_y)
        normalized_distance = distance_y / self.height
        distance_penalty_scale = 1.0  # Adjust this value as needed
        distance_penalty = normalized_distance * distance_penalty_scale

        # Apply the distance penalty
        reward -= distance_penalty

        # Get the next state
        next_state = self.get_state()

        # Store the experience in the agent's memory
        action = self.agent.last_action
        self.agent.store_experience((current_state, action, reward, next_state, done))

        # Update the state
        self.state = next_state

        # Update the total reward before printing
        self.total_reward += reward

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Optimize the agent's model
        self.agent.optimize_model()

        self.steps_done += 1

        if self.steps_done % self.target_update_interval == 0:
            self.agent.update_target_network()

        # Debugging statements
        print(f"Reward: {reward}, Total Reward: {self.total_reward}")
        print(f"Distance Penalty: {-distance_penalty}")
        print(f"Number of experiences collected: {len(self.agent.memory)}")
        print(f"Epsilon: {self.epsilon}")

    def ai_paddle_move_old(self) -> None:
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

    def ai_paddle_move(self) -> None:
        """
        Controls the AI paddle using the RL agent.
        """
        # Get the current state
        state = self.get_state()
        # Let the agent select an action
        action = self.agent.select_action(state, self.epsilon)
        # Map the action to dy
        dy = action - 1
        self.ai_paddle.move(dy=dy)

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

    def get_state(self) -> list:
        """
        Returns the current state of the environment as a list of normalized values.

        Returns:
            list: A list representing the normalized state.
        """
        # Normalize positions and velocities
        ball_x = self.ball.x / self.width
        ball_y = self.ball.y / self.height
        ball_x_vel = (self.ball.x_vel + self.ball.max_vel) / (2 * self.ball.max_vel)
        ball_y_vel = (self.ball.y_vel + self.ball.max_vel) / (2 * self.ball.max_vel)
        ai_paddle_y = self.ai_paddle.y / self.height
        human_paddle_y = self.human_paddle.y / self.height

        return [ball_x, ball_y, ball_x_vel, ball_y_vel, ai_paddle_y, human_paddle_y]

    def optimize_model(self) -> None:
        """
        Performs a single optimization step on the model.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences to sample

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # Compute expected Q values
        expected_state_action_values = rewards + (
            self.gamma * next_state_values * (1 - dones)
        )

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Training loss: {loss.item()}")
