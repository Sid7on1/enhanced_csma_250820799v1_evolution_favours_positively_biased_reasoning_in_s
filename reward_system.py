import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity, calculate_flow

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.

    This class is responsible for calculating rewards based on the agent's actions
    and the environment's state. It uses the velocity-threshold and Flow Theory
    algorithms from the research paper to calculate rewards.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system.

        Args:
            config: Configuration object containing settings for the reward system.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict) -> float:
        """
        Calculate the reward for the given state, action, and next state.

        Args:
            state: Current state of the environment.
            action: Action taken by the agent.
            next_state: Next state of the environment.

        Returns:
            The calculated reward.
        """
        try:
            # Calculate velocity
            velocity = calculate_velocity(state, action, next_state)

            # Calculate flow
            flow = calculate_flow(state, action, next_state)

            # Calculate reward using Flow Theory
            reward = self.reward_model.calculate_reward(velocity, flow)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to make it more suitable for the agent's learning process.

        Args:
            reward: Reward to be shaped.

        Returns:
            The shaped reward.
        """
        try:
            # Apply reward shaping using the velocity-threshold algorithm
            shaped_reward = self.reward_model.shape_reward(reward)

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class RewardModel:
    """
    Reward model used by the reward system.

    This class is responsible for calculating rewards based on the velocity and flow
    values calculated by the reward system.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model.

        Args:
            config: Configuration object containing settings for the reward model.
        """
        self.config = config

    def calculate_reward(self, velocity: float, flow: float) -> float:
        """
        Calculate the reward using Flow Theory.

        Args:
            velocity: Velocity value.
            flow: Flow value.

        Returns:
            The calculated reward.
        """
        try:
            # Calculate reward using Flow Theory
            reward = (velocity + flow) / 2

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to make it more suitable for the agent's learning process.

        Args:
            reward: Reward to be shaped.

        Returns:
            The shaped reward.
        """
        try:
            # Apply reward shaping using the velocity-threshold algorithm
            shaped_reward = reward + self.config.reward_shaping_threshold

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class Config:
    """
    Configuration object for the reward system.

    This class contains settings for the reward system, such as the reward shaping
    threshold.
    """

    def __init__(self):
        """
        Initialize the configuration object.
        """
        self.reward_shaping_threshold = 0.1

class RewardSystemError(Exception):
    """
    Exception raised by the reward system.

    This exception is raised when an error occurs during reward calculation or
    shaping.
    """

    def __init__(self, message: str):
        """
        Initialize the exception.

        Args:
            message: Error message.
        """
        self.message = message

def calculate_velocity(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the velocity value.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The calculated velocity value.
    """
    try:
        # Calculate velocity using the formula from the research paper
        velocity = np.sqrt((next_state["x"] - state["x"]) ** 2 + (next_state["y"] - state["y"]) ** 2)

        return velocity

    except RewardSystemError as e:
        logger.error(f"Error calculating velocity: {e}")
        return 0.0

def calculate_flow(state: Dict, action: Dict, next_state: Dict) -> float:
    """
    Calculate the flow value.

    Args:
        state: Current state of the environment.
        action: Action taken by the agent.
        next_state: Next state of the environment.

    Returns:
        The calculated flow value.
    """
    try:
        # Calculate flow using the formula from the research paper
        flow = np.sqrt((next_state["x"] - state["x"]) ** 2 + (next_state["y"] - state["y"]) ** 2) * action["speed"]

        return flow

    except RewardSystemError as e:
        logger.error(f"Error calculating flow: {e}")
        return 0.0