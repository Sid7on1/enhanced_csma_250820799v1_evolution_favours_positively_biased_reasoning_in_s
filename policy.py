import logging
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from typing import Dict, List, Tuple
from policy.config import Config
from policy.models import PolicyNetwork
from policy.utils import calculate_velocity, calculate_flow
from policy.exceptions import PolicyError
from policy.data import PolicyData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Policy:
    def __init__(self, config: Config):
        self.config = config
        self.policy_network = PolicyNetwork(config)
        self.optimizer = Adam(self.policy_network.parameters(), lr=config.lr)
        self.velocity_threshold = config.velocity_threshold
        self.flow_threshold = config.flow_threshold

    def train(self, data: PolicyData):
        try:
            # Validate input data
            if not isinstance(data, PolicyData):
                raise PolicyError("Invalid input data")

            # Calculate velocity and flow
            velocity = calculate_velocity(data)
            flow = calculate_flow(data)

            # Update policy network
            self.optimizer.zero_grad()
            loss = self.policy_network.calculate_loss(velocity, flow)
            loss.backward()
            self.optimizer.step()

            # Log training metrics
            logger.info(f"Training loss: {loss.item():.4f}")
            logger.info(f"Velocity: {velocity:.4f}")
            logger.info(f"Flow: {flow:.4f}")

        except PolicyError as e:
            logger.error(f"Error during training: {e}")

    def predict(self, data: PolicyData):
        try:
            # Validate input data
            if not isinstance(data, PolicyData):
                raise PolicyError("Invalid input data")

            # Calculate velocity and flow
            velocity = calculate_velocity(data)
            flow = calculate_flow(data)

            # Get policy output
            output = self.policy_network.predict(velocity, flow)

            # Log prediction metrics
            logger.info(f"Predicted output: {output:.4f}")
            logger.info(f"Velocity: {velocity:.4f}")
            logger.info(f"Flow: {flow:.4f}")

            return output

        except PolicyError as e:
            logger.error(f"Error during prediction: {e}")

    def save(self, path: str):
        try:
            # Save policy network
            torch.save(self.policy_network.state_dict(), path)

            # Log save metrics
            logger.info(f"Policy network saved to {path}")

        except Exception as e:
            logger.error(f"Error during save: {e}")

    def load(self, path: str):
        try:
            # Load policy network
            self.policy_network.load_state_dict(torch.load(path))

            # Log load metrics
            logger.info(f"Policy network loaded from {path}")

        except Exception as e:
            logger.error(f"Error during load: {e}")


class PolicyNetwork(nn.Module):
    def __init__(self, config: Config):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, velocity: float, flow: float):
        x = torch.tensor([velocity, flow], dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_loss(self, velocity: float, flow: float):
        output = self.forward(velocity, flow)
        loss = (output - torch.tensor([velocity, flow], dtype=torch.float32)) ** 2
        return loss.mean()


def calculate_velocity(data: PolicyData) -> float:
    # Calculate velocity using Flow Theory
    velocity = np.mean(data.velocity)
    return velocity


def calculate_flow(data: PolicyData) -> float:
    # Calculate flow using Flow Theory
    flow = np.mean(data.flow)
    return flow


class PolicyError(Exception):
    pass


class Config:
    def __init__(self):
        self.lr = 0.001
        self.velocity_threshold = 0.5
        self.flow_threshold = 0.5
        self.input_dim = 2
        self.hidden_dim = 64
        self.output_dim = 2


class PolicyData:
    def __init__(self, velocity: List[float], flow: List[float]):
        self.velocity = velocity
        self.flow = flow