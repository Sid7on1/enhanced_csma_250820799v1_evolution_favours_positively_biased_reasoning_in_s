import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from threading import Lock

# Define constants and configuration
class Config:
    VELOCITY_THRESHOLD = 0.5
    FLOW_THEORY_THRESHOLD = 0.8
    POPULATION_SIZE = 100
    ADAPTIVE_LEARNING_RATE = 0.01

# Define exception classes
class EnvironmentException(Exception):
    pass

class InvalidConfigurationException(EnvironmentException):
    pass

class InvalidAgentException(EnvironmentException):
    pass

# Define data structures and models
class Agent:
    def __init__(self, id: int, velocity: float, flow: float):
        self.id = id
        self.velocity = velocity
        self.flow = flow

class EnvironmentState:
    def __init__(self, agents: List[Agent], population_size: int):
        self.agents = agents
        self.population_size = population_size

# Define validation functions
def validate_config(config: Dict) -> bool:
    if 'velocity_threshold' not in config or 'flow_theory_threshold' not in config:
        return False
    if config['velocity_threshold'] < 0 or config['velocity_threshold'] > 1:
        return False
    if config['flow_theory_threshold'] < 0 or config['flow_theory_threshold'] > 1:
        return False
    return True

def validate_agent(agent: Agent) -> bool:
    if agent.velocity < 0 or agent.velocity > 1:
        return False
    if agent.flow < 0 or agent.flow > 1:
        return False
    return True

# Define utility methods
def calculate_velocity(agent: Agent) -> float:
    return agent.velocity * Config.ADAPTIVE_LEARNING_RATE

def calculate_flow(agent: Agent) -> float:
    return agent.flow * Config.ADAPTIVE_LEARNING_RATE

# Define the main environment class
class Environment:
    def __init__(self, config: Dict):
        if not validate_config(config):
            raise InvalidConfigurationException('Invalid configuration')
        self.config = config
        self.state = EnvironmentState([], Config.POPULATION_SIZE)
        self.lock = Lock()

    def create_agent(self, id: int, velocity: float, flow: float) -> Agent:
        if not validate_agent(Agent(id, velocity, flow)):
            raise InvalidAgentException('Invalid agent')
        return Agent(id, velocity, flow)

    def add_agent(self, agent: Agent):
        with self.lock:
            self.state.agents.append(agent)

    def remove_agent(self, agent_id: int):
        with self.lock:
            self.state.agents = [agent for agent in self.state.agents if agent.id != agent_id]

    def update_agent(self, agent_id: int, velocity: float, flow: float):
        with self.lock:
            for agent in self.state.agents:
                if agent.id == agent_id:
                    agent.velocity = velocity
                    agent.flow = flow
                    break

    def get_agents(self) -> List[Agent]:
        with self.lock:
            return self.state.agents

    def get_population_size(self) -> int:
        with self.lock:
            return self.state.population_size

    def calculate_velocity_threshold(self) -> float:
        return Config.VELOCITY_THRESHOLD

    def calculate_flow_theory_threshold(self) -> float:
        return Config.FLOW_THEORY_THRESHOLD

    def apply_velocity_threshold(self, agent: Agent) -> bool:
        return agent.velocity >= self.calculate_velocity_threshold()

    def apply_flow_theory_threshold(self, agent: Agent) -> bool:
        return agent.flow >= self.calculate_flow_theory_threshold()

    def step(self):
        with self.lock:
            for agent in self.state.agents:
                velocity = calculate_velocity(agent)
                flow = calculate_flow(agent)
                self.update_agent(agent.id, velocity, flow)

    def reset(self):
        with self.lock:
            self.state = EnvironmentState([], Config.POPULATION_SIZE)

# Define integration interfaces
class EnvironmentInterface(ABC):
    @abstractmethod
    def create_environment(self, config: Dict) -> Environment:
        pass

class EnvironmentFactory(EnvironmentInterface):
    def create_environment(self, config: Dict) -> Environment:
        return Environment(config)

# Define unit test compatibility
import unittest
from unittest.mock import Mock

class TestEnvironment(unittest.TestCase):
    def test_create_environment(self):
        config = {'velocity_threshold': 0.5, 'flow_theory_threshold': 0.8}
        environment = EnvironmentFactory().create_environment(config)
        self.assertIsInstance(environment, Environment)

    def test_create_agent(self):
        config = {'velocity_threshold': 0.5, 'flow_theory_threshold': 0.8}
        environment = EnvironmentFactory().create_environment(config)
        agent = environment.create_agent(1, 0.6, 0.9)
        self.assertIsInstance(agent, Agent)

    def test_add_agent(self):
        config = {'velocity_threshold': 0.5, 'flow_theory_threshold': 0.8}
        environment = EnvironmentFactory().create_environment(config)
        agent = environment.create_agent(1, 0.6, 0.9)
        environment.add_agent(agent)
        self.assertIn(agent, environment.get_agents())

    def test_remove_agent(self):
        config = {'velocity_threshold': 0.5, 'flow_theory_threshold': 0.8}
        environment = EnvironmentFactory().create_environment(config)
        agent = environment.create_agent(1, 0.6, 0.9)
        environment.add_agent(agent)
        environment.remove_agent(agent.id)
        self.assertNotIn(agent, environment.get_agents())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = {'velocity_threshold': 0.5, 'flow_theory_threshold': 0.8}
    environment = EnvironmentFactory().create_environment(config)
    agent = environment.create_agent(1, 0.6, 0.9)
    environment.add_agent(agent)
    environment.step()
    logging.info(f'Agents: {environment.get_agents()}')
    logging.info(f'Population size: {environment.get_population_size()}')
    unittest.main(argv=[''], verbosity=2, exit=False)