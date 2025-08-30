import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from collections import deque
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
MEMORY_SIZE = 100000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.1

# Experience replay buffer class
class ExperienceReplayBuffer(ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    @abstractmethod
    def add(self, experience: Dict):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Dict]:
        pass

# Implementation of experience replay buffer using a deque
class DequeExperienceReplayBuffer(ExperienceReplayBuffer):
    def add(self, experience: Dict):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict]:
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in batch]

# Implementation of experience replay buffer using a pandas DataFrame
class PandasExperienceReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.df = pd.DataFrame(columns=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, experience: Dict):
        self.df = pd.concat([self.df, pd.DataFrame([experience])], ignore_index=True)
        self.df = self.df.iloc[-self.capacity:]

    def sample(self, batch_size: int) -> List[Dict]:
        batch = self.df.sample(n=batch_size)
        return batch.to_dict(orient='records')

# Memory class
class Memory:
    def __init__(self):
        self.buffer = DequeExperienceReplayBuffer(MEMORY_SIZE)
        self.lock = Lock()

    def add(self, experience: Dict):
        with self.lock:
            self.buffer.add(experience)

    def sample(self, batch_size: int) -> List[Dict]:
        with self.lock:
            return self.buffer.sample(batch_size)

    def get_size(self) -> int:
        with self.lock:
            return len(self.buffer.buffer)

# Experience class
@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

# Agent class
class Agent:
    def __init__(self):
        self.memory = Memory()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < EPSILON:
            return np.random.randint(2)
        else:
            return torch.argmax(self.model(state)).item()

    def store_experience(self, experience: Experience):
        self.memory.add(experience)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        states = torch.tensor([x['state'] for x in batch])
        actions = torch.tensor([x['action'] for x in batch])
        rewards = torch.tensor([x['reward'] for x in batch])
        next_states = torch.tensor([x['next_state'] for x in batch])
        dones = torch.tensor([x['done'] for x in batch])

        q_values = self.model(states)
        next_q_values = self.model(next_states)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]

        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        loss = (q_values - target_q_values.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Main function
def main():
    agent = Agent()

    # Simulate some experiences
    for i in range(1000):
        state = np.random.rand(4)
        action = agent.select_action(state)
        next_state = np.random.rand(4)
        reward = np.random.rand()
        done = np.random.rand() < 0.1

        experience = Experience(state, action, reward, next_state, done)
        agent.store_experience(experience)

        if i % 100 == 0:
            agent.replay()

if __name__ == '__main__':
    main()