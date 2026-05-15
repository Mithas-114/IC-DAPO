"""Base class for example selection strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class SelectionStrategy(ABC):
    """Abstract base class for example selection strategies."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples
        self.num_examples = len(examples)
        self.selection_counts = np.zeros(self.num_examples, dtype=np.int32)
        self.reward_sums = np.zeros(self.num_examples, dtype=np.float32)

    @abstractmethod
    def select(self, num_groups: int, num_shots_per_group: int) -> List[List[int]]:
        """Select examples for multiple groups.
        
        Returns:
            List of groups, where each group is a list of example indices.
        """
        pass

    def update(self, selected_indices: List[int], rewards: List[float]):
        """Update strategy statistics after observing rewards."""
        for idx, reward in zip(selected_indices, rewards):
            if np.isnan(reward) or np.isinf(reward):
                continue
            self.selection_counts[idx] += 1
            self.reward_sums[idx] += reward