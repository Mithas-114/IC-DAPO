"""Random selection strategy."""

import random
from typing import List

from verl.utils.fewshots.strategies.base import SelectionStrategy


class RandomStrategy(SelectionStrategy):
    """Uniform random selection without replacement within each group."""
    
    def select(self, num_groups: int, num_shots_per_group: int) -> List[List[int]]:
        groups = []
        for _ in range(num_groups):
            if num_shots_per_group <= self.num_examples:
                group_indices = random.sample(range(self.num_examples), num_shots_per_group)
            else:
                group_indices = [random.randint(0, self.num_examples - 1) 
                               for _ in range(num_shots_per_group)]
            groups.append(group_indices)
        return groups

    def update(self, *args, **kwargs):
        # No-op for random strategy
        pass
