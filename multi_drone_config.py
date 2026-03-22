from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class MultiDroneConfig:
    # Environment size and config
    grid_size: Tuple[int, int, int] 
    change_altitude: bool
    start_positions: list[int]
    goal_positions: list[int]

    # MDP parameters
    discount_factor: float
    step_cost: float
    collision_penalty: float
    goal_reward: float
    max_num_steps: int
    alpha: float

    # Optional obstacle cells
    obstacle_cells: Optional[list] = None    
    seed: Optional[int] = None