from dataclasses import dataclass, field
from typing import Optional

from domain.state.state_type import StateType
from domain.state.state_role import StateRole


@dataclass
class State:
    name: str
    state_type: StateType
    role: StateRole
    tie_group: Optional[str]
    absorbing: bool = False  # C-terminal can only go to C-terminal

    # Parameters
    transitions: dict = field(default_factory=dict) 
    emissions: dict = field(default_factory=dict) 
