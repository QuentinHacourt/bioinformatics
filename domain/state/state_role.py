from enum import Enum, auto


class StateRole(Enum):
    # Inner loop roles
    N_TERMINAL = auto()
    INNER_LADDER = auto()
    C_TERMINAL = auto()

    # TM strand roles
    AROMATIC_BELT = auto()
    TM_EXTERIOR = auto()
    TM_INTERIOR = auto()

    # Outer loop roles
    GLOBULAR = auto()
    OUTER_LADDER = auto()
