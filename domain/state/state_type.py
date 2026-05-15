from enum import Enum, auto


class StateType(Enum):
    INNER_LOOP = auto()
    TRANSMEMBRANE = auto()
    OUTER_LOOP = auto()
