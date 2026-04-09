from dataclasses import dataclass
from enum import Enum


class Label(Enum):
    SIGNAL = 1
    INNER = 2
    TRANS_MEMBRANE = 3
    OUTER = 4


@dataclass
class Segment:
    def __init__(self, label: Label, begin: int, end: int) -> None:
        self.label = label
        self.begin = begin
        self.end = end

    def __repr__(self) -> str:
        return f"label: {self.label}, begin: {self.begin}, end: {self.end}"


@dataclass
class Protein:
    def __init__(self, name: str, sequence: str, segments: list[Segment]) -> None:
        self.name = name
        self.sequence = sequence
        self.segments = segments

    def __repr__(self) -> str:
        return (
            f"name: {self.name}, sequence: {self.sequence}, segments: {self.segments}"
        )
