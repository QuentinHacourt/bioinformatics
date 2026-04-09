from dataclasses import dataclass

@dataclass
class Interval:
    low: int
    up: int

@dataclass
class BetaStrand:
    name: str
    transmembrane: Interval