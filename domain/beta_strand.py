from dataclasses import dataclass


@dataclass
class Interval:
    low: int
    up: int

    def __repr__(self):
        return f"low: {self.low}, up: {self.up}"


@dataclass
class BetaStrand:
    name: str
    transmembrane: Interval

    def __repr__(self):
        return f"name: {self.name}, transmembrane: [{self.transmembrane}]"
