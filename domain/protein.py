from dataclasses import dataclass

@dataclass
class Protein:

    def __init__(self, name, code, sequence):
        self.name = name
        self. code = code
        self.sequence = sequence

    def __repr__(self):
        return f'name: {self.name}, code: {self.code}, sequence: {self.sequence}.'