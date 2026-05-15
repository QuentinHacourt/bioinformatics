from dataclasses import dataclass
import inspect


@dataclass
class Protein:
    name: str
    sequence: str
    labels: str

    def __repr__(self) -> str:
        return inspect.cleandoc(f"""
            === Protein: {self.name} ===
            sequence: {self.sequence}
            labels:   {self.labels}
        """)
