from dataclasses import dataclass
import random
from domain.probabilities.transition import transition
from domain.protein import Protein
from domain.state.state import State
import numpy as np

from domain.probabilities.emission import emission
from domain.state.util import build_states
from domain.probabilities.joint_probability import (
    build_emission_matrix,
    build_index,
    build_initial_distribution,
    build_transition_matrix,
)
from domain.probabilities.joint_probability import build_tied_groups
from utils.xml_reader import collect_data


@dataclass
class DataWrapper:
    train: list[Protein]
    test: list[Protein]
    proteins: list[Protein]
    states: list[State]
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    log_A: np.ndarray
    log_B: np.ndarray
    log_pi: np.ndarray
    name_to_idx: dict[str, int]
    idx_to_name: list[State]
    tied_groups: dict[str, list[int]]

    def __init__(self):
        self.proteins = collect_data()
        self.states = build_states()
        emission(self.states, self.proteins)
        transition(self.states)

        self.idx_to_name, self.name_to_idx = build_index(self.states)
        self.tied_groups = build_tied_groups(self.states, self.name_to_idx)
        self.A = build_transition_matrix(self.states, self.name_to_idx)
        self.B = build_emission_matrix(self.states, self.name_to_idx)
        self.pi = build_initial_distribution(
            self.states, self.name_to_idx, self.proteins
        )

        self.log_A = np.log(np.where(self.A > 0, self.A, 1e-16))
        self.log_B = np.log(np.where(self.B > 0, self.B, 1e-16))
        self.log_pi = np.log(np.where(self.pi > 0, self.pi, 1e-16))

    def sample(self):
        self.train = random.sample(self.proteins, 7)
        self.test = [item for item in self.proteins if item not in self.train]
