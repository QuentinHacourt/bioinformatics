import domain.amino_acid as aa
import numpy as np
from scipy.special import logsumexp
from domain.state.state import State
from domain.protein import Protein

AMINO_ACIDS = aa.AMINO_ACIDS
AA_TO_IDX = aa.AMINO_ACID_TO_INDEX


def build_index(states: list[State]) -> tuple[list[State], dict[str, int]]:
    idx_to_name = sorted(states, key=lambda x: x.name)
    name_to_idx = {state.name: i for i, state in enumerate(idx_to_name)}
    return idx_to_name, name_to_idx


def build_transition_matrix(
    states: list[State],
    name_to_idx: dict[str, int],
) -> np.ndarray:
    n = len(states)
    A = np.zeros((n, n))

    for state in states:
        i = name_to_idx[state.name]
        for target_name, prob in state.transitions.items():
            j = name_to_idx[target_name]
            A[i, j] = prob

    return A


def build_emission_matrix(
    states: list[State],
    name_to_idx: dict[str, int],
) -> np.ndarray:
    n = len(states)
    B = np.zeros((n, len(AMINO_ACIDS)))

    for state in states:
        i = name_to_idx[state.name]
        for aa, prob in state.emissions.items():
            k = AA_TO_IDX[aa]
            B[i, k] = prob

    return B


def build_initial_distribution(
    states: list[State],
    name_to_idx: dict[str, int],
    proteins: list[Protein] | None = None, 
) -> np.ndarray:
    pi = np.zeros(len(states))

    if proteins is None:
        pi[name_to_idx["inner_n_term"]] = 1.0
        return pi

    starts = {"I": 0, "O": 0, "T": 0}
    for protein in proteins:
        if protein.labels:
            starts[protein.labels[0]] = starts.get(protein.labels[0], 0) + 1

    total = sum(starts.values())
    print(f"  Start label distribution: {starts}")

    inner_start = name_to_idx["inner_n_term"]
    outer_start = name_to_idx["outer_ladder_0"]

    pi[inner_start] = starts.get("I", 0) / total
    pi[outer_start] = starts.get("O", 0) / total
    
    if starts.get("T", 0) > 0:
        pi[name_to_idx["tm_aromatic_top_0"]] = starts["T"] / total

    return pi


def encode_sequence(sequence: str) -> np.ndarray:
    encoded = []
    for aa in sequence:
        if aa in AA_TO_IDX:
            encoded.append(AA_TO_IDX[aa])
        else:
            print(f"Warning: unknown amino acid '{aa}', skipping.")

    return np.array(encoded, dtype=int)


def forward_log(sequence, A, B, pi):
    T = sequence.shape[0]
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0, :] = log_pi + log_B[:, sequence[0]]

    for t in range(1, T):
        log_alpha[t, :] = (
            logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0)
            + log_B[:, sequence[t]]
        )

    return log_alpha


def backward_log(sequence, A, B):
    T = sequence.shape[0]
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))

    log_beta = np.full((T, N), -np.inf)
    log_beta[T - 1, :] = 0.0  

    for t in range(T - 2, -1, -1):
        log_beta[t, :] = logsumexp(
            log_A + log_B[:, sequence[t + 1]] + log_beta[t + 1, :],
            axis=1,
        )

    return log_beta


def run_forward_backward(protein, states):
    idx_to_name, name_to_idx = build_index(states)
    A = build_transition_matrix(states, name_to_idx)
    B = build_emission_matrix(states, name_to_idx)
    pi = build_initial_distribution(states, name_to_idx)
    obs = encode_sequence(protein.sequence)

    log_alpha = forward_log(obs, A, B, pi)
    log_beta = backward_log(obs, A, B)

    index = {"idx_to_name": idx_to_name, "name_to_idx": name_to_idx}
    return log_alpha, log_beta, index

def forward_joint_log(
    sequence: np.ndarray,
    state_path: list[int],
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    T = len(sequence)
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    log_alpha_joint = np.full((T, N), -np.inf)

    s0 = state_path[0]
    log_alpha_joint[0, s0] = log_pi[s0] + log_B[s0, sequence[0]]

    for t in range(1, T):
        prev_s = state_path[t - 1]
        curr_s = state_path[t]
        log_alpha_joint[t, curr_s] = (
            log_alpha_joint[t - 1, prev_s]
            + log_A[prev_s, curr_s]
            + log_B[curr_s, sequence[t]]
        )

    return log_alpha_joint