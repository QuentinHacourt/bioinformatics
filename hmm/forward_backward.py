import domain.amino_acid as aa
import numpy as np
from scipy.special import logsumexp
from domain.state.state import State

AMINO_ACIDS = aa.amino_acids
AA_TO_IDX = aa.amino_acid_to_index


def build_index(states: dict[str, State]) -> tuple[list[str], dict[str, int]]:
    idx_to_name = sorted(states.keys())
    name_to_idx = {name: i for i, name in enumerate(idx_to_name)}
    return idx_to_name, name_to_idx


def build_transition_matrix(
    states: dict[str, State],
    name_to_idx: dict[str, int],
) -> np.ndarray:
    n = len(states)
    A = np.zeros((n, n))

    for name, state in states.items():
        i = name_to_idx[name]
        for target_name, prob in state.transitions.items():
            j = name_to_idx[target_name]
            A[i, j] = prob

    return A


def build_emission_matrix(
    states: dict[str, State],
    name_to_idx: dict[str, int],
) -> np.ndarray:
    n = len(states)
    B = np.zeros((n, len(AMINO_ACIDS)))

    for name, state in states.items():
        i = name_to_idx[name]
        for aa, prob in state.emissions.items():
            k = AA_TO_IDX[aa]
            B[i, k] = prob

    return B


def build_initial_distribution(
    states: dict[str, State],
    name_to_idx: dict[str, int],
) -> np.ndarray:
    pi = np.zeros(len(states))
    pi[name_to_idx["inner_n_term"]] = 1.0
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
        # for each destination state j, sum over all source states i
        log_alpha[t, :] = (
            logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0)  # shape (N,)
            + log_B[:, sequence[t]]
        )

    return log_alpha


def backward_log(sequence, A, B):
    T = sequence.shape[0]
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))

    log_beta = np.full((T, N), -np.inf)
    log_beta[T - 1, :] = 0.0  # log(1) = 0

    for t in range(T - 2, -1, -1):
        # for each source state j, sum over all destination states k
        log_beta[t, :] = logsumexp(
            log_A + log_B[:, sequence[t + 1]] + log_beta[t + 1, :],  # broadcast over j
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
