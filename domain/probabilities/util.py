import numpy as np
from domain.amino_acid import AMINO_ACID_TO_INDEX, AMINO_ACIDS

from domain.protein import Protein
from domain.state.state import State


def annotation_to_state_sequence(
    annotated_sequence: str,
    name_to_idx: dict[str, int],
    obs,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_pi: np.ndarray,
) -> list[int]:

    T = len(annotated_sequence)
    N = len(name_to_idx)

    def allowed(label_char: str) -> set[int]:
        if label_char == "I":
            return {
                name_to_idx["inner_n_term"],
                name_to_idx["inner_c_term"],
                *[name_to_idx[f"inner_ladder_{i}"] for i in range(12)],
            }
        elif label_char == "O":
            return {
                name_to_idx["outer_globular"],
                *[name_to_idx[f"outer_ladder_{i}"] for i in range(12)],
            }
        elif label_char == "T":
            return {
                *[name_to_idx[f"tm_aromatic_top_{i}"] for i in range(2)],
                *[name_to_idx[f"tm_aromatic_bottom_{i}"] for i in range(2)],
                *[name_to_idx[f"tm_exterior_{i}"] for i in range(14)],
                *[name_to_idx[f"tm_interior_{i}"] for i in range(16)],
            }
        return set()

    dp = np.full((T, N), -np.inf)
    backptr = np.zeros((T, N), dtype=int)

    for j in allowed(annotated_sequence[0]):
        dp[0, j] = log_pi[j] + log_B[j, obs[0]]
    for t in range(1, T):
        allowed_prev = allowed(annotated_sequence[t - 1])
        allowed_curr = allowed(annotated_sequence[t])

        for j in allowed_curr:
            best_i, best_score = -1, -np.inf
            for i in allowed_prev:
                if log_A[i, j] > -1e10:
                    s = dp[t - 1, i] + log_A[i, j] + log_B[j, obs[t]]
                    if s > best_score:
                        best_score, best_i = s, i

            dp[t, j] = best_score
            backptr[t, j] = best_i

    allowed_last = allowed(annotated_sequence[-1])
    best_last = max(allowed_last, key=lambda j: dp[-1, j])

    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()
    return path


def build_tied_groups(
    states: list[State],
    name_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for state in states:
        group = state.tie_group if state.tie_group else state.name
        groups.setdefault(group, []).append(name_to_idx[state.name])
    return groups


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
            k = AMINO_ACID_TO_INDEX[aa]
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
        if aa in AMINO_ACID_TO_INDEX:
            encoded.append(AMINO_ACID_TO_INDEX[aa])
        else:
            print(f"Warning: unknown amino acid '{aa}', skipping.")

    return np.array(encoded, dtype=int)
