import numpy as np
from dataclasses import dataclass
from domain.probabilities.util import (
    build_index,
    build_initial_distribution,
    encode_sequence,
)
from domain.protein import Protein
from domain.state.state import State
from utils.print_report import print_viterbi


def state_to_label(state_name: str) -> str:
    if "inner" in state_name:
        return "I"
    elif "outer" in state_name or "globular" in state_name:
        return "O"
    else:
        return "T"


def viterbi(
    obs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    idx_to_name: list[State],
) -> tuple[float, str]:
    T = len(obs)
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-16))
    log_B = np.log(np.where(B > 0, B, 1e-16))
    log_pi = np.log(np.where(pi > 0, pi, 1e-16))

    dp = np.full((T, N), -np.inf)
    backptr = np.zeros((T, N), dtype=int)

    dp[0, :] = log_pi + log_B[:, obs[0]]

    for t in range(1, T):
        for j in range(N):
            scores = dp[t - 1, :] + log_A[:, j]
            backptr[t, j] = np.argmax(scores)
            dp[t, j] = scores[backptr[t, j]] + log_B[j, obs[t]]

    best_last = int(np.argmax(dp[-1, :]))
    log_prob = dp[-1, best_last]

    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    labeling = "".join(state_to_label(idx_to_name[s].name) for s in path)
    return log_prob, labeling


def decode(
    protein: Protein,
    states: list[State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
) -> tuple[str, float]:

    idx_to_name, name_to_idx = build_index(states)
    pi = build_initial_distribution(states, name_to_idx)
    obs = encode_sequence(protein.sequence)

    log_prob, labeling = viterbi(obs, trained_A, trained_B, pi, idx_to_name)

    return labeling, log_prob


def evaluate(
    proteins: list[Protein],
    states: list[State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
) -> tuple[ dict[str, int], float]:

    total_residues = 0
    correct = 0
    scores = {}

    for protein in proteins:
        predicted, log_prob = decode(protein, states, trained_A, trained_B)
        true_labels = protein.labels

        if len(predicted) != len(true_labels):
            print(f"  {protein.name}: length mismatch, skipping")
            continue

        matches = sum(p == t for p, t in zip(predicted, true_labels))
        accuracy = matches / len(true_labels) * 100
        total_residues += len(true_labels)
        correct += matches

        scores[protein.name] = accuracy

    overall = correct / total_residues * 100 if total_residues > 0 else 0

    print_viterbi(scores, overall)

    return scores, overall
