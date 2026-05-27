import numpy as np
from dataclasses import dataclass, field
from hmm.forward_backward import (
    build_index,
    build_transition_matrix,
    build_emission_matrix,
    build_initial_distribution,
    encode_sequence,
)
from domain.protein import Protein
from domain.state.state import State


# Map state names to their label character
def state_to_label(state_name: str) -> str:
    if "inner" in state_name:
        return "I"
    elif "outer" in state_name or "globular" in state_name:
        return "O"
    else:
        return "T"


@dataclass
class Hypothesis:
    """One candidate labeling being tracked by N-best."""

    log_prob: float  # cumulative log probability
    path: list[int]  # sequence of state indices
    labeling: str  # sequence of I/T/O characters

    def label(self, idx_to_name: list[str]) -> str:
        return self.labeling


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


def n_best(
    obs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    idx_to_name: list[State],
    N_hyp: int = 25, 
) -> tuple[float, str]:
  
    T = len(obs)
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-16))
    log_B = np.log(np.where(B > 0, B, 1e-16))
    log_pi = np.log(np.where(pi > 0, pi, 1e-16))

    hypotheses: list[Hypothesis] = []
    for s in range(N):
        lp = log_pi[s] + log_B[s, obs[0]]
        if lp > -np.inf:
            hypotheses.append(
                Hypothesis(
                    log_prob=lp,
                    path=[s],
                    labeling=state_to_label(idx_to_name[s].name),
                )
            )

    hypotheses.sort(key=lambda h: h.log_prob, reverse=True)
    hypotheses = hypotheses[:N_hyp]

    for t in range(1, T):
        candidates: list[Hypothesis] = []

        for hyp in hypotheses:
            prev_s = hyp.path[-1]
            for next_s in range(N):
                trans_lp = log_A[prev_s, next_s]
                if trans_lp == -np.inf:
                    continue

                new_lp = hyp.log_prob + trans_lp + log_B[next_s, obs[t]]

                candidates.append(
                    Hypothesis(
                        log_prob=new_lp,
                        path=hyp.path + [next_s],
                        labeling=hyp.labeling + state_to_label(idx_to_name[next_s].name),
                    )
                )

        if not candidates:
            print(f"Warning: no candidates at t={t}, falling back to Viterbi")
            return viterbi(obs, A, B, pi, idx_to_name)

        candidates.sort(key=lambda h: h.log_prob, reverse=True)
        hypotheses = candidates[:N_hyp]

    best = hypotheses[0]
    return best.log_prob, best.labeling


def decode(
    protein: Protein,
    states: list[State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
    method: str = "n-best", # n-best | viterbi
    N_hyp: int = 10,
) -> tuple[str, float]:
 
    idx_to_name, name_to_idx = build_index(states)
    pi = build_initial_distribution(states, name_to_idx)
    obs = encode_sequence(protein.sequence)

    if method == "viterbi":
        log_prob, labeling = viterbi(obs, trained_A, trained_B, pi, idx_to_name)
    else:
        log_prob, labeling = n_best(obs, trained_A, trained_B, pi, idx_to_name, N_hyp)

    return labeling, log_prob


def evaluate(
    proteins: list[Protein],
    states: list[State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
    method: str = "n_best",
) -> list[str]:
 
    total_residues = 0
    correct = 0
    temp_res: list[str] = []

    for protein in proteins:
        predicted, log_prob = decode(protein, states, trained_A, trained_B, method)
        true_labels = protein.labels

        if len(predicted) != len(true_labels):
            print(f"  {protein.name}: length mismatch, skipping")
            continue

        matches = sum(p == t for p, t in zip(predicted, true_labels))
        accuracy = matches / len(true_labels) * 100
        total_residues += len(true_labels)
        correct += matches

        print(
            f"  {protein.name:<10} accuracy: {accuracy:5.1f}%  "
            f"log_prob: {log_prob:.1f}  len: {len(true_labels)}"
        )
        temp_res.append(f"{protein.name:<10}\t accuracy: {accuracy:5.1f},\t log_prob: {log_prob:.1f}\t  len: {len(true_labels)}\n")
    
    overall = correct / total_residues * 100 if total_residues > 0 else 0
    print(f"\nOverall per-residue accuracy: {overall:.2f}\n")
    print(f"  ({correct}/{total_residues} residues correct)\n")
    temp_res.append(f"\nOverall per-residue accuracy: {overall:.2f}\n({correct}/{total_residues} residues correct)\n\n")

    return temp_res
