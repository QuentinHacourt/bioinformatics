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
    idx_to_name: list[str],
) -> tuple[float, str]:
    """
    Standard Viterbi — most probable path through states.
    Used as baseline and fallback.
    Returns (log_prob, labeling string).
    """
    T = len(obs)
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    # dp[t, j] = best log prob ending in state j at time t
    dp = np.full((T, N), -np.inf)
    backptr = np.zeros((T, N), dtype=int)

    dp[0, :] = log_pi + log_B[:, obs[0]]

    for t in range(1, T):
        for j in range(N):
            scores = dp[t - 1, :] + log_A[:, j]
            backptr[t, j] = np.argmax(scores)
            dp[t, j] = scores[backptr[t, j]] + log_B[j, obs[t]]

    # Traceback
    best_last = int(np.argmax(dp[-1, :]))
    log_prob = dp[-1, best_last]

    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    labeling = "".join(state_to_label(idx_to_name[s]) for s in path)
    return log_prob, labeling


def n_best(
    obs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    idx_to_name: list[str],
    N_hyp: int = 100,  # number of hypotheses to keep at each step
) -> tuple[float, str]:
    """
    N-best decoding algorithm (Schwartz & Chow, 1990) as described in
    Krogh (1997) — finds the most probable LABELING rather than path.

    Since multiple state paths can produce the same I/T/O labeling,
    this is guaranteed to return a result at least as good as Viterbi.

    Returns (log_prob, labeling string).
    """
    T = len(obs)
    N = A.shape[0]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    # Initialise: one hypothesis per state at t=0
    hypotheses: list[Hypothesis] = []
    for s in range(N):
        lp = log_pi[s] + log_B[s, obs[0]]
        if lp > -np.inf:
            hypotheses.append(
                Hypothesis(
                    log_prob=lp,
                    path=[s],
                    labeling=state_to_label(idx_to_name[s]),
                )
            )

    # Keep only top N_hyp by log prob
    hypotheses.sort(key=lambda h: h.log_prob, reverse=True)
    hypotheses = hypotheses[:N_hyp]

    # Expand hypotheses timestep by timestep
    for t in range(1, T):
        candidates: list[Hypothesis] = []

        for hyp in hypotheses:
            prev_s = hyp.path[-1]

            # Try every possible next state
            for next_s in range(N):
                trans_lp = log_A[prev_s, next_s]
                if trans_lp == -np.inf:
                    continue  # impossible transition, skip

                new_lp = hyp.log_prob + trans_lp + log_B[next_s, obs[t]]

                candidates.append(
                    Hypothesis(
                        log_prob=new_lp,
                        path=hyp.path + [next_s],
                        labeling=hyp.labeling + state_to_label(idx_to_name[next_s]),
                    )
                )

        if not candidates:
            # Fallback — shouldn't happen with a valid model
            print(f"Warning: no candidates at t={t}, falling back to Viterbi")
            return viterbi(obs, A, B, pi, idx_to_name)

        # Prune to top N_hyp
        candidates.sort(key=lambda h: h.log_prob, reverse=True)
        hypotheses = candidates[:N_hyp]

    # Best surviving hypothesis
    best = hypotheses[0]
    return best.log_prob, best.labeling


def decode(
    protein: Protein,
    states: dict[str, State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
    method: str = "n_best",  # "n_best" or "viterbi"
    N_hyp: int = 100,
) -> tuple[str, float]:
    """
    Decode a protein sequence into a labeling (I/T/O per residue).

    Returns (labeling, log_prob).
    """
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
    states: dict[str, State],
    trained_A: np.ndarray,
    trained_B: np.ndarray,
    method: str = "n_best",
) -> None:
    """
    Run decoding on all proteins and print per-residue accuracy.
    """
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
