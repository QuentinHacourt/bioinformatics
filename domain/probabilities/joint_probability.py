import numpy as np
from scipy.special import logsumexp
from domain.protein import Protein
from domain.state.state import State, StateRole
from hmm.forward_backward import (
    build_emission_matrix,
    build_index,
    build_initial_distribution,
    build_transition_matrix,
    encode_sequence,
    forward_log,
)


def annotation_to_state_sequence(
    annotated_sequence: str,
    name_to_idx: dict[str, int],
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray
) -> list[int]:
    obs     = encode_sequence(annotated_sequence[0])
    T       = len(annotated_sequence)
    N       = len(name_to_idx)

    log_A  = np.log(np.where(A  > 0, A,  1e-300))
    log_B  = np.log(np.where(B  > 0, B,  1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

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
    dp      = np.full((T, N), -np.inf)
    backptr = np.zeros((T, N), dtype=int)

    for j in allowed(annotated_sequence[0]):
        dp[0, j] = log_pi[j] 

    for t in range(1, T):
        allowed_j = allowed(annotated_sequence[t])
        for j in allowed_j:
            scores = dp[t-1, :] + log_A[:, j]
            best_i = int(np.argmax(scores))
            dp[t, j]       = scores[best_i]
            backptr[t, j]  = best_i

    allowed_last = allowed(annotated_sequence[-1])
    best_last    = max(allowed_last, key=lambda j: dp[-1, j])

    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    return path


def joint_log_probability(
    protein: Protein,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    name_to_idx: dict[str, int]
) -> float:
  
    obs = encode_sequence(protein.sequence)
    state_path = annotation_to_state_sequence(protein.labels, name_to_idx, A, B, pi)

    if len(obs) != len(state_path):
        raise ValueError(
            f"{protein.name}: sequence length {len(obs)} != "
            f"annotation length {len(state_path)}"
        )

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    log_prob = log_pi[state_path[0]] + log_B[state_path[0], obs[0]]

    for t in range(1, len(obs)):
        i = state_path[t - 1]
        j = state_path[t]
        log_prob += log_A[i, j] + log_B[j, obs[t]]

    return log_prob


def cml_log_probability(
    protein: Protein,
    states: dict[str, State],
) -> float:
   
    idx_to_name, name_to_idx = build_index(states)
    A = build_transition_matrix(states, name_to_idx)
    B = build_emission_matrix(states, name_to_idx)
    pi = build_initial_distribution(states, name_to_idx)
    obs = encode_sequence(protein.sequence)

    log_joint = joint_log_probability(protein, A, B, pi, name_to_idx)

    log_alpha = forward_log(obs, A, B, pi)
    log_p_obs = logsumexp(log_alpha[-1, :])

    log_conditional = log_joint - log_p_obs

    print(f"{protein.name}")
    print(f"  log P(x,y):     {log_joint:.4f}")
    print(f"  log P(x):       {log_p_obs:.4f}")
    print(f"  log P(y|x):     {log_conditional:.4f}") 
    print(f"  P(y|x):         {np.exp(log_conditional):.4e}")

    return log_conditional


def total_cml_objective(proteins: list[Protein], states: dict[str, State]) -> float:
    total = sum(cml_log_probability(p, states) for p in proteins)
    print(f"\nTotal CML objective: {total:.4f}")
    return total
