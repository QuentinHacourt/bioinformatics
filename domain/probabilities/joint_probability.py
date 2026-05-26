import numpy as np
from pprint import pprint

def annotation_to_state_sequence(
    annotated_sequence: str,
    name_to_idx: dict[str, int],
    obs,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray
) -> list[int]:
    
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
        dp[0, j] = log_pi[j] + log_B[j, obs[0]]

    for t in range(1, T):
        allowed_prev = allowed(annotated_sequence[t - 1])
        allowed_curr = allowed(annotated_sequence[t])
        for j in allowed_curr:
            best_i, best_score = -1, -np.inf
            for i in allowed_prev:
                s = dp[t-1, i] + log_A[i, j] + log_B[j, obs[t]]
                if s > best_score:
                    best_score, best_i = s, i

            if best_i == -1:
                raise ValueError(
                    f"No valid predecessor at t={t} for state {j} — "
                    "topology constraint is infeasible for this annotation."
                )
            dp[t, j]      = best_score
            backptr[t, j] = best_i

        allowed_last = allowed(annotated_sequence[-1])
        best_last    = max(allowed_last, key=lambda j: dp[-1, j])

    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    return path
