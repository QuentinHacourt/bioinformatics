import numpy as np
from scipy.special import logsumexp
from hmm.forward_backward import (
    build_index,
    build_transition_matrix,
    build_emission_matrix,
    build_initial_distribution,
    encode_sequence,
    forward_log,
    backward_log,
)
from domain.probabilities.joint_probability import annotation_to_state_sequence
from domain.protein import Protein
from domain.state.state import State

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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


def compute_gradients(
    protein: Protein,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    obs = encode_sequence(protein.sequence)
    state_path = annotation_to_state_sequence(protein.labels, name_to_idx, A, B, pi)
    T = len(obs)
    N = A.shape[0]
    K = B.shape[1]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))

    log_alpha = forward_log(obs, A, B, pi)
    log_beta = backward_log(obs, A, B)
    log_p_obs = logsumexp(log_alpha[-1, :])

    log_alpha_joint = forward_joint_log(obs, state_path, A, B, pi)
    log_p_joint = log_alpha_joint[-1, state_path[-1]]

    grad_A = np.zeros((N, N))

    for t in range(1, T):
        i = state_path[t - 1]
        j = state_path[t]
        if A[i, j] > 0:
            grad_A[i, j] += 1.0 / A[i, j]

    for t in range(1, T):
        log_xi = (
            log_alpha[t - 1, :, None]
            + log_A
            + log_B[:, obs[t]]
            + log_beta[t, :]
            - log_p_obs
        )
        xi = np.exp(log_xi)
        with np.errstate(invalid="ignore", divide="ignore"):
            grad_A -= np.where(A > 0, xi / A, 0.0)

    grad_B = np.zeros((N, K))

    for t in range(T):
        j = state_path[t]
        k = obs[t]
        if B[j, k] > 0:
            grad_B[j, k] += 1.0 / B[j, k]

    log_gamma = log_alpha + log_beta - log_p_obs
    gamma = np.exp(log_gamma)

    for k in range(K):
        mask = obs == k
        expected = gamma[mask, :].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            grad_B[:, k] -= np.where(B[:, k] > 0, expected / B[:, k], 0.0)

    return grad_A, grad_B


def apply_gradients(
    A: np.ndarray,
    B: np.ndarray,
    grad_A: np.ndarray,
    grad_B: np.ndarray,
    learning_rate: float,
    tied_groups: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    A_new = A + learning_rate * grad_A
    B_new = B + learning_rate * grad_B

    eps = 1e-10
    A_new = np.clip(A_new, eps, None)
    B_new = np.clip(B_new, eps, None)

    A_new = np.where(A > 0, A_new, 0.0)
    B_new = np.where(B > 0, B_new, 0.0)

    A_row_sums = A_new.sum(axis=1, keepdims=True)
    A_new = np.where(A_row_sums > 0, A_new / A_row_sums, A_new)

    B_row_sums = B_new.sum(axis=1, keepdims=True)
    B_new = np.where(B_row_sums > 0, B_new / B_row_sums, B_new)

    for state_indices in tied_groups.values():
        if len(state_indices) < 2:
            continue
        avg_row = B_new[state_indices, :].mean(axis=0)
        avg_row = avg_row / avg_row.sum()
        B_new[state_indices, :] = avg_row

    return A_new, B_new


def build_tied_groups(
    states: dict[str, State],
    name_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for name, state in states.items():
        group = state.tie_group if state.tie_group else name
        groups.setdefault(group, []).append(name_to_idx[name])
    return groups


def cml_train(
    proteins: list[Protein],
    states: dict[str, State],
    n_iter: int = 100,
    learning_rate: float = 1.0,
    eps: float = 1e-10,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    idx_to_name, name_to_idx = build_index(states)
    tied_groups = build_tied_groups(states, name_to_idx)

    if A is None:
        A = build_transition_matrix(states, name_to_idx)
        B = build_emission_matrix(states, name_to_idx)
        
    pi = build_initial_distribution(states, name_to_idx, proteins)

    print(f"Starting CML training: {n_iter} iterations, lr={learning_rate}")
    print(f"  States: {len(states)}, Proteins: {len(proteins)}\n")

    prev_cml = None

    for iteration in range(n_iter):
        total_correct_A = np.zeros_like(A)
        total_correct_B = np.zeros_like(B)
        total_expected_A = np.zeros_like(A)
        total_expected_B = np.zeros_like(B)
        total_cml = 0.0

        for protein in proteins:
            try:
                correct_A, correct_B, expected_A, expected_B = compute_expected_counts(
                    protein, A, B, pi, name_to_idx
                )
                total_correct_A += correct_A
                total_correct_B += correct_B
                total_expected_A += expected_A
                total_expected_B += expected_B

                obs = encode_sequence(protein.sequence)
                state_path = annotation_to_state_sequence(protein.labels, name_to_idx, A, B, pi)
                log_alpha_joint = forward_joint_log(obs, state_path, A, B, pi)
                log_p_joint = log_alpha_joint[-1, state_path[-1]]
                log_alpha = forward_log(obs, A, B, pi)
                log_p_obs = logsumexp(log_alpha[-1, :])
                total_cml += log_p_joint - log_p_obs

            except ValueError as e:
                print(f"  Skipping {protein.name}: {e}")

        # A
        with np.errstate(divide="ignore", invalid="ignore"):
            A_ratio = np.where(
                total_expected_A > eps,
                np.divide(
                    total_correct_A,
                    total_expected_A,
                    out=np.ones_like(total_correct_A),
                    where=total_expected_A > eps,
                ),
                1.0,
            )
        A_ratio = np.clip(A_ratio, 1e-10, 1e10)
        A_new = A * np.power(A_ratio, learning_rate)
        A_new = np.clip(A_new, eps, None)
        A_new = np.where(A > 0, A_new, 0.0)
        A_row_sums = A_new.sum(axis=1, keepdims=True)
        A_new = np.where(A_row_sums > 0, A_new / A_row_sums, A_new)

        # B
        with np.errstate(divide="ignore", invalid="ignore"):
            B_ratio = np.where(
                total_expected_B > eps,
                np.divide(
                    total_correct_B,
                    total_expected_B,
                    out=np.ones_like(total_correct_B),
                    where=total_expected_B > eps,
                ),
                1.0,
            )
        B_ratio = np.clip(B_ratio, 1e-10, 1e10)
        B_new = B * np.power(B_ratio, learning_rate)
        B_new = np.clip(B_new, eps, None)
        B_new = np.where(B > 0, B_new, 0.0)
        B_row_sums = B_new.sum(axis=1, keepdims=True)
        B_new = np.where(B_row_sums > 0, B_new / B_row_sums, B_new)

        # Enforce tying
        for state_indices in tied_groups.values():
            if len(state_indices) < 2:
                continue
            avg_row = B_new[state_indices, :].mean(axis=0)
            avg_row = avg_row / avg_row.sum()
            B_new[state_indices, :] = avg_row

        A, B = A_new, B_new

        print(f"  iter {iteration+1:>3}/{n_iter}  CML objective: {total_cml:.4f}")

        if prev_cml is not None and abs(total_cml - prev_cml) / abs(prev_cml) < 1e-6:
            print(f"  Converged at iteration {iteration+1}")
            break

        prev_cml = total_cml

    return A, B


def compute_expected_counts(
    protein: Protein,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        correct_A  — transition counts on the correct path
        correct_B  — emission counts on the correct path
        expected_A — expected transition counts under full model
        expected_B — expected emission counts under full model
    """
    obs = encode_sequence(protein.sequence)
    state_path = annotation_to_state_sequence(protein.labels, name_to_idx, A, B, pi)
    T = len(obs)
    N = A.shape[0]
    K = B.shape[1]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))

    log_alpha = forward_log(obs, A, B, pi)
    log_beta = backward_log(obs, A, B)
    log_p_obs = logsumexp(log_alpha[-1, :])
    
    correct_A = np.zeros((N, N))
    correct_B = np.zeros((N, K))

    for t in range(1, T):
        i = state_path[t - 1]
        j = state_path[t]
        correct_A[i, j] += 1.0

    for t in range(T):
        j = state_path[t]
        k = obs[t]
        correct_B[j, k] += 1.0

    expected_A = np.zeros((N, N))
    expected_B = np.zeros((N, K))

    # xi: expected transition counts
    for t in range(1, T):
        log_xi = (
            log_alpha[t - 1, :, None]
            + log_A
            + log_B[:, obs[t]]
            + log_beta[t, :]
            - log_p_obs
        )
        expected_A += np.exp(log_xi)

    # gamma: expected state occupation counts
    log_gamma = log_alpha + log_beta - log_p_obs
    gamma = np.exp(log_gamma)  # (T, N)

    for k in range(K):
        mask = obs == k
        expected_B[:, k] = gamma[mask, :].sum(axis=0)

    return correct_A, correct_B, expected_A, expected_B
