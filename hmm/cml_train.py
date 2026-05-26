import numpy as np
from hmm.forward_backward import (
    encode_sequence,
    run_forward_backward
)
from hmm.clamped_forward_backward import run_clamped_forward_backward
from domain.probabilities.joint_probability import annotation_to_state_sequence
from domain.protein import Protein
from domain.state.state import State

def build_tied_groups(
    states: list[State],
    name_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for state in states:
        group = state.tie_group if state.tie_group else state.name
        groups.setdefault(group, []).append(name_to_idx[state.name])
    return groups

def cml_train(
    proteins: list[Protein],
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    name_to_idx: dict[str, int],
    tied_groups: dict[str, list[int]],
    n_iter: int = 100,
    learning_rate: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    
    print(f"Starting CML training: {n_iter} iterations, lr={learning_rate}")

    prev_cml    = None

    for iteration in range(n_iter):
        total_cml    = 0.0
        grad_A       = np.zeros_like(A)
        grad_B       = np.zeros_like(B)

        for protein in proteins:
            obs        = encode_sequence(protein.sequence)
            state_path = annotation_to_state_sequence(
                protein.labels, name_to_idx, obs, A, B, pi)

            log_alpha, log_beta, log_p_obs = run_forward_backward(A, B, pi, obs)

            log_alpha_clamped, log_beta_clamped, log_p_joint = run_clamped_forward_backward(obs, state_path, A, B, pi)

            total_cml += log_p_joint - log_p_obs

            correct_A, correct_B, expected_A, expected_B = compute_expected_counts(A, B, obs, state_path, log_alpha, log_beta, log_p_obs)

            grad_A += correct_A - expected_A
            grad_B += correct_B - expected_B

        A, B = apply_update(A, B, grad_A, grad_B, learning_rate, eps, tied_groups)

        print(f"  iter {iteration+1:>3}/{n_iter}  CML objective: {total_cml:.4f}")

        if prev_cml is not None and abs(total_cml - prev_cml) / abs(prev_cml) < eps:
            print(f"  Converged at iteration {iteration+1}")
            break
        prev_cml = total_cml

    return A, B


def apply_update(A, B, grad_A, grad_B, learning_rate, eps, tied_groups):
    A_new = A + learning_rate * grad_A
    B_new = B + learning_rate * grad_B

    A_new = np.clip(A_new, eps, None)
    B_new = np.clip(B_new, eps, None)
    A_new = np.where(A > 0, A_new, 0.0)
    B_new = np.where(B > 0, B_new, 0.0)

    A_row_sums = A_new.sum(axis=1, keepdims=True)
    B_row_sums = B_new.sum(axis=1, keepdims=True)
    A_new = np.where(A_row_sums > 0, A_new / A_row_sums, A_new)
    B_new = np.where(B_row_sums > 0, B_new / B_row_sums, B_new)

    for state_indices in tied_groups.values():
        if len(state_indices) < 2:
            continue
        avg_row = B_new[state_indices, :].mean(axis=0)
        avg_row = avg_row / avg_row.sum()
        B_new[state_indices, :] = avg_row

    return A_new, B_new


def compute_expected_counts(
    A: np.ndarray,
    B: np.ndarray,
    obs: np.ndarray,
    state_path: list[int],
    log_alpha,
    log_beta,
    log_p_obs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    T = len(obs)
    N = A.shape[0]
    K = B.shape[1]

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    
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

    for t in range(1, T):
        log_xi = (
            log_alpha[t - 1, :, None]
            + log_A
            + log_B[:, obs[t]]
            + log_beta[t, :]
            - log_p_obs
        )
        expected_A += np.exp(log_xi)

    log_gamma = log_alpha + log_beta - log_p_obs
    gamma = np.exp(log_gamma)  

    for k in range(K):
        mask = obs == k
        expected_B[:, k] = gamma[mask, :].sum(axis=0)

    return correct_A, correct_B, expected_A, expected_B
