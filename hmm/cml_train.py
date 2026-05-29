import numpy as np
from domain.data_wrapper import DataWrapper
from hmm.clamped_forward_backward import run_clamped_forward_backward
from domain.probabilities.joint_probability import (
    annotation_to_state_sequence,
    encode_sequence,
)
from hmm.forward_backward import run_forward_backward


def cml_train(
    dw: DataWrapper,
    n_iter: int = 100,
    learning_rate: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:

    print(f"Starting CML training: {n_iter} iterations, lr={learning_rate}")

    prev_cml = None

    for iteration in range(n_iter):
        total_cml = 0.0
        grad_A = np.zeros_like(dw.log_A)
        grad_B = np.zeros_like(dw.log_B)

        for protein in dw.train:
            obs = encode_sequence(protein.sequence)
            state_path = annotation_to_state_sequence(
                protein.labels, dw.name_to_idx, obs, dw.log_A, dw.log_B, dw.log_pi
            )

            log_alpha, log_beta, log_p_obs = run_forward_backward(
                dw.log_A, dw.log_B, dw.log_pi, obs
            )

            log_alpha_clamped, log_beta_clamped, log_p_joint = (
                run_clamped_forward_backward(dw, state_path, obs)
            )

            total_cml += log_p_joint - log_p_obs

            correct_A, correct_B, expected_A, expected_B = _compute_expected_counts(
                dw.log_A, dw.log_B, obs, state_path, log_alpha, log_beta, log_p_obs
            )

            grad_A += correct_A - expected_A
            grad_B += correct_B - expected_B

        num_proteins = len(dw.train)
        grad_A /= num_proteins
        grad_B /= num_proteins
        _apply_update(dw, grad_A, grad_B, learning_rate)

        print(f"  iter {iteration+1:>3}/{n_iter}  CML objective: {total_cml:.4f}")

        if prev_cml is not None and abs(total_cml - prev_cml) / abs(prev_cml) < eps:
            print(f"  Converged at iteration {iteration+1}")
            break
        prev_cml = total_cml

    return dw.A, dw.B


def _apply_update(dw, grad_A, grad_B, learning_rate):
    # dw.log_A = dw.log_A + learning_rate * grad_A
    # dw.log_B = dw.log_B + learning_rate * grad_B

    # A_new = np.exp(dw.log_A)
    # B_new = np.exp(dw.log_B)

    # A_row_sums = A_new.sum(axis=1, keepdims=True)
    # B_row_sums = B_new.sum(axis=1, keepdims=True)

    # A_new = np.where(A_row_sums > 0, A_new / A_row_sums, A_new)
    # B_new = np.where(B_row_sums > 0, B_new / B_row_sums, B_new)

    # for state_indices in dw.tied_groups.values():
    #     if len(state_indices) < 2:
    #         continue
    #     avg_row = B_new[state_indices, :].mean(axis=0)
    #     if avg_row.sum() > 0:
    #         avg_row = avg_row / avg_row.sum()
    #     B_new[state_indices, :] = avg_row

    # dw.A = A_new
    # dw.B = B_new
    # dw.log_A = np.log(np.where(dw.A > 0, dw.A, 1e-16))
    # dw.log_B = np.log(np.where(dw.B > 0, dw.B, 1e-16))
    # 1. Perform gradient ascent step
    A_new = dw.A + learning_rate * grad_A
    B_new = dw.B + learning_rate * grad_B

    # 2. Renormalize rows to preserve valid probability distributions (sum to 1)
    A_row_sums = A_new.sum(axis=1, keepdims=True)
    A_new = np.where(A_row_sums > 0, A_new / A_row_sums, A_new)

    B_row_sums = B_new.sum(axis=1, keepdims=True)
    B_new = np.where(B_row_sums > 0, B_new / B_row_sums, B_new)

    # 3. Apply state tying constraints by averaging emission rows across tied groups
    for state_indices in dw.tied_groups.values():
        if len(state_indices) < 2:
            continue
        avg_row = B_new[state_indices, :].mean(axis=0)
        if avg_row.sum() > 0:
            avg_row = avg_row / avg_row.sum()
        B_new[state_indices, :] = avg_row

    # 4. Save the updated parameters and pre-calculate logs back into the data wrapper
    dw.A = A_new
    dw.B = B_new
    dw.log_A = np.log(np.where(dw.A > 0, dw.A, 1e-16))
    dw.log_B = np.log(np.where(dw.B > 0, dw.B, 1e-16))


def _compute_expected_counts(
    log_A: np.ndarray,
    log_B: np.ndarray,
    obs: np.ndarray,
    state_path: list[int],
    log_alpha,
    log_beta,
    log_p_obs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    T = len(obs)
    N = log_A.shape[0]
    K = log_B.shape[1]

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
        mask: bool = obs == k
        expected_B[:, k] = gamma[mask, :].sum(axis=0)

    return correct_A, correct_B, expected_A, expected_B
