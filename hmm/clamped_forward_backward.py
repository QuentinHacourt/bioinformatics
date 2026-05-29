import numpy as np


def run_clamped_forward_backward(dw, state_path, obs):
    log_alpha_clamped = forward_clamped_log(
        obs, state_path, dw.log_A, dw.log_B, dw.log_pi
    )
    log_p_joint = log_alpha_clamped[-1, state_path[-1]]

    log_beta_clamped = backward_clamped_log(state_path, dw.log_A, dw.log_B, obs)

    return log_alpha_clamped, log_beta_clamped, log_p_joint


def forward_clamped_log(
    sequence: np.ndarray,
    state_path: list[int],
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_pi: np.ndarray,
) -> np.ndarray:
    T = len(sequence)
    N = log_A.shape[0]

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


def backward_clamped_log(state_path, log_A, log_B, obs):
    T = len(obs)
    N = log_A.shape[0]

    log_beta_clamped = np.full((T, N), -np.inf)
    log_beta_clamped[T - 1, state_path[T - 1]] = 0.0

    for t in range(T - 2, -1, -1):
        curr_s = state_path[t]
        next_s = state_path[t + 1]
        log_beta_clamped[t, curr_s] = (
            log_A[curr_s, next_s]
            + log_B[next_s, obs[t + 1]]
            + log_beta_clamped[t + 1, next_s]
        )
    return log_beta_clamped
