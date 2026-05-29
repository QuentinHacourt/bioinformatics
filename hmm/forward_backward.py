import domain.amino_acid as aa
import numpy as np
from scipy.special import logsumexp
from domain.state.state import State
from domain.protein import Protein

AMINO_ACIDS = aa.AMINO_ACIDS
AA_TO_IDX = aa.AMINO_ACID_TO_INDEX


def forward_log(sequence, log_A, log_B, log_pi):
    T = sequence.shape[0]
    N = log_A.shape[0]

    log_alpha = np.full((T, N), -np.inf)
    log_alpha[0, :] = log_pi + log_B[:, sequence[0]]

    for t in range(1, T):
        log_alpha[t, :] = (
            logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_B[:, sequence[t]]
        )

    return log_alpha


def backward_log(sequence, log_A, log_B):
    T = sequence.shape[0]
    N = log_A.shape[0]

    log_beta = np.full((T, N), -np.inf)
    log_beta[T - 1, :] = 0.0

    for t in range(T - 2, -1, -1):
        log_beta[t, :] = logsumexp(
            log_A + log_B[:, sequence[t + 1]] + log_beta[t + 1, :],
            axis=1,
        )

    return log_beta


def run_forward_backward(log_A, log_B, log_pi, obs):

    log_alpha = forward_log(obs, log_A, log_B, log_pi)
    log_beta = backward_log(obs, log_A, log_B)
    log_p_obs = logsumexp(log_alpha[-1, :])

    return log_alpha, log_beta, log_p_obs
