import pandas as pd
import numpy as np

# Source: https://adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/


def forward(
    sequence, transition_probabilities, emission_probabilities, initial_distribution
):
    # Initialization step
    alpha = np.zeros((sequence.shape[0], transition_probabilities.shape[0]))
    alpha[0, :] = initial_distribution * emission_probabilities[:, sequence[0]]

    # Induction step
    # a_{ij}
    # t = step count
    # j = state to
    for t in range(1, sequence.shape[0]):
        for j in range(transition_probabilities.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = (
                alpha[t - 1].dot(transition_probabilities[:, j])
                * emission_probabilities[j, sequence[t]]
            )

    return alpha


def backward(sequence, transition_probabilities, emission_probabilities):
    beta = np.zeros((sequence.shape[0], transition_probabilities.shape[0]))

    # setting beta(T) = 1
    beta[sequence.shape[0] - 1] = np.ones((transition_probabilities.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(sequence.shape[0] - 2, -1, -1):
        for j in range(transition_probabilities.shape[0]):
            beta[t, j] = (beta[t + 1] * emission_probabilities[:, sequence[t + 1]]).dot(
                transition_probabilities[j, :]
            )

    return beta


def forward_backward(
    sequence,
    transition_probabilities,
    emission_probabilities,
    initial_distribution,
    n_iter=100,
):
    M = transition_probabilities.shape[0]
    T = len(sequence)

    for n in range(n_iter):
        alpha = forward(
            sequence,
            transition_probabilities,
            emission_probabilities,
            initial_distribution,
        )
        beta = backward(sequence, transition_probabilities, emission_probabilities)

        # greek letter xi
        # the probability of being in state S_i at time instance t
        # and in state S_j at time instance t + 1
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t, :].T, transition_probabilities)
                * emission_probabilities[:, sequence[t + 1]].T,
                beta[t + 1, :],
            )
            for i in range(M):
                numerator = (
                    alpha[t, i]
                    * transition_probabilities[i, :]
                    * emission_probabilities[:, sequence[t + 1]].T
                    * beta[t + 1, :].T
                )
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        transition_probabilities = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape(
            (-1, 1)
        )

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = emission_probabilities.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            emission_probabilities[:, l] = np.sum(gamma[:, sequence == l], axis=1)

        emission_probabilities = np.divide(
            emission_probabilities, denominator.reshape((-1, 1))
        )

    return {"a": transition_probabilities, "b": emission_probabilities}


def modified_forward_backward():
    pass
