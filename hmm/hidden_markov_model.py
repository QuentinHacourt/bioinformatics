import numpy as np
from hmmlearn import hmm
import numpy.typing as npt

# from domain.sequence import Protein


def test(
    initial_distribution: list[float],
    transition_distribution: npt.NDArray[np.float64],
    observation_distribution: npt.NDArray[np.float64],
    # training_set: list[list[int]],
    test_set: list[int],
):
    states = 3

    model = hmm.CategoricalHMM(n_components=states, init_params="")

    model.startprob_ = np.array(initial_distribution)
    model.transmat_ = np.array(transition_distribution)
    model.emissionprob_ = np.array(observation_distribution)

    # trials, simulated_states = model.sample(10000)

    # print(trials[:10])
    # X_train = trials[: trials.shape[0] // 2]
    # X_test = trials[trials.shape[0] // 2 :]
    # X_train = np.concatenate([np.array(p) for p in training_set]).reshape(-1, 1)

    # # 3. Create a list of the length of each individual protein
    # lengths = [len(p) for p in training_set]

    # # 4. Pass both X and lengths to fit
    # model.fit(X_train, lengths=lengths)
    # model.fit(training_set)
    X = np.array(test_set).reshape(-1, 1)

    predicted_states = model.predict(X=[test_set])

    # print(predicted_states)
    return predicted_states
