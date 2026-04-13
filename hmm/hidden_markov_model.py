import numpy as np
from hmmlearn import hmm
import numpy.typing as npt

# from domain.sequence import Protein


def test(
    # data: list[str],
    initial_distribution: list[float],
    transition_distribution: npt.NDArray[np.float64],
    observation_distribution: npt.NDArray[np.float64],
    test_sequence: list[int],
):
    states = 3

    model = hmm.CategoricalHMM(n_components=states, random_state=42, init_params="te")

    model.startprob_ = np.array(initial_distribution)

    m = np.random.rand(3, 3)
    A = transition_distribution
    model.transmat_ = A

    m = np.random.rand(3, 20)
    B = observation_distribution
    print("observation matrix:")
    print(B)
    model.emissionprob_ = B

    trials, simulated_states = model.sample(10000)

    # print(trials[:10])
    X_train = trials[: trials.shape[0] // 2]
    X_test = trials[trials.shape[0] // 2 :]

    model.fit(X_train)

    predicted_states = model.predict(X=[test_sequence])

    print(predicted_states)
