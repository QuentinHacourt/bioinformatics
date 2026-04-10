import numpy as np
from hmmlearn import hmm


def test():
    states = 3

    model = hmm.CategoricalHMM(n_components=states, random_state=42)

    # TODO: berekenen aan de hand van data: alle lables tellen
    # en dan checken hoeveel er in, out en transmembraan zijn
    initial_distribution = np.array([1 / 3, 1 / 3, 1 / 3])
    model.startprob_ = initial_distribution

    m = np.random.rand(3, 3)
    # transition distribution: in, out or transmembraan (3x3 matrix)
    # TODO: calculate real probabilities
    A = m / m.sum(axis=1, keepdims=True)
    print("transition matrix:")
    print(A)
    model.transmat_ = A

    m = np.random.rand(3, 20)
    # observation: the amino acid sequence
    B = m / m.sum(axis=1, keepdims=True)
    print("observation matrix:")
    print(B)
    model.emissionprob_ = B

    # TODO: haal trials uit dataset (csv) en verdeel dataset in train en testing
    trials, simulated_states = model.sample(10000)

    print(trials[:10])
    X_train = trials[: trials.shape[0] // 2]
    X_test = trials[trials.shape[0] // 2 :]

    model.fit(X_train)

    # TODO: observation matrix
    exam_observations = [[0, 1, 2]]
    predicted_states = model.predict(X=[[0, 1, 2]])
    print(
        "Predict the Hidden State Transitions that were being the exam scores OK, Fail, Perfect: \n 0 -> Tired , "
        "1 -> Happy"
    )

    print(predicted_states)


test()
