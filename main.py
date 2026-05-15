from domain.protein import Protein

# from domain.amino_acid import find_indices, find_index
# import numpy as np
from scipy.special import logsumexp

# from utils.stats import (
#     observation_distribution,
#     transition_distribution,
#     initial_distribution,
# )
# from hmm.hidden_markov_model import test


from domain.probabilities.emission import emission

from domain.probabilities.joint_probability import (
    cml_log_probability,
    total_cml_objective,
)
from domain.probabilities.transition import transition
from domain.state.util import build_states, count_states

from hmm.forward_backward import run_forward_backward

# from utils.xml_reader import collect_data, remove_unlabeled


from utils.xml_reader import collect_data
from pprint import pprint


def main():
    print("Hello from bioinformatics project!")

    proteins = collect_data()
    states = build_states()
    emissions = emission(states, proteins)
    transition(states)

    protein = proteins[0]

    log_alpha, log_beta, index = run_forward_backward(protein, states)

    T = len(protein.sequence)

    T = len(protein.sequence)
    log_p_obs = logsumexp(log_alpha[T - 1, :])
    print(f"log P(obs): {log_p_obs:.4f}")

    # CML objective for one protein
    print("\n--- Single protein CML ---")
    cml_log_probability(protein, states)

    # CML objective over the whole training set
    print("\n--- Full training set CML ---")
    total_cml_objective(proteins, states)


if __name__ == "__main__":
    main()
