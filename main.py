from domain.protein import Protein
import random
# from domain.amino_acid import find_indices, find_index
import numpy as np
from scipy.special import logsumexp

# from utils.stats import (
#     observation_distribution,
#     transition_distribution,
#     initial_distribution,
# )
# from hmm.hidden_markov_model import test


from domain.probabilities.emission import emission

from domain.probabilities.joint_probability import (
    annotation_to_state_sequence,
    cml_log_probability,
    total_cml_objective,
)
from domain.probabilities.transition import transition
from domain.state.util import build_states, count_states

from hmm.forward_backward import (
    build_emission_matrix,
    build_index,
    build_initial_distribution,
    build_transition_matrix,
    encode_sequence,
    run_forward_backward,
)

# from utils.xml_reader import collect_data, remove_unlabeled


from utils.label_printer import label_printer
from utils.xml_reader import collect_data
from pprint import pprint
from hmm.cml_train import cml_train, forward_joint_log
from hmm.n_best import decode, evaluate


def main():
    proteins = collect_data()
    states = build_states()
    emission(states, proteins)
    transition(states)

    trained_A = None
    trained_B = None
    
    for r in [5,10,15]:
        for n in [50, 100,200,500]:
            for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
                for i in range(r):
                    train = random.sample(proteins, 7)
                    test = [item for item in proteins if item not in train]
                    trained_A, trained_B = cml_train(train, states, n_iter=n, learning_rate=lr, A=trained_A, B=trained_B)

                    print("\n--- Self-consistency test (N-best) ---")
                    n_best = evaluate(test, states, trained_A, trained_B, method="n_best")

                    print("\n--- Self-consistency test (Viterbi) ---")
                    viterbi = evaluate(test, states, trained_A, trained_B, method="viterbi")
                    protein = proteins[0]
                    labeling, log_prob = decode(protein, states, trained_A, trained_B)
                    print(f"\n{protein.name} predicted:  {labeling}...")
                    print(f"{protein.name} true:        {protein.labels}...")

                    with open(f"output/{n}-{r}-{lr}-{i}.txt", "w", encoding="utf-8") as f:
                        f.writelines(n_best)
                        f.writelines(viterbi)
                        f.write(f"\n{protein.name} predicted:  {labeling}...")
                        f.write(f"\n{protein.name} true:        {protein.labels}...")


def print_proteins(proteins, states, trained_A, trained_B):
    for protein in proteins:
        labeling, _ = decode(protein, states, trained_A, trained_B)
        label_printer(labeling, protein.labels, protein.name)

if __name__ == "__main__":
    main()
