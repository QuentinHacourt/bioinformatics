import random
from pprint import pprint

from domain.probabilities.emission import emission

from domain.probabilities.transition import transition
from domain.state.util import build_states

from hmm.forward_backward import (
    build_emission_matrix,
    build_index,
    build_initial_distribution,
    build_transition_matrix,
    encode_sequence
)
from utils.label_printer import label_printer
from utils.xml_reader import collect_data
from hmm.cml_train import cml_train, build_tied_groups
from hmm.n_best import decode, evaluate

from domain.probabilities.joint_probability import annotation_to_state_sequence

def main():
    proteins = collect_data()
    states = build_states()

    emission(states, proteins)
    transition(states)

    benchmark(states, 
              proteins, 
              [5], 
              [50], 
              [1.0])

def benchmark(states, proteins, samples, n_iter, learning_rates):
    idx_to_name, name_to_idx = build_index(states)
    tied_groups              = build_tied_groups(states, name_to_idx)
    emissions                = build_emission_matrix(states, name_to_idx)
    transitions              = build_transition_matrix(states, name_to_idx)
    initials                 = build_initial_distribution(states, 
                                                          name_to_idx, 
                                                          proteins)
    
    print(f"+++++++ TEST {proteins[0].name} +++++++++")
    test_state_sequence(proteins[0].labels, name_to_idx, idx_to_name, encode_sequence(proteins[0].sequence), transitions, emissions, initials)

    # for r in samples:
    #     for n in n_iter:
    #         for lr in learning_rates:
    #             for i in range(r):
    #                 train = random.sample(proteins, 7)
    #                 test = [item for item in proteins if item not in train]
    #                 transitions, emissions = cml_train(train,
    #                                                   transitions,
    #                                                   emissions,
    #                                                   initials,
    #                                                   name_to_idx,
    #                                                   tied_groups,
    #                                                   n_iter=n, learning_rate=lr)

    #                 print("\n--- Self-consistency test (N-best) ---")
    #                 n_best = evaluate(test, 
    #                                   states, 
    #                                   transitions, 
    #                                   emissions, 
    #                                   method="n_best")

    #                 print("\n--- Self-consistency test (Viterbi) ---")
    #                 viterbi = evaluate(test, 
    #                                    states, 
    #                                    transitions, 
    #                                    emissions, 
    #                                    method="viterbi")
    #                 protein = proteins[0]
    #                 labeling, log_prob = decode(protein, 
    #                                             states, 
    #                                             transitions, 
    #                                             emissions)
    #                 print(f"\n{protein.name} predicted:  {labeling}...")
    #                 print(f"{protein.name} true:        {protein.labels}...")

    #                 with open(f"output/{n}-{r}-{lr}-{i}.txt", "w", encoding="utf-8") as f:
    #                     f.writelines(n_best)
    #                     f.writelines(viterbi)
    #                     f.write(f"\n{protein.name} predicted:  {labeling}...")
    #                     f.write(f"\n{protein.name} true:        {protein.labels}...")


def print_proteins(proteins, states, transitions, emissions
):
    for protein in proteins:
        labeling, _ = decode(protein, states, transitions, emissions
)
        label_printer(labeling, protein.labels, protein.name)


def test_state_sequence(labels, name_to_idx, idx_to_name, obs, A, B, pi):
    test = annotation_to_state_sequence(labels, name_to_idx, obs, A, B, pi)
    res = [idx_to_name[i].name for i in test]
    print(res)

if __name__ == "__main__":
    main()
