from domain.protein import Protein

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
    train = proteins[7:]
    test = proteins[:7]
    states = build_states()
    emission(states, proteins)
    transition(states)

    # # Train
    # trained_A, trained_B = cml_train(train, states, n_iter=200, learning_rate=0.01)

    # # ── Diagnostic ───────────────────────────────────────────────────────
    # idx_to_name, name_to_idx = build_index(states)
    # A = build_transition_matrix(states, name_to_idx)
    # B = build_emission_matrix(states, name_to_idx)
    # pi = build_initial_distribution(states, name_to_idx)

    # for protein in proteins:
    #     obs = encode_sequence(protein.sequence)
    #     state_path = annotation_to_state_sequence(protein.labels, name_to_idx)
    #     log_alpha_joint = forward_joint_log(obs, state_path, A, B, pi)
    #     log_p_joint = log_alpha_joint[-1, state_path[-1]]
    #     print(f"{protein.name}: log_p_joint={log_p_joint:.2f}")

    #     # Also print the first broken transition
    #     log_A = np.log(np.where(A > 0, A, 1e-300))
    #     for t in range(1, len(obs)):
    #         i = state_path[t - 1]
    #         j = state_path[t]
    #         if log_A[i, j] < -100:
    #             print(f"  BROKEN at t={t}: {idx_to_name[i]} → {idx_to_name[j]}")
    #             break

    # ── Training ─────────────────────────────────────────────────────────
    trained_A, trained_B = cml_train(train, states, n_iter=50, learning_rate=0.5)

    # Decode and evaluate on training set (self-consistency test)
    print("\n--- Self-consistency test (N-best) ---")
    evaluate(test, states, trained_A, trained_B, method="n_best")

    print("\n--- Self-consistency test (Viterbi) ---")
    evaluate(test, states, trained_A, trained_B, method="viterbi")

    # Decode a single protein in detail
    protein = proteins[0]
    labeling, log_prob = decode(protein, states, trained_A, trained_B)
    print(f"\n{protein.name} predicted:  {labeling}...")
    print(f"{protein.name} true:        {protein.labels}...")


def print_proteins(proteins, states, trained_A, trained_B):
    for protein in proteins:
        labeling, _ = decode(protein, states, trained_A, trained_B)
        label_printer(labeling, protein.labels, protein.name)


# def main():
#     print("Hello from bioinformatics project!")

#     proteins = collect_data()
#     states = build_states()
#     emission(states, proteins)
#     transition(states)

#     protein = proteins[0]

#     # TODO: lus over alle proteines
#     log_alpha, log_beta, index = run_forward_backward(protein, states)

#     T = len(protein.sequence)

#     T = len(protein.sequence)
#     log_p_obs = logsumexp(log_alpha[T - 1, :])
#     print(f"log P(obs): {log_p_obs:.4f}")

#     # CML objective for one protein
#     print("\n--- Single protein CML ---")
#     cml_log_probability(protein, states)

#     # CML objective over the whole training set
#     print("\n--- Full training set CML ---")
#     total_cml_objective(proteins, states)


if __name__ == "__main__":
    main()
