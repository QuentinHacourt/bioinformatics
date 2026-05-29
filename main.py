import numpy as np
from domain.data_wrapper import DataWrapper
from utils.label_printer import label_printer
from hmm.cml_train import cml_train
from hmm.n_best import decode, evaluate
from domain.probabilities.joint_probability import (
    annotation_to_state_sequence,
    encode_sequence,
)


def main():
    dw = DataWrapper()
    benchmark(
        dw,
        [5],
        [200],
        [0.05, 0.06, 0.07, 0.8, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15],
    )


def benchmark(dw: DataWrapper, samples, n_iter, learning_rates):

    print(f"+++++++ TEST {dw.proteins[0].name} +++++++++")
    test_state_sequence(
        dw.proteins[0].labels,
        dw.name_to_idx,
        dw.idx_to_name,
        encode_sequence(dw.proteins[0].sequence),
        dw.log_A,
        dw.log_B,
        dw.log_pi,
    )

    for r in samples:
        for n in n_iter:
            for lr in learning_rates:
                for i in range(r):
                    dw.sample()
                    dw.A, dw.B = cml_train(
                        dw,
                        n_iter=n,
                        learning_rate=lr,
                    )

                    print("\n--- Self-consistency test (N-best) ---")
                    n_best = evaluate(dw.test, dw.states, dw.A, dw.B, method="n_best")

                    print("\n--- Self-consistency test (Viterbi) ---")
                    viterbi = evaluate(dw.test, dw.states, dw.A, dw.B, method="viterbi")
                    protein = dw.proteins[0]
                    labeling, log_prob = decode(protein, dw.states, dw.A, dw.B)
                    print(f"\n{protein.name} predicted:  {labeling}...")
                    print(f"{protein.name} true:        {protein.labels}...")

                    with open(
                        f"output/{n}-{r}-{lr}-{i}.txt", "w", encoding="utf-8"
                    ) as f:
                        f.writelines(n_best)
                        f.writelines(viterbi)
                        f.write(f"\n{protein.name} predicted:  {labeling}...")
                        f.write(f"\n{protein.name} true:        {protein.labels}...")


def print_proteins(proteins, states, transitions, emissions):
    for protein in proteins:
        labeling, _ = decode(protein, states, transitions, emissions)
        label_printer(labeling, protein.labels, protein.name)


def test_state_sequence(labels, name_to_idx, idx_to_name, obs, log_A, log_B, log_pi):
    test = annotation_to_state_sequence(labels, name_to_idx, obs, log_A, log_B, log_pi)
    res = [idx_to_name[i].name for i in test]
    print(res)


if __name__ == "__main__":
    main()
