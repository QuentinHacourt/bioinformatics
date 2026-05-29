import numpy as np
from domain.data_wrapper import DataWrapper
from utils.label_printer import label_printer
from hmm.cml_train import cml_train

# from hmm.n_best import decode, evaluate
from utils.print_report import print_labels, print_report, print_viterbi
from hmm.viterbi import evaluate, decode


def main():
    dw = DataWrapper()
    benchmark(
        dw,
        5,
        30,
        0.05,
    )


def benchmark(dw: DataWrapper, samples, n, lr):
    labels = {}
    v_scores = {}
    v_overall = 0.0
    for _ in range(samples):
        dw.sample()

        dw.A, dw.B = cml_train(
            dw,
            n_iter=n,
            learning_rate=lr,
        )

        print("\n--- Self-consistency test (Viterbi) ---")
        viterbi, v_scores, v_overall = evaluate(dw.test, dw.states, dw.A, dw.B)
        protein = dw.proteins[0]
        labeling, log_prob = decode(protein, dw.states, dw.A, dw.B)
        labels[protein.name] = labeling

    print_report(len(dw.train), len(dw.test), lr, samples, n)
    print_viterbi(v_scores, v_overall)
    for p in dw.proteins:
        label, _ = decode(p, dw.states, dw.A, dw.B)
        print_labels(p, label)


def print_proteins(proteins, states, transitions, emissions):
    for protein in proteins:
        labeling, _ = decode(protein, states, transitions, emissions)
        label_printer(labeling, protein.labels, protein.name)


if __name__ == "__main__":
    main()
