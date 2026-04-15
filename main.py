from utils.csv_reader import collect_data, find_protein_names, remove_unlabeled
from domain.sequence import Protein
from domain.amino_acid import find_indices, find_index
import numpy as np
from utils.stats import (
    observation_distribution,
    transition_distribution,
    initial_distribution,
)
from hmm.hidden_markov_model import test


def main():
    print("Hello from bioinformatics project!")

    protein_names: list[str] = find_protein_names("data/proteins/")

    # data: list[Protein] = collect_data(protein_names)

    # print("====== DATA ======")
    # print(data)
    # # proteins = csv.read_aminoacid_sequence()
    # # print("----- Proteins -----")
    # # print(proteins)

    # # betastrands = csv.read_annotations()
    # # print("----- Betastrands -----")
    # # print(betastrands)

    # print("===== STATS =====")
    # d = label_frequencies(data)

    data = collect_data(protein_names=protein_names)
    data = remove_unlabeled(data)

    id = initial_distribution(data)

    tm = transition_distribution(data)
    print(tm)

    od = observation_distribution(data)

    test_protein_names = ["2FCP"]
    test_data = collect_data(test_protein_names)
    test_data = remove_unlabeled(test_data)

    # print(tm.sum(axis=1))
    # print(od.sum(axis=1))

    training_set = []

    for protein in data:
        training_set.append([find_index(i) for i in protein.sequence])

    test_set = test_data[0].sequence

    test_set = [find_index(i) for i in test_set]

    # print(training_set)

    print(id)

    res = test(id, tm, od, test_set)
    print("ORIGINAL MATRIX")
    print(res)

    tm_debug = np.array([[0.5, 0.5, 0], [0.2, 0.6, 0.2], [0, 0.5, 0.5]])

    res = test(id, tm_debug, od, test_set)
    print("DEBUG MATRIX")
    print(res)


if __name__ == "__main__":
    main()
