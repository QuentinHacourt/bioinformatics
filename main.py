from utils.csv_reader import collect_data, find_protein_names, remove_unlabeled
from domain.sequence import Protein
import numpy as np
from utils.stats import (
    char_to_index,
    label_frequencies,
    observation_distribution,
    transition_distribution,
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

    id = label_frequencies(data)

    tm = transition_distribution(data)

    od = observation_distribution(data)

    test_protein_names = ["2FCP"]
    test_data = collect_data(test_protein_names)

    # print(tm.sum(axis=1))
    # print(od.sum(axis=1))

    t = test_data[0].sequence

    t = [char_to_index(i) for i in t]

    test(id, tm, od, t)


if __name__ == "__main__":
    main()
