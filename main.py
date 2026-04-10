from utils.csv_reader import collect_data, find_protein_names, remove_unlabeled
from domain.sequence import Protein
import numpy as np
from utils.stats import (
    label_frequencies,
    observation_distribution,
    transition_distribution,
)


def main():
    print("Hello from bioinformatics project!")

    # protein_names: list[str] = find_protein_names("data/proteins/")

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

    protein_names = ["2FCP"]
    data = collect_data(protein_names=protein_names)
    data = remove_unlabeled(data)

    # TODO: het klopt langs geen kant, som rij moet 1 zijn
    tm = transition_distribution(data)
    # TODO: het klopt langs geen kant, som rij moet 1 zijn
    od = observation_distribution(data)

    print(tm.sum(axis=1))


if __name__ == "__main__":
    main()
