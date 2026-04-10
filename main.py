from utils.csv_reader import collect_data, find_protein_names
from domain.sequence import Protein
from utils.stats import label_frequencies

def main():
    print("Hello from bioinformatics project!")

    protein_names: list[str] = find_protein_names("data/proteins/")

    data: list[Protein] = collect_data(protein_names)

    print("====== DATA ======")
    print(data)
    # proteins = csv.read_aminoacid_sequence()
    # print("----- Proteins -----")
    # print(proteins)

    # betastrands = csv.read_annotations()
    # print("----- Betastrands -----")
    # print(betastrands)

    print("===== STATS =====")
    d = label_frequencies(data)

if __name__ == "__main__":
    main()
