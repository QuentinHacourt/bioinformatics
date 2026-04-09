from utils.csv_reader import collect_data


def main():
    print("Hello from bioinformatics project!")

    protein_names: list[str] = ["a", "b", "c"]

    data = collect_data(protein_names)

    print("====== DATA ======")
    print(data)
    # proteins = csv.read_aminoacid_sequence()
    # print("----- Proteins -----")
    # print(proteins)

    # betastrands = csv.read_annotations()
    # print("----- Betastrands -----")
    # print(betastrands)


if __name__ == "__main__":
    main()
