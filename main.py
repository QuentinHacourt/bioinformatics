import utils.csv_reader as csv


def main():
    print("Hello from bioinformatics project!")
    proteins = csv.read_aminoacid_sequence()
    print("----- Proteins -----")
    print(proteins)

    betastrands = csv.read_annotations()
    print("----- Betastrands -----")
    print(betastrands)


if __name__ == "__main__":
    main()
