import utils.csv_reader as csv

def main():
    print("Hello from bioinformatics project!")
    proteins = csv.read_aminoacid_sequence()
    print(proteins)



if __name__ == "__main__":
    main()
