amino_acids: list[str] = list('ACDEFGHIKLMNPQRSTVWY')
amino_acid_to_index: dict[str, int] = {aa: i for i, aa in enumerate(amino_acids)}

def find_index(amino_acid: str) -> int:
    return amino_acid_to_index[amino_acid]

def find_indices(aa_seq: str) -> list[int]:
    return [find_index(i) for i in aa_seq]