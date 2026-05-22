AMINO_ACIDS: list[str] = list('ACDEFGHIKLMNPQRSTVWY')
AMINO_ACID_TO_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NB_OF_AMINO_ACID: int = len(AMINO_ACIDS)

def find_index(amino_acid: str) -> int:
    return AMINO_ACID_TO_INDEX[amino_acid]

def find_indices(aa_seq: str) -> list[int]:
    return [find_index(i) for i in aa_seq]