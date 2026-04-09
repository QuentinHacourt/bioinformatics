# import pandas as pd
# import numpy as np
# from domain.beta_strand import Interval, BetaStrand

# from domain.protein import Protein
import xml.etree.ElementTree as xml

from domain.sequence import Protein, Segment


def normalize_data() -> None:
    print("lmao")


def collect_data(protein_names: list[str]) -> list[Protein]:
    proteins: list[Protein] = []

    for protein_name in protein_names:
        sequence: str = find_sequence(protein_name)
        segments: list[Segment] = find_segments(protein_name)

        protein = Protein(name=protein_name, sequence=sequence, segments=segments)
        proteins.append(protein)

    return proteins


def find_sequence(protein_name: str) -> str:
    return ""


def find_segments(protein_name: str) -> list[Segment]:
    return []


# def read_aminoacid_sequence():
#     pdb = pd.read_csv("data/data.csv", delimiter=";")
#     pdb["data"] = pdb["data"].apply(lambda s: s.replace(" ", ""))
#     pdb = pdb.drop(columns=["Betastrands"])
#     proteins = df_to_protein(pdb)
#     return proteins


# def read_annotations():
#     annotations = pd.read_csv("data/12859_2003_145_MOESM1_ESM.txt", delimiter="\t")

#     # annotations = annotations.drop(
#     #     columns=["PDB", "Unnamed: 2", "Unnamed: 4", "PREDICTED"]
#     # )

#     betastrands = df_to_beta_strand(annotations)
#     return betastrands


# def df_to_protein(df: pd.DataFrame):
#     proteins = []
#     for row in df.itertuples():
#         proteins.append(Protein(name=row[1], code=row[2], sequence=row[3]))
#     return proteins


# def df_to_beta_strand(df: pd.DataFrame):
#     betastrands = []

#     for row in df.itertuples():
#         print(row)
#         bounds = row[4].split("-")
#         low = bounds[0]
#         up = bounds[1]
#         interval = Interval(low=low, up=up)
#         betastrand = BetaStrand(name=row[1], transmembrane=interval)
#         betastrands.append(betastrand)

#     return betastrands
