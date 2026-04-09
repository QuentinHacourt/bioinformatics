import pandas as pd
import numpy as np
import domain.beta_strand as b
from domain.protein import Protein

def read_aminoacid_sequence():
    pdb = pd.read_csv("data/data.csv", delimiter=";")
    pdb['data'] = pdb['data'].apply(lambda s: s.replace(' ', ''))
    pdb = pdb.drop(columns=['Betastrands'])
    proteins = df_to_protein(pdb)
    return proteins

def read_annotations():
    annotations = pd.read_csv("data/12859_2003_145_MOESM1_ESM.txt", delimiter='\t')
    annotations = annotations.drop(columns=['PDB', 'Unnamed: 2', 'Unnamed: 4', 'PREDICTED'])

def df_to_protein(df: pd.DataFrame):
    lst = []
    for row in df.itertuples():
        lst.append(Protein(
            name=row[1],
            code=row[2],
            sequence=row[3]
        ))
    return lst
