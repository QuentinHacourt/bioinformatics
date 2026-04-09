import xml.etree.ElementTree as et
from domain.sequence import Protein, Segment, Label
import os


def find_protein_names(path):
    names: list[str] = []
    obj = os.scandir(path)
    for entry in obj:
        if entry.is_file() and entry.name.endswith(".xml"):
            names.append(entry.name[:-4])

    return names


def collect_data(protein_names: list[str]) -> list[Protein]:
    proteins: list[Protein] = []

    for protein_name in protein_names:
        sequence: str = find_sequence(protein_name)
        segments: list[Segment] = find_segments(protein_name)

        protein = Protein(name=protein_name, sequence=sequence, segments=segments)
        proteins.append(protein)

    return proteins


def find_sequence(protein_name: str) -> str:
    path = f"data/proteins/{protein_name}.xml"
    tree = et.parse(path)
    root = tree.getroot()
    namespaces = {"ns": "https://pdbtm.unitmp.org"}

    chain = root.find("ns:CHAIN", namespaces=namespaces)

    if chain is None:
        return ""

    print(chain)
    sequence = chain.find("ns:SEQ", namespaces)
    if sequence is not None and sequence.text:
        return sequence.text.strip()

    return ""


def find_segments(protein_name: str) -> list[Segment]:
    path = f"data/proteins/{protein_name}.xml"
    tree = et.parse(path)
    root = tree.getroot()
    namespaces = {"ns": "https://pdbtm.unitmp.org"}

    chain = root.find("ns:CHAIN", namespaces=namespaces)

    if chain is None:
        return []

    regions = chain.findall("ns:REGION", namespaces=namespaces)

    segments: list[Segment] = []

    for region in regions:
        begin: int = int(region.get("seq_beg") or -1)
        end: int = int(region.get("seq_end") or -1)
        # 1 = in
        # 2 = out
        # B = tm
        # U = unlabled
        # I = beta barrel inside -> tm

        type: str = str(region.get("type"))
        label: Label

        match type:
            case "1":
                label = Label.INNER
            case "2":
                label = Label.OUTER
            case "B":
                label = Label.TRANS_MEMBRANE
            case "U":
                label = Label.UNLABELED
            case "I":
                label = Label.TRANS_MEMBRANE
            case _:
                label = Label.PANIEK
                print("paniek.jpg")

        segment: Segment = Segment(label=label, begin=begin, end=end)
        segments.append(segment)

    return segments


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
