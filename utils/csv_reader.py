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


def remove_unlabeled(proteins: list[Protein]) -> list[Protein]:
    offset: int = 0
    for protein in proteins:
        for segment in protein.segments[:]:
            if segment.label == Label.UNLABELED or segment.label == Label.SIGNAL:
                begin: int = segment.begin - 1 - offset
                end: int = segment.end - offset

                # remove from original list
                protein.segments.remove(segment)

                # update sequence
                front: str = protein.sequence[:begin]
                back: str = protein.sequence[end:]
                protein.sequence = front + back

                # update sequence
                offset += end - begin
            else:
                segment.begin = segment.begin - offset
                segment.end = segment.end - offset

    return proteins


def find_sequence(protein_name: str) -> str:
    path = f"data/proteins/{protein_name}.xml"
    tree = et.parse(path)
    root = tree.getroot()
    namespaces = {"ns": "https://pdbtm.unitmp.org"}

    chain = root.find("ns:CHAIN", namespaces=namespaces)

    if chain is None:
        return ""

    sequence = chain.find("ns:SEQ", namespaces)
    if sequence is not None and sequence.text:
        return sequence.text.replace(" ", "")

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
