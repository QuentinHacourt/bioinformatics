import xml.etree.ElementTree as et
import os
from domain.protein import Protein


def _find_file_names(path) -> list[str]:
    names: list[str] = []
    obj = os.scandir(path)
    for entry in obj:
        if entry.is_file() and entry.name.endswith(".xml"):
            names.append(entry.name)

    return names


def _find_sequence(root) -> str:
    namespaces = {"ns": "https://pdbtm.unitmp.org"}

    chain = root.find("ns:CHAIN", namespaces=namespaces)

    if chain is None:
        return ""

    sequence = chain.find("ns:SEQ", namespaces)

    if sequence is not None and sequence.text:
        return sequence.text.replace(" ", "")

    return ""


def _find_labels(root, sequence_length: int) -> str:
    namespaces = {"ns": "https://pdbtm.unitmp.org"}

    chain = root.find("ns:CHAIN", namespaces=namespaces)

    if chain is None:
        return ""

    regions = chain.findall("ns:REGION", namespaces=namespaces)

    labels = " " * sequence_length

    for region in regions:
        begin: int = int(region.get("seq_beg") or -1) - 1
        end: int = int(region.get("seq_end") or -1) - 1
        length = end - begin + 1
        # 1 = in
        # 2 = out
        # B = tm
        # U = unlabled
        # I = beta barrel inside -> tm

        type: str = str(region.get("type"))
        symbol = ""

        match type:
            case "1":
                symbol = "I"
            case "2":
                symbol = "O"
            case "B":
                symbol = "T"
            case "U":
                symbol = "U"
            case "I":
                symbol = "T"
            case _:
                raise ValueError("Error: Encountered an unknown label")

        labels = labels[:begin] + (symbol * length) + labels[begin + length :]

    labels_list = list(labels)

    for i in range(len(labels_list)):
        if labels_list[i] == " " and i > 0:
            labels_list[i] = labels_list[i - 1]
        elif labels_list[i] == " " and i == 0:
            labels_list[i] = "U"

    labels = "".join(labels_list)

    return labels


def _read_file(path: str, file_name: str) -> Protein:
    full_path: str = path + file_name

    name: str = file_name[:-4]
    sequence: str = ""
    labels: str = ""

    tree = et.parse(full_path)
    root = tree.getroot()

    sequence = _find_sequence(root)
    labels = _find_labels(root, len(sequence))

    if len(labels) != len(sequence):
        raise ValueError(
            "length of labels should be the same as the length of the sequence"
        )

    sequence, labels = _remove_unlabeled(sequence, labels)

    return Protein(name, sequence, labels)


def collect_data(path: str = "data/proteins/") -> list[Protein]:
    file_names: list[str] = _find_file_names(path)
    proteins: list[Protein] = []

    for file_name in file_names:
        proteins.append(_read_file(path, file_name))

    return proteins


def _remove_unlabeled(sequence, labels) -> tuple[str, str]:
    filtered = [(seq, label) for seq, label in zip(sequence, labels) if label != "U"]

    if not filtered:
        return "", ""

    seq_tuple, label_tuple = zip(*filtered)
    return "".join(seq_tuple), "".join(label_tuple)
