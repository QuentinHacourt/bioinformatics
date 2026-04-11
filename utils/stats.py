from domain.sequence import Label, Protein
import numpy as np
import numpy.typing as npt


def label_frequencies(proteins: list[Protein]) -> dict[Label, float]:
    dictionary: dict[Label, float] = {
        Label.INNER: 0,
        Label.OUTER: 0,
        Label.TRANS_MEMBRANE: 0,
    }
    totalLength: int = 0
    for protein in proteins:
        for segment in protein.segments:
            if (
                segment.label == Label.INNER
                or segment.label == Label.OUTER
                or segment.label == Label.TRANS_MEMBRANE
            ):
                length: int = segment.end - segment.begin + 1
                totalLength += length
                dictionary[segment.label] = dictionary[segment.label] + length
    print(dictionary)
    print(totalLength)
    dictionary = {key: value / totalLength for key, value in dictionary.items()}
    print(dictionary)
    return dictionary


def transition_distribution(proteins: list[Protein]) -> npt.NDArray[np.float64]:
    m: npt.NDArray[np.float64] = np.zeros((3, 3))
    totalcount: int = 0

    for protein in proteins:
        totalcount += len(protein.sequence)

        for i in range(0, len(protein.sequence) - 1):
            from_label: Label = find_label(i, protein)
            to_label: Label = find_label(i + 1, protein)
            from_index: int = label_to_index(from_label)
            to_index: int = label_to_index(to_label)
            m[from_index][to_index] += 1

    #transition_matrix = m / totalcount
    for i in range(0, 3):
        m[i] = m[i]/m[i].sum()
    return m


def observation_distribution(proteins: list[Protein]) -> npt.NDArray[np.float64]:
    m: npt.NDArray[np.float64] = np.zeros((3, 25))
    totalcount: dict[str, int] = {}

    for protein in proteins:
        for i in range(0, len(protein.sequence) - 1):
            character: str = protein.sequence[i]
            label: Label = find_label(i, protein)
            label_index = label_to_index(label)
            char_index = char_to_index(character)
            m[label_index][char_index] += 1

            if character in totalcount:
                totalcount[character] += 1
            else:
                totalcount[character] = 1

    print(totalcount)
    print(m)

    for i in range(0, 3):
        m[i] = m[i]/m[i].sum()


    # for k in totalcount:
    #     i = char_to_index(k)
    #     m[:, i] = m[:, i] / totalcount[k]

    print(m)
    return m


def find_label(index, protein: Protein) -> Label:
    for segment in protein.segments:
        if index <= segment.end and index >= segment.begin:
            return segment.label

    return Label.PANIEK


def label_to_index(label: Label) -> int:
    match label:
        case Label.INNER:
            return 0
        case Label.TRANS_MEMBRANE:
            return 1
        case Label.OUTER:
            return 2
        case _:
            return -1


def char_to_index(character: str) -> int:
    return ord(character) - ord("A")
