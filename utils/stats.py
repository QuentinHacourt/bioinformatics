from domain.sequence import Label, Protein

def label_frequencies(proteins: list[Protein]) -> dict[Label, float]:
    dictionary: dict[Label, float] = {Label.INNER: 0, Label.OUTER: 0, Label.TRANS_MEMBRANE: 0}
    totalLength: int = 0
    for protein in proteins:
        for segment in protein.segments:
            if segment.label == Label.INNER \
            or segment.label == Label.OUTER \
            or segment.label == Label.TRANS_MEMBRANE:
                length: int = segment.end - segment.begin  + 1
                totalLength += length
                dictionary[segment.label] = dictionary[segment.label] + length
    print(dictionary)
    print(totalLength)
    dictionary = {key : value / totalLength for key, value in dictionary.items()}
    print(dictionary)
    return dictionary