# grootte training set/test set
# learning rate
# aantal samples
# aantal iteraties
from domain.protein import Protein

FILENAME = "output/report.txt"


def print_report(
    training_size, test_size, learning_rate, samples, iterations_per_samples
):

    report = f"""==================== Report ====================
    training set size:          {training_size}
    testing set size:           {test_size}
    learning rate:              {learning_rate:.2f}
    samples:                    {samples}
    iterations (per sample):    {iterations_per_samples}
    """
    print(report)
    with open(FILENAME, "w", encoding="utf-8") as f:
        f.write(report)


def print_viterbi(data: dict[str, int], overall):
    report = "\n=== Self-consistency test (Viterbi) ===\n"
    for protein, score in data.items():
        report += f"{protein}       accuracy    {score:.2f}%\n"

    report += f"Overall score:         {overall:.2f}%\n"
    print(report)
    with open(FILENAME, "a", encoding="utf-8") as f:
        f.write(report)


def print_labels(protein: Protein, labels):
    report = f"""{protein.name}
    predicted:  {labels}
    actual:     {protein.labels}
    """

    print(report)
    with open(FILENAME, "a", encoding="utf-8") as f:
        f.write(report)
