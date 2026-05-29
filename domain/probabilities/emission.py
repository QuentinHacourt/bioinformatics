from collections import defaultdict
from domain.protein import Protein
from domain.state.state import State
import domain.amino_acid as aa
from domain.state.state_role import StateRole

ROLE_TO_TIE_GROUP: dict[StateRole, str] = {
    StateRole.N_TERMINAL: "tail",
    StateRole.C_TERMINAL: "tail",
    StateRole.INNER_LADDER: "inner_ladder",
    StateRole.OUTER_LADDER: "outer_ladder",
    StateRole.GLOBULAR: "outer_globular",
    StateRole.AROMATIC_BELT: "aromatic_belt",
    StateRole.TM_EXTERIOR: "tm_exterior",
    StateRole.TM_INTERIOR: "tm_interior",
}

LABEL_TO_ROLE: dict[str, list[StateRole]] = {
    "I": [
        StateRole.INNER_LADDER,
        StateRole.N_TERMINAL,
        StateRole.C_TERMINAL],
    "O": [StateRole.OUTER_LADDER, StateRole.GLOBULAR],
    "T": [
        StateRole.AROMATIC_BELT,
        StateRole.TM_EXTERIOR,
        StateRole.TM_INTERIOR,
    ],
}


def _count_amino_acid_per_group(proteins: list[Protein]) -> dict[str, dict[str, int]]:

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for protein in proteins:
        sequence = protein.sequence
        labels = protein.labels

        for amino_acid, label in zip(sequence, labels):
            roles = LABEL_TO_ROLE.get(label, [])
            for role in roles:
                group = ROLE_TO_TIE_GROUP[role]
                counts[group][amino_acid] += 1
    return counts

def emission(
    states: list[State], proteins: list[Protein], pseudocount: float = 1.0
) -> None:
    raw_counts = _count_amino_acid_per_group(proteins)

    emissions: dict[str, dict[str, float]] = {}

    all_tie_groups = set(ROLE_TO_TIE_GROUP.values())

    for group in all_tie_groups:
        amino_acid_counts = raw_counts.get(group, {})
        total = (
            sum(amino_acid_counts.get(amino_acid, 0) for amino_acid in aa.AMINO_ACIDS)
            + 20 * pseudocount
        )

        emissions[group] = {
            amino_acid: (amino_acid_counts.get(amino_acid, 0) + pseudocount) / total
            for amino_acid in aa.AMINO_ACIDS
        }

    for state in states:
        group = state.tie_group if state.tie_group else state.name
        if group in emissions:
            state.emissions = emissions[group]
