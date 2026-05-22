import domain.amino_acid as aa
from domain.state.state import State
from domain.state.state_role import StateRole
from domain.state.state_type import StateType

from collections import Counter, defaultdict


def build_states() -> list[State]:
    states:list[State] = []

    def add(name, stype, role, tie=None, absorbing=False):
        states.append(State(name, stype, role, tie, absorbing))

    # === Inner Loop Submodel ===
    # N-terminal
    add("inner_n_term", StateType.INNER_LOOP, StateRole.N_TERMINAL, tie="tail")

    # Ladder
    for i in range(12):  # 12 ladder states
        add(
            f"inner_ladder_{i}",
            StateType.INNER_LOOP,
            StateRole.INNER_LADDER,
            tie="inner_ladder",
        )

    # C-terminal
    add("inner_c_term", StateType.INNER_LOOP, StateRole.C_TERMINAL, tie="tail")

    # === Transmembrane Submodel ===
    # Aromatic Belt Top (extracellular)
    for i in range(2):
        add(
            f"tm_aromatic_top_{i}",
            StateType.TRANSMEMBRANE,
            StateRole.AROMATIC_BELT,
            tie="aromatic_belt",
        )

    # 14 exterior states
    for i in range(14):
        add(
            f"tm_exterior_{i}",
            StateType.TRANSMEMBRANE,
            StateRole.TM_EXTERIOR,
            tie="tm_exterior",
        )

    # 16 interior states
    for i in range(16):
        add(
            f"tm_interior_{i}",
            StateType.TRANSMEMBRANE,
            StateRole.TM_INTERIOR,
            tie="tm_interior",
        )

    # Aromatic Belt Bottom (periplasmic)
    for i in range(2):
        add(
            f"tm_aromatic_bottom_{i}",
            StateType.TRANSMEMBRANE,
            StateRole.AROMATIC_BELT,
            tie="aromatic_belt",
        )

    # === Outer Loop Submodel ===
    add("outer_globular", StateType.OUTER_LOOP, StateRole.GLOBULAR, tie=None)

    for i in range(12):
        add(
            f"outer_ladder_{i}",
            StateType.OUTER_LOOP,
            StateRole.OUTER_LADDER,
            tie="outer_ladder",
        )
    return states


def count_states(states: dict[str, State]) -> None:
    counts = Counter(s.state_type for s in states.values())
    print(f"Inner loop:    {counts[StateType.INNER_LOOP]:>3}")
    print(f"TM strand:     {counts[StateType.TRANSMEMBRANE]:>3}")
    print(f"Outer loop:    {counts[StateType.OUTER_LOOP]:>3}")
    print(f"Total:         {len(states):>3}")

    tie_groups = Counter(s.tie_group for s in states.values() if s.tie_group)
    print("\nTie groups (shared emission parameters):")

    for group, n in sorted(tie_groups.items()):
        print(f"    {group:<20} {n} states")


def build_tie_groups(states: list[State]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[State]] = {}

    for state in states:
        group_name = state.tie_group if state.tie_group else state.name
        groups.setdefault(group_name, []).append(state)

    group_emissions: dict[str, dict[str, float]] = {}
    for group_name, group_states in groups.items():
        shared_emissions = {amino_acid: 1.0 / 20 for amino_acid in aa.AMINO_ACIDS}
        group_emissions[group_name] = shared_emissions

        for state in group_states:
            state.emissions = shared_emissions
    return group_emissions


def verify_tying(states: list[State]) -> None:
    groups: dict[str, list[State]] = defaultdict(list)
    for state in states:
        group = state.tie_group if state.tie_group else state.name
        groups[group].append(state)

    print("Tie group verification:")
    for group, group_states in sorted(groups.items()):
        if len(group_states) == 1:
            print(f"   {group:<20} 1 state (untied)")
            continue

        first = group_states[0].emissions
        all_shared = all(s.emissions is first for s in group_states[1:])
        status = "OK shared" if all_shared else "FAIL independent"
        print(f"    {group:<20} {len(group_states)} states - {status}")
