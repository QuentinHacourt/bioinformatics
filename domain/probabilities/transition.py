from domain.state.state import State
from domain.state.state_role import StateRole


def _get_states_by_role(states: dict[str, State], role: StateRole) -> list[State]:
    result = [state for state in states.values() if state.role == role]

    result.sort(
        key=lambda state: (
            int(state.name.split("_")[-1]) if state.name[-1].isdigit() else 0
        )
    )
    return result


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total == 0:
        return raw

    return {k: v / total for k, v in raw.items()}


def transition(states: dict[str, State]) -> None:
    inner_ladder = _get_states_by_role(states, StateRole.INNER_LADDER)
    outer_ladder = _get_states_by_role(states, StateRole.OUTER_LADDER)
    arom_top = _get_states_by_role(states, StateRole.AROMATIC_BELT)[:2]  # first two
    arom_bottom = _get_states_by_role(states, StateRole.AROMATIC_BELT)[2:]  # last two
    tm_ext = _get_states_by_role(states, StateRole.TM_EXTERIOR)
    tm_int = _get_states_by_role(states, StateRole.TM_INTERIOR)

    n_term = states["inner_n_term"]
    c_term = states["inner_c_term"]
    globular = states["outer_globular"]

    # === N-terminal ===
    n_term.transitions = _normalize(
        {
            n_term.name: 1.0,
            inner_ladder[0].name: 1.0,
        }
    )

    # === Inner Ladder ===
    for i, state in enumerate(inner_ladder):
        if i < len(inner_ladder) - 1:
            state.transitions = _normalize(
                {
                    inner_ladder[i + 1].name: 1.0,
                }
            )
        else:
            state.transitions = _normalize(
                {
                    arom_top[0].name: 1.0,
                    c_term.name: 1.0,
                }
            )

        # === C-terminal ===
        c_term.transitions = {c_term.name: 1.0}

        # === Aromatic Belt ===
        for i, state in enumerate(arom_top):
            if i < len(arom_top) - 1:
                state.transitions = _normalize({arom_top[i + 1].name: 1.0})
            else:
                state.transitions = _normalize({tm_ext[0].name: 1.0})

        MIN_STRAND = 7
        MAX_STRAND = 17

        for i, ext_state in enumerate(tm_ext):
            position = i * 2 + 1
            next_int = tm_int[i] if i < len(tm_int) else None

            trans: dict[str, float] = {}

            if next_int:
                trans[next_int.name] = 1.0

            if position >= MIN_STRAND:
                trans[arom_bottom[0].name] = 1.0

            ext_state.transitions = _normalize(trans)

        for i, int_state in enumerate(tm_int):
            # TODO: swap interior and exterior indices
            position = i * 2 + 2
            next_ext = tm_ext[i + 1] if i + 1 < len(tm_ext) else None

            trans: dict[str, float] = {}

            if next_ext and position < MAX_STRAND:
                trans[next_ext.name] = 1.0

            if position >= MIN_STRAND:
                trans[arom_bottom[0].name] = 1.0

            int_state.transitions = _normalize(trans)

        # === Aromatic Belt ===
        for i, state in enumerate(arom_bottom):
            if i < len(arom_bottom) - 1:
                state.transitions = _normalize({arom_bottom[i + 1].name: 1.0})
            else:
                state.transitions = _normalize({outer_ladder[0].name: 1.0})

        # === Outer ladder ===
        for i, state in enumerate(outer_ladder):
            trans: dict[str, float] = {globular.name: 1.0}

            if i < len(outer_ladder) - 1:
                trans[outer_ladder[i + 1].name] = 1.0
            else:
                trans[arom_top[0].name] = 1.0

            state.transitions = _normalize(trans)

        globular.transitions = _normalize(
            {
                globular.name: 1.0,
                outer_ladder[-1].name: 1.0,
            }
        )
