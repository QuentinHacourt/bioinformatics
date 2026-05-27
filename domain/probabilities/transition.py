from domain.state.state import State
from domain.state.state_role import StateRole


def _get_states_by_role(states: list[State], role: StateRole) -> list[State]:
    result = [state for state in states if state.role == role]

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

def transition(states: list[State]) -> None:
    inner_ladder = _get_states_by_role(states, StateRole.INNER_LADDER)
    outer_ladder = _get_states_by_role(states, StateRole.OUTER_LADDER)
    arom = _get_states_by_role(states, StateRole.AROMATIC_BELT)
    arom_top = [state for state in arom if "tm_aromatic_top" in state.name]
    arom_bottom = [state for state in arom if "tm_aromatic_bottom" in state.name]
    tm_ext = _get_states_by_role(states, StateRole.TM_EXTERIOR)
    tm_int = _get_states_by_role(states, StateRole.TM_INTERIOR)

    n_term = [state for state in states if state.name == "inner_n_term"][0]
    c_term = [state for state in states if state.name == "inner_c_term"][0]
    globular = [state for state in states if state.name == "outer_globular"][0]

    MIN_STRAND = 7
    MAX_STRAND = 17

    # === N-terminal ===
    n_term.transitions = _normalize(
        {
            n_term.name: 1.0,
            inner_ladder[0].name: 1.0,
            inner_ladder[1].name: 1.0,
            inner_ladder[2].name: 1.0,
            inner_ladder[3].name: 1.0,
            inner_ladder[4].name: 1.0,
            inner_ladder[5].name: 1.0,
            arom_top[0].name: 1.0
        }
    )

    # === C-terminal ===
    c_term.transitions = {c_term.name: 1.0}

    # === Inner Ladder ===
    for i, state in enumerate(inner_ladder):
        match i:
            case 0:
                state.transitions = _normalize({inner_ladder[1].name: 1.0})
            case 1:
                state.transitions = _normalize({inner_ladder[2].name: 1.0})
            case 2:
                state.transitions = _normalize({inner_ladder[3].name: 1.0})
            case 3:
                state.transitions = _normalize({inner_ladder[4].name: 1.0})
            case 4:
                state.transitions = _normalize({inner_ladder[5].name: 1.0})
            case 5:
                state.transitions = _normalize({arom_top[0].name: 1.0})
            case 6:
                state.transitions = _normalize(
                    {
                        arom_top[0].name: 1.0,
                        inner_ladder[5].name: 1.0,
                        inner_ladder[7].name: 1.0,
                    }
                )
            case 7:
                state.transitions = _normalize(
                    {
                        inner_ladder[8].name: 1.0,
                        inner_ladder[5].name: 1.0,
                        inner_ladder[4].name: 1.0,
                    }
                )
            case 8:
                state.transitions = _normalize(
                    {
                        inner_ladder[9].name: 1.0,
                        inner_ladder[3].name: 1.0,
                        inner_ladder[4].name: 1.0,
                    }
                )
            case 9:
                state.transitions = _normalize(
                    {
                        inner_ladder[10].name: 1.0,
                        inner_ladder[2].name: 1.0,
                        inner_ladder[3].name: 1.0,
                    }
                )
            case 10:
                state.transitions = _normalize(
                    {
                        inner_ladder[11].name: 1.0,
                        inner_ladder[1].name: 1.0,
                        inner_ladder[2].name: 1.0,
                    }
                )
            case 11:
                state.transitions = _normalize(
                    {
                        c_term.name: 1.0,
                        inner_ladder[0].name: 1.0,
                        inner_ladder[1].name: 1.0,
                    }
                )
            case _:
                raise ValueError(
                    f"there is no inner ladder element with the following number in our model: {i}"
                )

    # === Aromatic Belt Top ===
    for i, state in enumerate(arom_top):
        match i:
            case 0:
                state.transitions = _normalize({tm_int[0].name: 1.0})
            case 1:
                state.transitions = _normalize({outer_ladder[0].name: 1.0})
            case _:
                raise ValueError(
                    f"there is no Aromatic Belt Top element with the following number in our model: {i}"
                )

    # === TM Exterior ===
    for i, ext_state in enumerate(tm_ext):
        if i < 7:
            ext_state.transitions = _normalize({tm_int[i + 1].name: 1.0})
        else:
            ext_state.transitions = _normalize({tm_int[i + 2].name: 1.0})

    # === TM Interior ===
    for i, int_state in enumerate(tm_int):
        match i:
            case 0:
                int_state.transitions = _normalize({tm_ext[0].name: 1.0})
            case 1:
                int_state.transitions = _normalize({tm_ext[1].name: 1.0})
            case 2:
                int_state.transitions = _normalize(
                    {
                        tm_ext[2].name: 1.0,
                        tm_ext[3].name: 1.0,
                        tm_ext[4].name: 1.0,
                        tm_ext[5].name: 1.0,
                        tm_ext[6].name: 1.0,
                        arom_top[1].name: 1.0,
                    },
                )
            case 3:
                int_state.transitions = _normalize({tm_ext[3].name: 1.0})
            case 4:
                int_state.transitions = _normalize({tm_ext[4].name: 1.0})
            case 5:
                int_state.transitions = _normalize({tm_ext[5].name: 1.0})
            case 6:
                int_state.transitions = _normalize({tm_ext[6].name: 1.0})
            case 7:
                int_state.transitions = _normalize({arom_top[1].name: 1.0})
            case 8:
                int_state.transitions = _normalize({tm_ext[7].name: 1.0})
            case 9:
                int_state.transitions = _normalize({tm_ext[8].name: 1.0})
            case 10:
                int_state.transitions = _normalize(
                    {
                        tm_ext[9].name: 1.0,
                        tm_ext[10].name: 1.0,
                        tm_ext[11].name: 1.0,
                        tm_ext[12].name: 1.0,
                        tm_ext[13].name: 1.0,
                        arom_bottom[1].name: 1.0,
                    },
                )
            case 11:
                int_state.transitions = _normalize({tm_ext[10].name: 1.0})
            case 12:
                int_state.transitions = _normalize({tm_ext[11].name: 1.0})
            case 13:
                int_state.transitions = _normalize({tm_ext[12].name: 1.0})
            case 14:
                int_state.transitions = _normalize({tm_ext[13].name: 1.0})
            case 15:
                int_state.transitions = _normalize({arom_bottom[1].name: 1.0})

    # === Aromatic Belt Bottom ===
    for i, state in enumerate(arom_bottom):
        if i == 0:
            state.transitions = _normalize({tm_int[8].name: 1.0})
        else:
            state.transitions = _normalize({inner_ladder[6].name: 1.0})

    # === Outer Ladder ===
    for i, state in enumerate(outer_ladder):
        match i:
            case 0:
                state.transitions = _normalize(
                    {
                        outer_ladder[1].name: 1.0,
                        outer_ladder[11].name: 1.0,
                        arom_bottom[0].name: 1.0,
                    }
                )
            case 1:
                state.transitions = _normalize(
                    {
                        outer_ladder[2].name: 1.0,
                        outer_ladder[11].name: 1.0,
                        outer_ladder[10].name: 1.0,
                    }
                )
            case 2:
                state.transitions = _normalize(
                    {
                        outer_ladder[3].name: 1.0,
                        outer_ladder[10].name: 1.0,
                        outer_ladder[9].name: 1.0,
                    }
                )
            case 3:
                state.transitions = _normalize(
                    {
                        outer_ladder[4].name: 1.0,
                        outer_ladder[8].name: 1.0,
                        outer_ladder[9].name: 1.0,
                    }
                )
            case 4:
                state.transitions = _normalize(
                    {
                        outer_ladder[5].name: 1.0,
                        outer_ladder[7].name: 1.0,
                        outer_ladder[8].name: 1.0,
                    }
                )
            case 5:
                state.transitions = _normalize(
                    {
                        globular.name: 1.0,
                        outer_ladder[6].name: 1.0,
                        outer_ladder[7].name: 1.0,
                    }
                )
            case 6:
                state.transitions = _normalize({outer_ladder[7].name: 1.0})
            case 7:
                state.transitions = _normalize({outer_ladder[8].name: 1.0})
            case 8:
                state.transitions = _normalize({outer_ladder[9].name: 1.0})
            case 9:
                state.transitions = _normalize({outer_ladder[10].name: 1.0})
            case 10:
                state.transitions = _normalize({outer_ladder[11].name: 1.0})
            case 11:
                state.transitions = _normalize({arom_bottom[0].name: 1.0})

    # === Globular ===
    globular.transitions = _normalize(
        {
            globular.name: 1.0,
            outer_ladder[6].name: 1.0,
        }
    )
