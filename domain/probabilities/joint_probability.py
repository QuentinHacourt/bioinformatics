import numpy as np
from scipy.special import logsumexp
from domain.protein import Protein
from domain.state.state import State, StateRole
from hmm.forward_backward import (
    build_emission_matrix,
    build_index,
    build_initial_distribution,
    build_transition_matrix,
    encode_sequence,
    forward_log,
)


def annotation_to_state_sequence(
    annotated_sequence: str,
    name_to_idx: dict[str, int],
) -> list[int]:
    """
    Map each character in the annotation to a concrete state index.

    Strategy: walk through the annotation and track position within
    each segment (run of same label), then pick the appropriate state.

    I → inner_ladder_0..13 (cycle through ladder states)
    T → alternate tm_exterior / tm_interior, with aromatic belt at edges
    O → outer_ladder_0..12, with globular for long outer loops
    """
    result = []

    # Track position within current run of same label
    current_label = None
    pos_in_segment = 0

    # Pre-build lookup lists in order
    inner_ladder = [f"inner_ladder_{i}" for i in range(12)]
    outer_ladder = [f"outer_ladder_{i}" for i in range(12)]
    arom_top = [f"tm_aromatic_top_{i}" for i in range(2)]
    arom_bottom = [f"tm_aromatic_bottom_{i}" for i in range(2)]
    tm_ext = [f"tm_exterior_{i}" for i in range(14)]
    tm_int = [f"tm_interior_{i}" for i in range(16)]

    # We need to know segment lengths ahead of time for TM
    # so first parse into segments
    segments: list[tuple[str, int]] = []  # (label_char, length)
    i = 0
    while i < len(annotated_sequence):
        char = annotated_sequence[i]
        j = i
        while j < len(annotated_sequence) and annotated_sequence[j] == char:
            j += 1
        segments.append((char, j - i))
        i = j

    for label_char, length in segments:
        if label_char == "I":
            # Walk through inner ladder states, capped at ladder length
            for pos in range(length):
                state_name = inner_ladder[min(pos, len(inner_ladder) - 1)]
                result.append(name_to_idx[state_name])

        elif label_char == "O":
            # First few positions go through outer ladder,
            # long loops overflow into globular
            for pos in range(length):
                if pos < len(outer_ladder):
                    state_name = outer_ladder[pos]
                else:
                    state_name = "outer_globular"
                result.append(name_to_idx[state_name])

        elif label_char == "T":
            # TM strand layout:
            #   positions 0..2       → aromatic belt top (3 states)
            #   positions 3..end-3   → alternating exterior/interior
            #   positions end-2..end → aromatic belt bottom (3 states)
            belt_size = 2
            core_start = belt_size
            core_end = length - belt_size

            for pos in range(length):
                if pos < belt_size:
                    state_name = arom_top[pos]
                elif pos >= length - belt_size:
                    belt_pos = pos - (length - belt_size)
                    state_name = arom_bottom[belt_pos]
                else:
                    core_pos = pos - core_start
                    if core_pos % 2 == 0:
                        state_name = tm_ext[min(core_pos // 2, len(tm_ext) - 1)]
                    else:
                        state_name = tm_int[min(core_pos // 2, len(tm_int) - 1)]

                result.append(name_to_idx[state_name])

    return result


def joint_log_probability(
    protein: Protein,
    A: np.ndarray,
    B: np.ndarray,
    pi: np.ndarray,
    name_to_idx: dict[str, int],
) -> float:
    """
    Compute log P(x, y | θ) — the joint probability of the sequence
    and its labeling, following the annotated path exactly.

    This is the numerator in the CML criterion:
        P(y | x, θ) = P(x, y | θ) / P(x | θ)
    """
    obs = encode_sequence(protein.sequence)
    state_path = annotation_to_state_sequence(protein.labels, name_to_idx)

    if len(obs) != len(state_path):
        raise ValueError(
            f"{protein.name}: sequence length {len(obs)} != "
            f"annotation length {len(state_path)}"
        )

    log_A = np.log(np.where(A > 0, A, 1e-300))
    log_B = np.log(np.where(B > 0, B, 1e-300))
    log_pi = np.log(np.where(pi > 0, pi, 1e-300))

    # Start: initial state probability × emission of first residue
    log_prob = log_pi[state_path[0]] + log_B[state_path[0], obs[0]]

    # Walk the annotated path
    for t in range(1, len(obs)):
        i = state_path[t - 1]
        j = state_path[t]
        log_prob += log_A[i, j] + log_B[j, obs[t]]

    return log_prob


def cml_log_probability(
    protein: Protein,
    states: dict[str, State],
) -> float:
    """
    log P(y | x, θ) = log P(x, y | θ) - log P(x | θ)

    This is what CML training maximises.
    A value of 0.0 means the model is certain of the correct labeling.
    More negative = worse.
    """
    idx_to_name, name_to_idx = build_index(states)
    A = build_transition_matrix(states, name_to_idx)
    B = build_emission_matrix(states, name_to_idx)
    pi = build_initial_distribution(states, name_to_idx)
    obs = encode_sequence(protein.sequence)

    # Numerator: joint probability of sequence + correct labeling
    log_joint = joint_log_probability(protein, A, B, pi, name_to_idx)

    # Denominator: total probability of sequence (forward algorithm)
    log_alpha = forward_log(obs, A, B, pi)
    log_p_obs = logsumexp(log_alpha[-1, :])

    log_conditional = log_joint - log_p_obs

    print(f"{protein.name}")
    print(f"  log P(x,y):     {log_joint:.4f}")
    print(f"  log P(x):       {log_p_obs:.4f}")
    print(f"  log P(y|x):     {log_conditional:.4f}")  # 0.0 = perfect
    print(f"  P(y|x):         {np.exp(log_conditional):.4e}")

    return log_conditional


def total_cml_objective(proteins: list[Protein], states: dict[str, State]) -> float:
    """
    Sum of log P(y|x) over all training proteins.
    CML training maximises this value.
    """
    total = sum(cml_log_probability(p, states) for p in proteins)
    print(f"\nTotal CML objective: {total:.4f}")
    return total
