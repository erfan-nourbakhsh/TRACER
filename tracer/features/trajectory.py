
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .conflict_detector import ConflictDetector


@dataclass
class TrajectoryFeatures:
    osc_score: float
    coverage_rate: float
    conflict_count: int
    fill_velocity: float
    domain_shifts: int

    def to_vector(self):
        return np.array([
            self.osc_score,
            self.coverage_rate,
            self.conflict_count,
            self.fill_velocity,
            self.domain_shifts,
        ], dtype=np.float32)


def compute_osc_score(belief_history):
    slot_values = {}

    for bs in belief_history:
        for slot, value in bs.items():
            if slot not in slot_values:
                slot_values[slot] = []
            if not slot_values[slot] or slot_values[slot][-1] != value:
                slot_values[slot].append(value)

    osc_count = sum(1 for vals in slot_values.values() if len(vals) > 2)
    return float(osc_count)


def compute_coverage_rate(belief_state, required_slots, domain):
    if not required_slots:
        return 1.0

    filled = 0
    for slot in required_slots:
        full_slot = f"{domain}-{slot}"
        if full_slot in belief_state and belief_state[full_slot]:
            filled += 1

    return filled / len(required_slots)


def compute_fill_velocity(belief_history, t):
    current_filled = len(belief_history[t]) if t < len(belief_history) else 0

    if t < 2:
        return current_filled / max(t + 1, 1)

    past_filled = len(belief_history[t - 2]) if t - 2 < len(belief_history) else 0
    return (current_filled - past_filled) / 2.0


def compute_domain_shifts(domain_history):
    return len(set(d for d in domain_history if d))


def compute_dialogue_features(dialogue, required_slots_fn,
                               conflict_detector=None):
    turns = dialogue.turns
    n = len(turns)
    if n == 0:
        return []

    belief_history = [t.belief_state for t in turns]
    domain_history = [t.domain for t in turns]
    user_utterances = [t.user_utterance for t in turns]
    turn_deltas = [t.turn_delta for t in turns]

    features = []
    cumulative_conflicts = 0

    for t in range(n):
        osc = compute_osc_score(belief_history[:t + 1])

        domain = domain_history[t]
        required = required_slots_fn(domain) if domain else []
        coverage = compute_coverage_rate(belief_history[t], required, domain)

        if conflict_detector and t > 0:
            prev_bs = belief_history[t - 1]
            if conflict_detector.detect_conflict(
                user_utterances[t], prev_bs, turn_deltas[t]
            ):
                cumulative_conflicts += 1
        conflict_count = cumulative_conflicts

        velocity = compute_fill_velocity(belief_history, t)

        shifts = compute_domain_shifts(domain_history[:t + 1])

        feat = TrajectoryFeatures(
            osc_score=osc,
            coverage_rate=coverage,
            conflict_count=conflict_count,
            fill_velocity=velocity,
            domain_shifts=shifts,
        )
        features.append(feat.to_vector())

    return features
