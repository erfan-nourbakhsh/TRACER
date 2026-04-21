
import numpy as np
from typing import Dict, List, Optional


def compute_abcd_osc_score(turns, scenario):
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})

    trackable_values = {}
    for key, val in {**personal, **order}.items():
        if val and key != "products":
            trackable_values[key] = str(val).lower()

    mention_counts = {k: 0 for k in trackable_values}

    for turn in turns:
        if turn.speaker == "customer":
            text_lower = turn.text.lower()
            for key, val in trackable_values.items():
                if val in text_lower:
                    mention_counts[key] += 1

    osc = sum(1 for count in mention_counts.values() if count > 2)
    return float(osc)


def compute_abcd_coverage_rate(actions_performed, required_actions):
    if not required_actions:
        return 1.0

    completed = 0
    for req in required_actions:
        if req in actions_performed:
            completed += 1

    return completed / len(required_actions)


def compute_abcd_conflict_count(turns):
    frustration_keywords = [
        "no that's not", "that's wrong", "i said", "i already",
        "not what i", "incorrect", "i didn't say", "you're wrong",
        "that's not right", "i meant", "no, i", "please listen",
        "frustrated", "annoyed", "this is ridiculous",
    ]

    count = 0
    for turn in turns:
        if turn.speaker == "customer":
            text_lower = turn.text.lower()
            if any(kw in text_lower for kw in frustration_keywords):
                count += 1
    return count


def compute_abcd_fill_velocity(turns, scenario, t):
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})

    trackable = {}
    for key, val in {**personal, **order}.items():
        if val and key != "products":
            trackable[key] = str(val).lower()

    mentioned = set()
    counts = []
    for turn in turns[:t + 1]:
        if turn.speaker == "customer":
            text_lower = turn.text.lower()
            for key, val in trackable.items():
                if val in text_lower:
                    mentioned.add(key)
        counts.append(len(mentioned))

    if t < 2:
        return counts[t] / max(t + 1, 1) if t < len(counts) else 0.0

    current = counts[t] if t < len(counts) else 0
    past = counts[t - 2] if t - 2 < len(counts) else 0
    return (current - past) / 2.0


def compute_abcd_dialogue_features(dialogue, kb):
    turns_raw = []
    subflow = dialogue.metadata.get("subflow", "")
    required_actions = kb.get(subflow, [])
    actions_so_far = []

    features = []
    scenario = dialogue.metadata.get("scenario", {})

    for t_idx, turn in enumerate(dialogue.turns):
        performed = dialogue.metadata.get("actions_performed", [])
        actions_up_to_t = performed[:min(t_idx + 1, len(performed))]

        osc = compute_abcd_osc_score_at_turn(dialogue.turns, scenario, t_idx)
        coverage = compute_abcd_coverage_rate(actions_up_to_t, required_actions)
        conflict = compute_abcd_conflict_count_at_turn(dialogue.turns, t_idx)
        velocity = compute_abcd_fill_velocity_at_turn(dialogue.turns, scenario, t_idx)
        domain_shifts = 1

        feat = np.array([osc, coverage, conflict, velocity, domain_shifts],
                        dtype=np.float32)
        features.append(feat)

    return features


def compute_abcd_osc_score_at_turn(turns, scenario, t):
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})

    trackable = {}
    for key, val in {**personal, **order}.items():
        if val and key != "products":
            trackable[key] = str(val).lower()

    mention_counts = {k: 0 for k in trackable}
    for turn in turns[:t + 1]:
        text_lower = turn.user_utterance.lower() if hasattr(turn, 'user_utterance') else ""
        for key, val in trackable.items():
            if val in text_lower:
                mention_counts[key] += 1

    return float(sum(1 for c in mention_counts.values() if c > 2))


def compute_abcd_conflict_count_at_turn(turns, t):
    frustration_keywords = [
        "no that's not", "that's wrong", "i said", "i already",
        "not what i", "incorrect", "i didn't say",
        "that's not right", "i meant", "no, i",
    ]

    count = 0
    for turn in turns[:t + 1]:
        text = turn.user_utterance.lower() if hasattr(turn, 'user_utterance') else ""
        if any(kw in text for kw in frustration_keywords):
            count += 1
    return count


def compute_abcd_fill_velocity_at_turn(turns, scenario, t):
    personal = scenario.get("personal", {})
    order = scenario.get("order", {})

    trackable = {}
    for key, val in {**personal, **order}.items():
        if val and key != "products":
            trackable[key] = str(val).lower()

    mentioned = set()
    counts = []
    for turn in turns[:t + 1]:
        text = turn.user_utterance.lower() if hasattr(turn, 'user_utterance') else ""
        for key, val in trackable.items():
            if val in text:
                mentioned.add(key)
        counts.append(len(mentioned))

    if t < 2:
        return counts[t] / max(t + 1, 1) if t < len(counts) else 0.0

    current = counts[t] if t < len(counts) else 0
    past = counts[t - 2] if t - 2 < len(counts) else 0
    return (current - past) / 2.0
