
import numpy as np
from typing import Dict, List, Tuple

from ..evaluation.metrics import compute_early_detection_score


def find_pareto_threshold(dev_predictions, dev_labels,
                          n_thresholds=100, max_fpr=0.15):
    thresholds = np.linspace(0.1, 0.9, n_thresholds)
    results = []

    for tau in thresholds:
        eds_result = compute_early_detection_score(
            dev_predictions, dev_labels, threshold=tau
        )

        n_fp = 0
        n_success = 0
        for dial_id, preds in dev_predictions.items():
            if dev_labels.get(dial_id, True):
                n_success += 1
                if any(p > tau for p in preds):
                    n_fp += 1

        fpr = n_fp / max(n_success, 1)

        results.append({
            "threshold": float(tau),
            "eds": eds_result["eds"],
            "detection_rate": eds_result["detection_rate"],
            "mean_detection_turn": eds_result["mean_detection_turn"],
            "fpr": fpr,
        })

    valid = [r for r in results if r["fpr"] <= max_fpr]
    if not valid:
        valid = sorted(results, key=lambda r: r["fpr"])[:10]

    best = max(valid, key=lambda r: r["eds"])

    return {
        "optimal_threshold": best["threshold"],
        "optimal_eds": best["eds"],
        "optimal_fpr": best["fpr"],
        "optimal_detection_rate": best["detection_rate"],
        "all_results": results,
    }


def compute_intervention_cost(dev_predictions, dev_labels, threshold,
                               lambda_efficiency=1.0):
    n_interventions = 0
    n_dialogues = 0
    total_extra_turns = 0

    for dial_id, preds in dev_predictions.items():
        n_dialogues += 1
        interventions_in_dial = sum(1 for p in preds if p > threshold)
        n_interventions += interventions_in_dial
        total_extra_turns += interventions_in_dial

    avg_extra_turns = total_extra_turns / max(n_dialogues, 1)
    intervention_rate = n_interventions / max(n_dialogues, 1)

    return {
        "avg_extra_turns": avg_extra_turns,
        "intervention_rate": intervention_rate,
        "n_dialogues": n_dialogues,
    }
