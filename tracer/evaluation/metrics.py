
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    precision_score, recall_score, f1_score, roc_curve,
)
from typing import Dict, List, Tuple


def compute_auc_roc(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, predictions)


def compute_auc_roc_per_turn(predictions, labels, turn_positions):
    predictions = np.array(predictions)
    labels = np.array(labels)
    turn_positions = np.array(turn_positions)

    results = {}
    for t in sorted(set(turn_positions)):
        mask = turn_positions == t
        t_preds = predictions[mask]
        t_labels = labels[mask]
        if len(np.unique(t_labels)) < 2 or len(t_labels) < 10:
            continue
        results[int(t)] = roc_auc_score(t_labels, t_preds)

    return results


def compute_f1_metrics(predictions, labels, threshold=0.5):
    predictions = np.array(predictions)
    labels = np.array(labels)
    binary_preds = (predictions >= threshold).astype(int)

    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    accuracy = np.mean(binary_preds == labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
    }


def find_optimal_threshold(predictions, labels, metric="f1"):
    predictions = np.array(predictions)
    labels = np.array(labels)

    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])

    best_threshold = 0.5
    best_value = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        binary_preds = (predictions >= t).astype(int)
        value = f1_score(labels, binary_preds, zero_division=0)
        if value > best_value:
            best_value = value
            best_threshold = t

    return best_threshold, best_value


def compute_early_detection_score(dialogue_predictions, dialogue_labels, threshold=0.5):
    eds_scores = []
    detection_turns = []
    n_failed = 0
    n_detected = 0
    detected_at_turn_zero = 0

    for dial_id, preds in dialogue_predictions.items():
        label = dialogue_labels.get(dial_id, True)
        if label:
            continue

        n_failed += 1
        T = len(preds)
        if T == 0:
            continue

        t_detect = None
        for t, p in enumerate(preds):
            if p > threshold:
                t_detect = t
                break

        if t_detect is not None:
            n_detected += 1
            if t_detect == 0:
                detected_at_turn_zero += 1
            eds_i = (T - t_detect) / T
            eds_scores.append(eds_i)
            detection_turns.append(t_detect)
        else:
            eds_scores.append(0.0)

    eds = np.mean(eds_scores) if eds_scores else 0.0
    mean_detect = np.mean(detection_turns) if detection_turns else float("inf")
    detection_rate = n_detected / max(n_failed, 1)
    all_detected_at_turn_zero = bool(
        n_detected > 0 and detected_at_turn_zero == n_detected
    )

    return {
        "eds": eds,
        "mean_detection_turn": mean_detect,
        "detection_rate": detection_rate,
        "n_failed": n_failed,
        "n_detected": n_detected,
        "detected_at_turn_zero": detected_at_turn_zero,
        "all_detected_at_turn_zero": all_detected_at_turn_zero,
        "sanity_check_passed": not all_detected_at_turn_zero,
    }


def compute_belief_state_stability(belief_histories, recovery_turn):
    if recovery_turn <= 0 or recovery_turn >= len(belief_histories) - 1:
        return 0.0

    before_history = belief_histories[:recovery_turn + 1]
    osc_before = _count_oscillations(before_history)

    after_history = belief_histories[recovery_turn:]
    osc_after = _count_oscillations(after_history)

    bss = 1.0 - (osc_after / max(osc_before, 1))
    return max(bss, -1.0)


def _count_oscillations(belief_history):
    if len(belief_history) < 2:
        return 0

    changes = 0
    for i in range(1, len(belief_history)):
        prev = belief_history[i - 1]
        curr = belief_history[i]
        for slot in set(list(prev.keys()) + list(curr.keys())):
            if prev.get(slot) != curr.get(slot):
                changes += 1
    return changes


def compute_all_metrics(predictions, labels, turn_positions,
                        dialogue_predictions, dialogue_labels,
                        threshold=0.5):
    auc = compute_auc_roc(predictions, labels)
    per_turn_auc = compute_auc_roc_per_turn(predictions, labels, turn_positions)
    f1_metrics = compute_f1_metrics(predictions, labels, threshold)
    optimal_thresh, optimal_f1 = find_optimal_threshold(predictions, labels)
    eds_metrics = compute_early_detection_score(
        dialogue_predictions, dialogue_labels, threshold
    )

    return {
        "auc_roc": auc,
        "per_turn_auc": per_turn_auc,
        **f1_metrics,
        "optimal_threshold": optimal_thresh,
        "optimal_f1": optimal_f1,
        **eds_metrics,
    }
