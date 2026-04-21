
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .metrics import compute_auc_roc, compute_f1_metrics

logger = logging.getLogger(__name__)


def serialize_turns(turns: List[Any], max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for turn in (turns[:max_turns] if max_turns is not None else turns):
        if isinstance(turn, dict):
            user = turn.get("user_utterance", "") or turn.get("user", "")
            system = turn.get("system_utterance", "") or turn.get("system", "")
            belief_state = turn.get("belief_state", {})
            domain = turn.get("domain", "")
            turn_idx = turn.get("turn_idx")
        else:
            user = getattr(turn, "user_utterance", "")
            system = getattr(turn, "system_utterance", "")
            belief_state = getattr(turn, "belief_state", {})
            domain = getattr(turn, "domain", "")
            turn_idx = getattr(turn, "turn_idx", None)
        out.append(
            {
                "turn_idx": turn_idx,
                "user_utterance": user,
                "system_utterance": system,
                "belief_state": belief_state,
                "domain": domain,
            }
        )
    return out


def collect_predictions_by_outcome(
    predictions: np.ndarray,
    labels: np.ndarray,
    dialogue_ids: Optional[List[str]] = None,
    turn_indices: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    binary = (preds >= threshold).astype(int)

    out = {"fp": [], "fn": [], "tp": [], "tn": []}
    for i in range(len(preds)):
        p, l = float(preds[i]), int(labs[i])
        b = binary[i]
        rec = {"pred": p, "label": l}
        if dialogue_ids is not None and i < len(dialogue_ids):
            rec["dialogue_id"] = dialogue_ids[i]
        if turn_indices is not None and i < len(turn_indices):
            rec["turn_idx"] = int(turn_indices[i])

        if b == 1 and l == 0:
            out["fp"].append(rec)
        elif b == 0 and l == 1:
            out["fn"].append(rec)
        elif b == 1 and l == 1:
            out["tp"].append(rec)
        else:
            out["tn"].append(rec)

    return out


def summarize_error_rates(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    groups = collect_predictions_by_outcome(predictions, labels, threshold=threshold)
    n_fp = len(groups["fp"])
    n_fn = len(groups["fn"])
    n_tp = len(groups["tp"])
    n_tn = len(groups["tn"])
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    return {
        "n_false_positive": n_fp,
        "n_false_negative": n_fn,
        "n_true_positive": n_tp,
        "n_true_negative": n_tn,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "fp_rate": n_fp / max(n_neg, 1),
        "fn_rate": n_fn / max(n_pos, 1),
        "precision": n_tp / max(n_tp + n_fp, 1),
        "recall": n_tp / max(n_tp + n_fn, 1),
    }


def select_case_study_dialogues(
    dialogue_predictions: Dict[str, List[float]],
    dialogue_labels: Dict[str, bool],
    dialogue_turns: Optional[Dict[str, List[Any]]] = None,
    threshold: float = 0.5,
    n_fp: int = 1,
    n_fn: int = 1,
    n_tp: int = 1,
    n_tn: int = 1,
) -> Dict[str, List[Dict]]:
    by_outcome = defaultdict(list)

    for dial_id, preds in dialogue_predictions.items():
        label = dialogue_labels.get(dial_id, True)
        preds_arr = np.array(preds)
        any_positive = (preds_arr > threshold).any()
        first_detect_turn = None
        for turn_idx, pred in enumerate(preds_arr):
            if pred > threshold:
                first_detect_turn = turn_idx
                break
        outcome = (1 if any_positive else 0, 1 if not label else 0)
        if outcome == (1, 0):
            key = "false_positive"
        elif outcome == (0, 1):
            key = "false_negative"
        elif outcome == (1, 1):
            key = "true_positive"
        else:
            key = "true_negative"

        rec = {
            "dialogue_id": dial_id,
            "label": "fail" if not label else "success",
            "preds_per_turn": preds_arr.tolist(),
            "max_pred": float(preds_arr.max()),
            "mean_pred": float(preds_arr.mean()),
            "first_detect_turn": first_detect_turn,
        }
        if dialogue_turns and dial_id in dialogue_turns:
            rec["turns"] = serialize_turns(dialogue_turns[dial_id], max_turns=8)
        by_outcome[key].append(rec)

    result = {}
    for key, target in [
        ("false_positive", n_fp),
        ("false_negative", n_fn),
        ("true_positive", n_tp),
        ("true_negative", n_tn),
    ]:
        pool = by_outcome.get(key, [])
        result[key] = pool[:target]

    return result


def feature_importance_from_predictions(
    predictions_full: np.ndarray,
    predictions_ablated: Dict[str, np.ndarray],
    labels: np.ndarray,
    metric: str = "auc",
) -> Dict[str, float]:
    if metric == "auc":
        base_score = compute_auc_roc(predictions_full, labels)
    else:
        m = compute_f1_metrics(predictions_full, labels)
        base_score = m.get("f1", 0.0)

    importance = {}
    for name, preds in predictions_ablated.items():
        if metric == "auc":
            ablated_score = compute_auc_roc(preds, labels)
        else:
            m = compute_f1_metrics(preds, labels)
            ablated_score = m.get("f1", 0.0)
        importance[name] = float(base_score - ablated_score)

    return importance


def format_case_study_report(
    case_studies: Dict[str, List[Dict]],
    max_turn_preview: int = 5,
) -> str:
    lines = ["# Error analysis – case studies", ""]
    for outcome, examples in case_studies.items():
        lines.append(f"## {outcome.replace('_', ' ').title()}")
        for i, ex in enumerate(examples, 1):
            lines.append(f"  Example {i}: dialogue_id={ex.get('dialogue_id', '?')}")
            lines.append(
                f"    label={ex.get('label')}, max_pred={ex.get('max_pred', 0):.3f}, "
                f"first_detect_turn={ex.get('first_detect_turn')}"
            )
            preds = ex.get("preds_per_turn", [])
            if preds:
                preview = preds[:max_turn_preview]
                lines.append(f"    preds (first {len(preview)} turns): {preview}")
            if "turns" in ex and ex["turns"]:
                for t_idx, turn in enumerate(ex["turns"][:3]):
                    u = (turn.get("user_utterance", "") or "")[:60]
                    s = (turn.get("system_utterance", "") or "")[:60]
                    lines.append(f"    turn {t_idx}: user: {u}... | sys: {s}...")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def select_detection_examples(
    dialogue_predictions: Dict[str, List[float]],
    dialogue_labels: Dict[str, bool],
    dialogue_turns: Optional[Dict[str, List[Any]]] = None,
    threshold: float = 0.5,
    max_examples: int = 2,
) -> Dict[str, List[Dict[str, Any]]]:
    early_detects = []
    missed_failures = []

    for dial_id, preds in dialogue_predictions.items():
        is_success = dialogue_labels.get(dial_id, True)
        preds_arr = np.asarray(preds)
        if preds_arr.size == 0 or is_success:
            if is_success:
                continue
        detect_turn = None
        for idx, pred in enumerate(preds_arr):
            if pred > threshold:
                detect_turn = idx
                break

        record = {
            "dialogue_id": dial_id,
            "preds_per_turn": preds_arr.tolist(),
            "max_pred": float(preds_arr.max()) if preds_arr.size else 0.0,
            "first_detect_turn": detect_turn,
        }
        if dialogue_turns and dial_id in dialogue_turns:
            record["turns"] = serialize_turns(dialogue_turns[dial_id], max_turns=8)

        if not is_success and detect_turn is not None:
            dialogue_len = len(preds_arr)
            record["detection_fraction"] = (detect_turn + 1) / max(dialogue_len, 1)
            early_detects.append(record)
        elif not is_success and detect_turn is None:
            missed_failures.append(record)

    early_detects = sorted(
        early_detects,
        key=lambda ex: (ex.get("detection_fraction", 1.0), -ex.get("max_pred", 0.0)),
    )[:max_examples]
    missed_failures = sorted(
        missed_failures,
        key=lambda ex: ex.get("max_pred", 0.0),
    )[:max_examples]

    return {
        "early_detection_hits": early_detects,
        "missed_failures": missed_failures,
    }


def summarize_failure_modes_from_examples(
    case_studies: Dict[str, List[Dict[str, Any]]],
    recovery_examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    mode_counts = defaultdict(int)
    supporting_examples = defaultdict(list)

    def classify_case(case: Dict[str, Any]) -> str:
        turns = case.get("turns", [])
        if not turns:
            return "insufficient_context"

        belief_states = [turn.get("belief_state", {}) or {} for turn in turns]
        domains = [turn.get("domain", "") for turn in turns if turn.get("domain")]
        last_bs = belief_states[-1] if belief_states else {}
        slot_counts = [len(bs) for bs in belief_states]

        slot_changes = 0
        for prev, curr in zip(belief_states[:-1], belief_states[1:]):
            for slot in set(prev.keys()) | set(curr.keys()):
                if prev.get(slot) != curr.get(slot):
                    slot_changes += 1
        if slot_changes >= max(3, len(turns)):
            return "belief_state_instability"

        if len(turns) >= 4 and len(last_bs) <= 1:
            return "missing_information_or_slot_coverage"

        if len(set(domains)) >= 2:
            return "domain_switching_or_misalignment"

        return "unclear_or_context_specific"

    for outcome in ("false_positive", "false_negative", "true_positive", "true_negative"):
        for case in case_studies.get(outcome, []):
            mode = classify_case(case)
            mode_counts[mode] += 1
            supporting_examples[mode].append(
                {
                    "dialogue_id": case.get("dialogue_id"),
                    "outcome": outcome,
                    "first_detect_turn": case.get("first_detect_turn"),
                }
            )

    if recovery_examples:
        weak_recovery = [
            ex for ex in recovery_examples
            if ex.get("recovery_utility", 1.0) < 0.25
        ]
        if weak_recovery:
            mode_counts["weak_recovery_targeting"] += len(weak_recovery)
            supporting_examples["weak_recovery_targeting"].extend(
                {
                    "dialogue_id": ex.get("dialogue_id"),
                    "outcome": "recovery",
                    "first_detect_turn": ex.get("trigger_turn"),
                }
                for ex in weak_recovery[:3]
            )

    dominant = sorted(mode_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "dominant_failure_modes": [
            {
                "mode": mode,
                "count": count,
                "examples": supporting_examples[mode][:3],
            }
            for mode, count in dominant
        ]
    }


def run_full_error_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    dialogue_predictions: Optional[Dict[str, List[float]]] = None,
    dialogue_labels: Optional[Dict[str, bool]] = None,
    dialogue_ids: Optional[List[str]] = None,
    turn_indices: Optional[List[int]] = None,
    dialogue_turns: Optional[Dict[str, List[Any]]] = None,
    recovery_results: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    summary = summarize_error_rates(predictions, labels, threshold)
    groups = collect_predictions_by_outcome(
        predictions, labels, dialogue_ids, turn_indices, threshold
    )

    report = {
        "summary": summary,
        "n_false_positives": len(groups["fp"]),
        "n_false_negatives": len(groups["fn"]),
        "sample_fp": groups["fp"][:10],
        "sample_fn": groups["fn"][:10],
    }

    if dialogue_predictions and dialogue_labels:
        case_studies = select_case_study_dialogues(
            dialogue_predictions,
            dialogue_labels,
            dialogue_turns=dialogue_turns,
            threshold=threshold,
            n_fp=2, n_fn=2, n_tp=1, n_tn=1,
        )
        report["case_studies"] = case_studies
        report["detection_examples"] = select_detection_examples(
            dialogue_predictions,
            dialogue_labels,
            dialogue_turns=dialogue_turns,
            threshold=threshold,
            max_examples=2,
        )
        if recovery_results:
            report["recovery_case_studies"] = {
                "good_recovery_examples": sorted(
                    [r for r in recovery_results if r.get("Status") == "available" for r in r.get("TopExamples", [])],
                    key=lambda ex: ex.get("recovery_utility", 0.0),
                    reverse=True,
                )[:3],
                "bad_recovery_examples": sorted(
                    [r for r in recovery_results if r.get("Status") == "available" for r in r.get("LowExamples", [])],
                    key=lambda ex: ex.get("recovery_utility", 1.0),
                )[:3],
            }
        report.update(
            summarize_failure_modes_from_examples(
                case_studies,
                recovery_examples=(
                    report.get("recovery_case_studies", {}).get("good_recovery_examples", [])
                    + report.get("recovery_case_studies", {}).get("bad_recovery_examples", [])
                ),
            )
        )
        report["case_study_report"] = format_case_study_report(case_studies)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(
                {k: v for k, v in report.items() if k != "case_study_report"},
                f,
                indent=2,
            )
        if "case_study_report" in report:
            text_path = output_path.replace(".json", "_report.txt")
            with open(text_path, "w") as f:
                f.write(report["case_study_report"])
        logger.info("Error analysis written to %s", output_path)

    return report
