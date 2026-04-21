
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.unified import UnifiedDialogue, UnifiedTurn


FEATURE_NAMES = ["OscScore", "CoverageRate", "ConflictCount", "FillVelocity", "DomainShifts"]
PRIMARY_TAXONOMY_LABELS = [
    "Information Incompleteness",
    "Contradiction / Misalignment",
    "State Instability",
    "Task Drift / Domain Confusion",
]


def load_unified_from_cache(cache_dir: str, split: str):
    path = os.path.join(cache_dir, f"unified_{split}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    dialogues = []
    for d in data:
        turns = [
            UnifiedTurn(
                turn_idx=t["turn_idx"],
                user_utterance=t["user_utterance"],
                system_utterance=t["system_utterance"],
                belief_state=t["belief_state"],
                turn_delta=t["turn_delta"],
                domain=t["domain"],
            )
            for t in d["turns"]
        ]
        dialogues.append(UnifiedDialogue(
            dialogue_id=d["dialogue_id"],
            dataset=d["dataset"],
            domains=d["domains"],
            turns=turns,
            success=d["success"],
            metadata=d.get("metadata", {}),
        ))
    return dialogues


def load_features(cache_dir: str, split: str):
    path = os.path.join(cache_dir, f"features_{split}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def collect_feature_values_by_outcome(dialogues, features_dict, dataset_filter=None):
    success_vals = {k: [] for k in FEATURE_NAMES}
    fail_vals = {k: [] for k in FEATURE_NAMES}

    for dial in dialogues:
        if dataset_filter and dial.dataset != dataset_filter:
            continue
        feats = features_dict.get(dial.dialogue_id, [])
        if not feats:
            continue
        arr = np.array(feats)
        mean_per_dial = arr.mean(axis=0)
        if dial.success:
            for i, name in enumerate(FEATURE_NAMES):
                success_vals[name].append(mean_per_dial[i])
        else:
            for i, name in enumerate(FEATURE_NAMES):
                fail_vals[name].append(mean_per_dial[i])

    return success_vals, fail_vals


def table1_feature_means_by_outcome(dialogues, features_dict, dataset_filter=None):
    success_vals, fail_vals = collect_feature_values_by_outcome(
        dialogues, features_dict, dataset_filter
    )

    rows = []
    for name in FEATURE_NAMES:
        s, f = success_vals[name], fail_vals[name]
        if len(s) < 2 or len(f) < 2:
            rows.append({
                "Feature": name,
                "Mean (Success)": np.mean(s) if s else 0,
                "Mean (Fail)": np.mean(f) if f else 0,
                "p-value": 1.0,
            })
            continue
        stat, p = stats.mannwhitneyu(s, f, alternative="two-sided")
        rows.append({
            "Feature": name,
            "Mean (Success)": np.mean(s),
            "Mean (Fail)": np.mean(f),
            "p-value": p,
        })

    return pd.DataFrame(rows)


def feature_evolution_by_outcome(dialogues, features_dict, dataset_filter=None, max_turns=50):
    n_bins = 20
    success_bins = {i: [] for i in range(n_bins)}
    fail_bins = {i: [] for i in range(n_bins)}

    for dial in dialogues:
        if dataset_filter and dial.dataset != dataset_filter:
            continue
        feats = features_dict.get(dial.dialogue_id, [])
        if not feats:
            continue
        T = len(feats)
        for t_idx, f in enumerate(feats):
            norm_pos = t_idx / max(T, 1)
            bin_idx = min(int(norm_pos * n_bins), n_bins - 1)
            if dial.success:
                success_bins[bin_idx].append(f)
            else:
                fail_bins[bin_idx].append(f)

    return success_bins, fail_bins, n_bins


def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _serialize_turns(turns, max_turns=4):
    snippets = []
    for turn in turns[:max_turns]:
        snippets.append({
            "turn_idx": int(turn.turn_idx),
            "domain": turn.domain,
            "user": turn.user_utterance,
            "system": turn.system_utterance,
        })
    return snippets


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extract_changed_slots(prev_bs, curr_bs):
    changed = []
    slots = set(prev_bs) | set(curr_bs)
    for slot in slots:
        prev_val = prev_bs.get(slot)
        curr_val = curr_bs.get(slot)
        if prev_val != curr_val:
            changed.append(slot)
    return changed


def _detect_repeated_slot_reversals(turns):
    slot_history = {}
    event_turns = []
    for idx, turn in enumerate(turns):
        for slot, value in turn.belief_state.items():
            history = slot_history.setdefault(slot, [])
            if not history or history[-1] != value:
                if value in history[:-1]:
                    event_turns.append(idx)
                history.append(value)
    return sorted(set(event_turns))


def _detect_domain_shift_turns(turns):
    event_turns = []
    prev_domain = None
    for idx, turn in enumerate(turns):
        domain = turn.domain or ""
        if prev_domain and domain and domain != prev_domain:
            event_turns.extend([max(0, idx - 1), idx, min(len(turns) - 1, idx + 1)])
        if domain:
            prev_domain = domain
    return sorted(set(event_turns))


def _detect_conflict_turns(turns):
    cues = (
        "no", "not", "actually", "instead", "wrong", "sorry", "rather", "double check",
        "i said", "i meant", "that is not", "do not", "don't",
    )
    event_turns = []
    for idx, turn in enumerate(turns):
        text = f"{turn.user_utterance} {turn.system_utterance}".lower()
        if any(cue in text for cue in cues):
            event_turns.extend([max(0, idx - 1), idx, min(len(turns) - 1, idx + 1)])
    return sorted(set(event_turns))


def _detect_incompleteness_turns(turns):
    if not turns:
        return []
    n = len(turns)
    anchors = {0, max(0, n // 2), max(0, n - 3), max(0, n - 2), n - 1}
    return sorted(i for i in anchors if 0 <= i < n)


def select_review_turns(dialogue, primary_label, secondary_label="", max_turns=5):
    turns = dialogue.turns
    if not turns:
        return []

    if primary_label == "Contradiction / Misalignment":
        candidate_idxs = _detect_conflict_turns(turns)
    elif primary_label == "State Instability":
        candidate_idxs = _detect_repeated_slot_reversals(turns)
    elif primary_label == "Task Drift / Domain Confusion":
        candidate_idxs = _detect_domain_shift_turns(turns)
    else:
        candidate_idxs = _detect_incompleteness_turns(turns)

    if secondary_label:
        if secondary_label == "Contradiction / Misalignment":
            candidate_idxs.extend(_detect_conflict_turns(turns))
        elif secondary_label == "State Instability":
            candidate_idxs.extend(_detect_repeated_slot_reversals(turns))
        elif secondary_label == "Task Drift / Domain Confusion":
            candidate_idxs.extend(_detect_domain_shift_turns(turns))
        elif secondary_label == "Information Incompleteness":
            candidate_idxs.extend(_detect_incompleteness_turns(turns))

    if not candidate_idxs:
        n = len(turns)
        candidate_idxs = sorted({0, max(0, n // 3), max(0, n // 2), max(0, (2 * n) // 3), n - 1})

    ordered = []
    seen = set()
    for idx in candidate_idxs:
        if 0 <= idx < len(turns) and idx not in seen:
            ordered.append(idx)
            seen.add(idx)
        if len(ordered) >= max_turns:
            break

    if len(ordered) < max_turns:
        fallback = sorted({0, len(turns) // 2, len(turns) - 1})
        for idx in fallback:
            if 0 <= idx < len(turns) and idx not in seen:
                ordered.append(idx)
                seen.add(idx)
            if len(ordered) >= max_turns:
                break

    return _serialize_turns([turns[idx] for idx in ordered], max_turns=max_turns)


def compute_dialogue_feature_stats(features):
    arr = np.array(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return {
            "num_turns": 0,
            "osc_mean": 0.0,
            "osc_max": 0.0,
            "osc_final": 0.0,
            "cov_mean": 0.0,
            "cov_min": 0.0,
            "cov_final": 0.0,
            "conf_mean": 0.0,
            "conf_max": 0.0,
            "conf_final": 0.0,
            "vel_mean": 0.0,
            "vel_min": 0.0,
            "vel_final": 0.0,
            "shifts_mean": 0.0,
            "shifts_max": 0.0,
            "shifts_final": 0.0,
            "late_cov_gain": 0.0,
        }

    first_half_end = max(1, arr.shape[0] // 2)
    cov_first = arr[:first_half_end, 1]
    cov_second = arr[first_half_end:, 1]
    late_cov_gain = float(cov_second.mean() - cov_first.mean()) if len(cov_second) else 0.0

    return {
        "num_turns": int(arr.shape[0]),
        "osc_mean": float(arr[:, 0].mean()),
        "osc_max": float(arr[:, 0].max()),
        "osc_final": float(arr[-1, 0]),
        "cov_mean": float(arr[:, 1].mean()),
        "cov_min": float(arr[:, 1].min()),
        "cov_final": float(arr[-1, 1]),
        "conf_mean": float(arr[:, 2].mean()),
        "conf_max": float(arr[:, 2].max()),
        "conf_final": float(arr[-1, 2]),
        "vel_mean": float(arr[:, 3].mean()),
        "vel_min": float(arr[:, 3].min()),
        "vel_final": float(arr[-1, 3]),
        "shifts_mean": float(arr[:, 4].mean()),
        "shifts_max": float(arr[:, 4].max()),
        "shifts_final": float(arr[-1, 4]),
        "late_cov_gain": late_cov_gain,
    }


def score_failure_taxonomy(stats_dict):
    scores = {
        "Information Incompleteness": 0.0,
        "Contradiction / Misalignment": 0.0,
        "Task Drift / Domain Confusion": 0.0,
        "State Instability": 0.0,
    }
    evidence = {label: [] for label in scores}

    if stats_dict["cov_final"] <= 0.45:
        scores["Information Incompleteness"] += 3.0
        evidence["Information Incompleteness"].append(f"final coverage low ({stats_dict['cov_final']:.2f})")
    if stats_dict["cov_mean"] <= 0.60:
        scores["Information Incompleteness"] += 2.0
        evidence["Information Incompleteness"].append(f"mean coverage low ({stats_dict['cov_mean']:.2f})")
    if stats_dict["vel_mean"] <= 0.20:
        scores["Information Incompleteness"] += 1.5
        evidence["Information Incompleteness"].append(f"fill velocity stalled ({stats_dict['vel_mean']:.2f})")
    if stats_dict["late_cov_gain"] <= 0.05:
        scores["Information Incompleteness"] += 1.0
        evidence["Information Incompleteness"].append(f"little late coverage gain ({stats_dict['late_cov_gain']:.2f})")

    if stats_dict["conf_max"] >= 2:
        scores["Contradiction / Misalignment"] += 3.0
        evidence["Contradiction / Misalignment"].append(f"conflicts accumulate ({stats_dict['conf_max']:.1f})")
    elif stats_dict["conf_max"] >= 1:
        scores["Contradiction / Misalignment"] += 1.5
        evidence["Contradiction / Misalignment"].append(f"at least one contradiction ({stats_dict['conf_max']:.1f})")
    if stats_dict["conf_mean"] >= 0.75:
        scores["Contradiction / Misalignment"] += 1.5
        evidence["Contradiction / Misalignment"].append(f"conflicts persist on average ({stats_dict['conf_mean']:.2f})")
    if stats_dict["conf_final"] >= 1:
        scores["Contradiction / Misalignment"] += 1.0
        evidence["Contradiction / Misalignment"].append(f"dialogue ends with active contradiction signal ({stats_dict['conf_final']:.1f})")

    if stats_dict["shifts_max"] >= 3:
        scores["Task Drift / Domain Confusion"] += 3.0
        evidence["Task Drift / Domain Confusion"].append(f"many domains visited ({stats_dict['shifts_max']:.1f})")
    elif stats_dict["shifts_max"] >= 2:
        scores["Task Drift / Domain Confusion"] += 1.5
        evidence["Task Drift / Domain Confusion"].append(f"multiple domains visited ({stats_dict['shifts_max']:.1f})")
    if stats_dict["shifts_mean"] >= 1.75:
        scores["Task Drift / Domain Confusion"] += 1.0
        evidence["Task Drift / Domain Confusion"].append(f"domain switching sustained ({stats_dict['shifts_mean']:.2f})")
    if stats_dict["cov_final"] <= 0.70:
        scores["Task Drift / Domain Confusion"] += 1.0
        evidence["Task Drift / Domain Confusion"].append("coverage remains incomplete after domain changes")

    if stats_dict["osc_max"] >= 2:
        scores["State Instability"] += 3.0
        evidence["State Instability"].append(f"multiple oscillating slots ({stats_dict['osc_max']:.1f})")
    elif stats_dict["osc_max"] >= 1:
        scores["State Instability"] += 1.75
        evidence["State Instability"].append(f"at least one oscillating slot ({stats_dict['osc_max']:.1f})")
    if stats_dict["osc_mean"] >= 0.50:
        scores["State Instability"] += 1.5
        evidence["State Instability"].append(f"oscillation persists on average ({stats_dict['osc_mean']:.2f})")
    if stats_dict["osc_final"] >= 1:
        scores["State Instability"] += 1.0
        evidence["State Instability"].append(f"oscillation still present at the end ({stats_dict['osc_final']:.1f})")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1]
    insufficient_evidence = top_score < 2.5
    mixed_signal = second_score >= 2.5 and (top_score - second_score) <= 1.0
    secondary_label = second_label if mixed_signal else ""
    rationale_items = evidence.get(top_label, [])[:2]
    if mixed_signal:
        rationale_items += evidence.get(second_label, [])[:1]
    if insufficient_evidence:
        rationale_items.append("evidence is weak, so this assignment should be treated as low confidence")

    return {
        "primary_label": top_label,
        "secondary_label": secondary_label,
        "mixed_signal": mixed_signal,
        "insufficient_evidence": insufficient_evidence,
        "scores": scores,
        "rationale_items": rationale_items,
        "top_score": float(top_score),
        "second_score": float(second_score),
    }


def build_failure_taxonomy_records(dialogues, features_dict, dataset_filter=None):
    records = []

    for dial in dialogues:
        if dial.success:
            continue
        if dataset_filter and dial.dataset != dataset_filter:
            continue

        feats = features_dict.get(dial.dialogue_id, [])
        if not feats:
            records.append({
                "dialogue_id": dial.dialogue_id,
                "dataset": dial.dataset,
                "primary_label": "Information Incompleteness",
                "secondary_label": "",
                "mixed_signal": False,
                "insufficient_evidence": True,
                "confidence": 0.0,
                "rationale": "missing feature cache",
                "num_turns": len(dial.turns),
                "turn_snippets": select_review_turns(dial, "Information Incompleteness"),
            })
            continue

        stats_dict = compute_dialogue_feature_stats(feats)
        taxonomy_result = score_failure_taxonomy(stats_dict)
        scores = taxonomy_result["scores"]
        primary_label = taxonomy_result["primary_label"]
        secondary_label = taxonomy_result["secondary_label"]
        top_score = taxonomy_result["top_score"]
        second_score = taxonomy_result["second_score"]
        rationale_items = taxonomy_result["rationale_items"] or [f"top score {top_score:.1f}"]
        record = {
            "dialogue_id": dial.dialogue_id,
            "dataset": dial.dataset,
            "primary_label": primary_label,
            "secondary_label": secondary_label,
            "mixed_signal": taxonomy_result["mixed_signal"],
            "insufficient_evidence": taxonomy_result["insufficient_evidence"],
            "confidence": float(top_score - second_score),
            "rationale": "; ".join(rationale_items[:3]),
            "score_information_incompleteness": scores["Information Incompleteness"],
            "score_contradiction_misalignment": scores["Contradiction / Misalignment"],
            "score_task_drift_domain_confusion": scores["Task Drift / Domain Confusion"],
            "score_state_instability": scores["State Instability"],
            **stats_dict,
            "turn_snippets": select_review_turns(dial, primary_label, secondary_label),
        }
        records.append(record)

    return records


def table2_failure_taxonomy(dialogues, features_dict, dataset_filter=None):
    records = build_failure_taxonomy_records(dialogues, features_dict, dataset_filter)
    counts = {label: 0 for label in PRIMARY_TAXONOMY_LABELS}
    examples = {label: [] for label in PRIMARY_TAXONOMY_LABELS}

    for record in records:
        label = record["primary_label"]
        counts[label] = counts.get(label, 0) + 1
        if len(examples[label]) < 3:
            examples[label].append(record["dialogue_id"])

    total = max(len(records), 1)
    rows = []
    mixed_count = sum(1 for record in records if _normalize_bool(record["mixed_signal"]))
    insufficient_count = sum(1 for record in records if _normalize_bool(record["insufficient_evidence"]))
    for label in PRIMARY_TAXONOMY_LABELS:
        rows.append({
            "Primary Label": label,
            "Count": counts.get(label, 0),
            "Percent": 100.0 * counts.get(label, 0) / total,
            "Example Dialogues": ", ".join(examples.get(label, [])),
        })

    rows.append({
        "Primary Label": "mixed_signal=yes",
        "Count": mixed_count,
        "Percent": 100.0 * mixed_count / total,
        "Example Dialogues": "",
    })
    rows.append({
        "Primary Label": "insufficient_evidence=yes",
        "Count": insufficient_count,
        "Percent": 100.0 * insufficient_count / total,
        "Example Dialogues": "",
    })

    return pd.DataFrame(rows), pd.DataFrame(records)


def taxonomy_threshold_reference():
    rows = [
        {
            "Primary Label": "Information Incompleteness",
            "Primary cues": "low final/mean coverage with low fill velocity and little late improvement",
            "Threshold guide": "cov_final <= 0.45, cov_mean <= 0.60, vel_mean <= 0.20",
        },
        {
            "Primary Label": "Contradiction / Misalignment",
            "Primary cues": "repeated user-belief contradictions or unresolved conflicts",
            "Threshold guide": "conf_max >= 2 or conf_mean >= 0.75",
        },
        {
            "Primary Label": "Task Drift / Domain Confusion",
            "Primary cues": "multiple domain switches while coverage remains incomplete",
            "Threshold guide": "shifts_max >= 3, or shifts_max >= 2 with cov_final <= 0.70",
        },
        {
            "Primary Label": "State Instability",
            "Primary cues": "slots change value repeatedly and remain unstable",
            "Threshold guide": "osc_max >= 2, or osc_max >= 1 with persistent oscillation",
        },
        {
            "Primary Label": "mixed_signal=yes",
            "Primary cues": "a second failure mode is clearly visible and close in strength to the primary label",
            "Threshold guide": "second score >= 2.5 and within 1.0 of the top score",
        },
        {
            "Primary Label": "insufficient_evidence=yes",
            "Primary cues": "no dominant signal crosses a reliable threshold",
            "Threshold guide": "top score < 2.5",
        },
    ]
    return pd.DataFrame(rows)


def build_taxonomy_manual_validation_sample(records_df, sample_size_per_class=12, seed=13, exclude_dialogue_ids=None):
    rng = random.Random(seed)
    validation_rows = []
    exclude_dialogue_ids = set(exclude_dialogue_ids or [])

    for label in PRIMARY_TAXONOMY_LABELS:
        full_subset = records_df[records_df["primary_label"] == label]
        subset = full_subset[~full_subset["dialogue_id"].astype(str).isin(exclude_dialogue_ids)]
        if len(subset) < sample_size_per_class:
            subset = full_subset
        if subset.empty:
            continue
        idxs = list(subset.index)
        rng.shuffle(idxs)
        chosen = idxs[:sample_size_per_class]
        for idx in chosen:
            row = subset.loc[idx]
            snippets = row.get("turn_snippets", [])[:5]
            validation_rows.append({
                "dialogue_id": row["dialogue_id"],
                "dataset": row["dataset"],
                "primary_label_predicted": row["primary_label"],
                "secondary_label_predicted": row.get("secondary_label", ""),
                "mixed_signal_predicted": "yes" if _normalize_bool(row.get("mixed_signal")) else "no",
                "insufficient_evidence_predicted": "yes" if _normalize_bool(row.get("insufficient_evidence")) else "no",
                "rationale": row["rationale"],
                "num_turns": row["num_turns"],
                "snippet_1": json.dumps(snippets[0], ensure_ascii=True) if len(snippets) > 0 else "",
                "snippet_2": json.dumps(snippets[1], ensure_ascii=True) if len(snippets) > 1 else "",
                "snippet_3": json.dumps(snippets[2], ensure_ascii=True) if len(snippets) > 2 else "",
                "snippet_4": json.dumps(snippets[3], ensure_ascii=True) if len(snippets) > 3 else "",
                "snippet_5": json.dumps(snippets[4], ensure_ascii=True) if len(snippets) > 4 else "",
                "manual_primary_label": "",
                "manual_secondary_label": "",
                "manual_mixed_signal": "",
                "manual_insufficient_evidence": "",
                "manual_notes": "",
            })

    return pd.DataFrame(validation_rows)


def main():
    parser = argparse.ArgumentParser(description="Empirical analysis for TRACER")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="mwoz", help="Filter by dataset: mwoz, sgd, abcd or all")
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--taxonomy_sample_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import omegaconf
    cfg = omegaconf.OmegaConf.load(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    plot_dir = args.plot_dir or cfg.output.get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    dataset_filter = None if args.dataset == "all" else args.dataset
    dialogues = load_unified_from_cache(cache_dir, args.split)
    features_dict = load_features(cache_dir, args.split)

    if not dialogues or not features_dict:
        print("No data found. Run preprocess_all.py and compute_features.py first.")
        return

    df1 = table1_feature_means_by_outcome(dialogues, features_dict, dataset_filter)
    print("Table 1: Feature means (Success vs Fail) and Mann-Whitney U p-values")
    print(df1.to_string(index=False))
    df1.to_csv(os.path.join(plot_dir, "table1_feature_means.csv"), index=False)

    df2, taxonomy_records = table2_failure_taxonomy(dialogues, features_dict, dataset_filter)
    print("\nTable 2: Failure taxonomy")
    print(df2.to_string(index=False))
    df2.to_csv(os.path.join(plot_dir, "table2_failure_taxonomy.csv"), index=False)
    taxonomy_records.to_json(
        os.path.join(plot_dir, "table2_failure_taxonomy_details.json"),
        orient="records",
        indent=2,
    )
    taxonomy_records.drop(columns=["turn_snippets"], errors="ignore").to_csv(
        os.path.join(plot_dir, "table2_failure_taxonomy_details.csv"),
        index=False,
    )
    threshold_df = taxonomy_threshold_reference()
    threshold_df.to_csv(os.path.join(plot_dir, "table2_failure_taxonomy_rules.csv"), index=False)
    existing_manual_path = os.path.join(plot_dir, "table2_failure_taxonomy_manual_validation.csv")
    exclude_dialogue_ids = set()
    if os.path.exists(existing_manual_path):
        try:
            existing_df = pd.read_csv(existing_manual_path)
            if "dialogue_id" in existing_df.columns:
                exclude_dialogue_ids = set(existing_df["dialogue_id"].dropna().astype(str))
        except Exception:
            exclude_dialogue_ids = set()
    validation_df = build_taxonomy_manual_validation_sample(
        taxonomy_records,
        sample_size_per_class=args.taxonomy_sample_size,
        seed=args.seed,
        exclude_dialogue_ids=exclude_dialogue_ids,
    )
    validation_df.to_csv(
        os.path.join(plot_dir, "table2_failure_taxonomy_manual_validation.csv"),
        index=False,
    )
    print("\nTaxonomy rule reference")
    print(threshold_df.to_string(index=False))
    print(
        f"\nManual validation sample saved to "
        f"{os.path.join(plot_dir, 'table2_failure_taxonomy_manual_validation.csv')}"
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping Figures")
    else:
        success_bins, fail_bins, n_bins = feature_evolution_by_outcome(
            dialogues, features_dict, dataset_filter
        )
        x = np.linspace(0, 1, n_bins)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for feat_idx, name in enumerate(FEATURE_NAMES):
            ax = axes[feat_idx]
            s_means = [np.mean([f[feat_idx] for f in success_bins[i]]) if success_bins[i] else 0 for i in range(n_bins)]
            s_stds = [np.std([f[feat_idx] for f in success_bins[i]]) if success_bins[i] else 0 for i in range(n_bins)]
            f_means = [np.mean([f[feat_idx] for f in fail_bins[i]]) if fail_bins[i] else 0 for i in range(n_bins)]
            f_stds = [np.std([f[feat_idx] for f in fail_bins[i]]) if fail_bins[i] else 0 for i in range(n_bins)]
            ax.fill_between(x, np.array(s_means) - np.array(s_stds), np.array(s_means) + np.array(s_stds), alpha=0.3, color="green")
            ax.plot(x, s_means, color="green", label="Success")
            ax.fill_between(x, np.array(f_means) - np.array(f_stds), np.array(f_means) + np.array(f_stds), alpha=0.3, color="red")
            ax.plot(x, f_means, color="red", label="Fail")
            ax.set_title(name)
            ax.legend()
            ax.set_xlabel("Normalized turn position")
        axes[-1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "feature_evolution.pdf"), bbox_inches="tight")
        plt.close()
        print(f"Figure 1 saved to {plot_dir}/feature_evolution.pdf")

        success_vals, fail_vals = collect_feature_values_by_outcome(
            dialogues, features_dict, dataset_filter
        )
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for feat_idx, name in enumerate(FEATURE_NAMES):
            ax = axes[feat_idx]
            s = np.array(success_vals.get(name, []))
            f = np.array(fail_vals.get(name, []))
            if s.size == 0 and f.size == 0:
                ax.set_visible(False)
                continue
            bins = 30
            if s.size > 0:
                ax.hist(s, bins=bins, alpha=0.5, color="green", density=True, label="Success")
            if f.size > 0:
                ax.hist(f, bins=bins, alpha=0.5, color="red", density=True, label="Fail")
            ax.set_title(name)
            ax.legend()
        axes[-1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "feature_distributions.pdf"), bbox_inches="tight")
        plt.close()
        print(f"Figure 2 saved to {plot_dir}/feature_distributions.pdf")

    print("Empirical analysis complete.")


if __name__ == "__main__":
    main()
