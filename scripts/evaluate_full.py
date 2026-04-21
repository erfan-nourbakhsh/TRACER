
import argparse
import json
import os
import sys
import re
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.unified import (
    UnifiedDialogue,
    UnifiedTurn,
    PrefixDialogueDataset,
    collate_dialogue_batch,
)
from tracer.models.dual_stream import TRACERPredictor, FeaturesOnlyPredictor, TextOnlyPredictor
from tracer.models.baselines import (
    NoIntervention,
    FixedScheduleConfirmation,
    SlotConfidenceTrigger,
    FeatureThresholdEnsemble,
)
from tracer.models.belief_encoder import BeliefStateEncoder
from tracer.evaluation.metrics import (
    compute_auc_roc,
    compute_f1_metrics,
    compute_early_detection_score,
    compute_auc_roc_per_turn,
)
from tracer.recovery.threshold import find_pareto_threshold
from tracer.recovery.template_recovery import generate_template_recovery
from tracer.recovery.t5_recovery import T5RecoveryGenerator
from tracer.recovery.llm_recovery import LLMRecoveryGenerator
from tracer.evaluation.error_analysis import run_full_error_analysis
from tracer.data.domain_slots import get_required_slots_mwoz


def load_config(config_path: str):
    import omegaconf
    return omegaconf.OmegaConf.load(config_path)


def load_unified_and_features(cache_dir: str, split: str, dataset_filter=None, suffix: str = ""):
    suffix_part = f"_{suffix}" if suffix else ""
    path_dial = os.path.join(cache_dir, f"unified_{split}{suffix_part}.json")
    path_feat = os.path.join(cache_dir, f"features_{split}{suffix_part}.json")
    if not os.path.exists(path_dial) or not os.path.exists(path_feat):
        return [], {}
    with open(path_dial, "r") as f:
        data = json.load(f)
    dialogues = []
    for d in data:
        if dataset_filter and d.get("dataset") != dataset_filter:
            continue
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
    with open(path_feat, "r") as f:
        features_dict = json.load(f)
    if dataset_filter:
        features_dict = {d["dialogue_id"]: features_dict.get(d["dialogue_id"]) for d in data if d.get("dataset") == dataset_filter and d["dialogue_id"] in features_dict}
    else:
        features_dict = {k: v for k, v in features_dict.items() if any(d.dialogue_id == k for d in dialogues)}
    for k in list(features_dict.keys()):
        if features_dict[k] is not None:
            features_dict[k] = [np.array(x, dtype=np.float32) for x in features_dict[k]]
    return dialogues, features_dict


def run_predictor(model, loader, device, model_type="tracer"):
    preds, labels, dial_ids, turn_positions, dialogue_lengths = [], [], [], [], []
    for batch in loader:
        features = batch["features"].to(device)
        traj_mask = batch["traj_mask"].to(device)
        labs = batch["labels"].numpy()
        dial_ids.extend(batch.get("dialogue_id", [f"unk_{i}" for i in range(len(labs))]))
        turn_positions.extend(batch["turn_position"].numpy().tolist())
        dialogue_lengths.extend(batch["dialogue_length"].numpy().tolist())

        if model_type in ("tracer", "full") and hasattr(model, "forward"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(features, traj_mask, input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
        elif model_type == "features_only":
            with torch.no_grad():
                logits = model(features, traj_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
        elif model_type == "text_only":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = np.zeros(len(labs))
        preds.extend(probs)
        labels.extend(labs)
    return (
        np.array(preds),
        np.array(labels),
        dial_ids,
        np.array(turn_positions),
        np.array(dialogue_lengths),
    )

def build_dialogue_prediction_sequences(predictions, labels, dialogue_ids, turn_positions):
    dialogue_predictions = {}
    dialogue_labels = {}

    for pred, label, dial_id, turn_idx in zip(predictions, labels, dialogue_ids, turn_positions):
        if dial_id not in dialogue_predictions:
            dialogue_predictions[dial_id] = []
        dialogue_predictions[dial_id].append((int(turn_idx), float(pred)))
        dialogue_labels[dial_id] = bool(label == 0)

    sorted_predictions = {}
    for dial_id, seq in dialogue_predictions.items():
        seq = sorted(seq, key=lambda x: x[0])
        sorted_predictions[dial_id] = [pred for _, pred in seq]

    return sorted_predictions, dialogue_labels


def compute_and_plot_per_turn_auc(predictions, labels, turn_positions, plot_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_turn_auc = compute_auc_roc_per_turn(
        np.array(predictions), np.array(labels), np.array(turn_positions)
    )
    if not per_turn_auc:
        return {}

    turns = sorted(per_turn_auc.keys())
    auc_vals = [per_turn_auc[t] for t in turns]

    plt.figure(figsize=(6, 4))
    plt.plot(turns, auc_vals, marker="o")
    plt.xlabel("Turn index")
    plt.ylabel("AUC-ROC")
    plt.title("Per-turn AUC-ROC for TRACER")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return per_turn_auc


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _slot_name_variants(slot_name: str):
    slot = slot_name.lower().replace("_", " ")
    return {
        slot,
        slot.replace("book ", ""),
        slot.replace("price range", "pricerange"),
        slot.replace("pricerange", "price range"),
        slot.replace("leaveat", "leave at"),
        slot.replace("arriveby", "arrive by"),
    }


def _compute_missing_slots(belief_state, required_slots, domain):
    missing = []
    for slot in required_slots:
        full_key = f"{domain}-{slot}"
        if not belief_state.get(full_key):
            missing.append(slot)
    return missing


def _compute_future_unstable_slots(dialogue, trigger_turn):
    unstable = set()
    if trigger_turn >= len(dialogue.turns) - 1:
        return unstable
    current_bs = dialogue.turns[trigger_turn].belief_state
    future_turns = dialogue.turns[trigger_turn + 1 :]
    for turn in future_turns:
        future_bs = turn.belief_state
        for slot in set(current_bs.keys()) | set(future_bs.keys()):
            if current_bs.get(slot) != future_bs.get(slot):
                unstable.add(slot)
    return unstable


def _compute_groundedness_score(utterance, belief_state):
    text = _normalize_text(utterance)
    if not text or not belief_state:
        return 0.0

    grounded = 0
    considered = 0
    for slot, value in sorted(belief_state.items())[:6]:
        if not value:
            continue
        considered += 1
        slot_name = slot.split("-", 1)[-1] if "-" in slot else slot
        slot_match = any(variant in text for variant in _slot_name_variants(slot_name))
        value_match = _normalize_text(str(value)) in text
        if slot_match or value_match:
            grounded += 1

    return grounded / max(considered, 1)


def _compute_problem_targeting_score(utterance, missing_slots, unstable_slots):
    text = _normalize_text(utterance)
    if not text:
        return 0.0

    targeted = 0
    problem_slots = set()
    for slot in missing_slots:
        problem_slots.add(slot)
    for slot in unstable_slots:
        slot_name = slot.split("-", 1)[-1] if "-" in slot else slot
        problem_slots.add(slot_name)

    if not problem_slots:
        return 0.0

    for slot_name in problem_slots:
        if any(variant in text for variant in _slot_name_variants(slot_name)):
            targeted += 1

    return targeted / len(problem_slots)


def _compute_missing_slot_request_score(utterance, missing_slots):
    text = _normalize_text(utterance)
    if not missing_slots or not text:
        return 0.0
    mentioned = 0
    for slot_name in missing_slots:
        if any(variant in text for variant in _slot_name_variants(slot_name)):
            mentioned += 1
    return mentioned / len(missing_slots)


def _compute_clarification_intent_score(utterance):
    text = _normalize_text(utterance)
    if not text:
        return 0.0
    keywords = [
        "confirm",
        "clarify",
        "correct",
        "right",
        "make sure",
        "just to",
        "could you provide",
        "could you confirm",
        "is that",
        "would you like",
        "i still need",
    ]
    keyword_hit = any(keyword in text for keyword in keywords)
    question_like = "?" in utterance
    return 1.0 if keyword_hit or question_like else 0.0


def _score_recovery_utterance(utterance, dialogue, trigger_turn, required_slots):
    if trigger_turn >= len(dialogue.turns):
        return None

    belief_state = dialogue.turns[trigger_turn].belief_state or {}
    domain = dialogue.turns[trigger_turn].domain or (dialogue.domains[0] if dialogue.domains else "")
    missing_slots = _compute_missing_slots(belief_state, required_slots, domain) if domain else []
    unstable_slots = _compute_future_unstable_slots(dialogue, trigger_turn)

    groundedness = _compute_groundedness_score(utterance, belief_state)
    problem_targeting = _compute_problem_targeting_score(utterance, missing_slots, unstable_slots)
    missing_slot_request = _compute_missing_slot_request_score(utterance, missing_slots)
    clarification_intent = _compute_clarification_intent_score(utterance)

    recovery_utility = (
        0.35 * groundedness
        + 0.35 * problem_targeting
        + 0.20 * clarification_intent
        + 0.10 * missing_slot_request
    )

    return {
        "groundedness": groundedness,
        "problem_targeting": problem_targeting,
        "missing_slot_request": missing_slot_request,
        "clarification_intent": clarification_intent,
        "recovery_utility": recovery_utility,
        "missing_slots": missing_slots,
        "unstable_slots": sorted(unstable_slots),
        "belief_state_size": len(belief_state),
        "utterance_length_tokens": len((utterance or "").split()),
    }


def _build_recovery_turn_context(dialogue, trigger_turn):
    ctx_turns = []
    for idx in range(max(0, trigger_turn - 2), trigger_turn + 1):
        turn = dialogue.turns[idx]
        ctx_turns.append((turn.user_utterance or "", turn.system_utterance or ""))
    return ctx_turns


def _find_first_trigger_turn(preds, threshold):
    for idx, pred in enumerate(preds):
        if pred > threshold:
            return idx
    return None


def _assign_trigger_quality_bucket(dialogue, trigger_turn):
    dialogue_length = max(len(dialogue.turns), 1)
    trigger_fraction = trigger_turn / max(dialogue_length - 1, 1)
    if dialogue.success:
        return "False positive", trigger_fraction
    if trigger_fraction <= 0.33:
        return "Early correct", trigger_fraction
    if trigger_fraction >= 0.66:
        return "Late correct", trigger_fraction
    return "Middle correct", trigger_fraction


def _summarize_recovery_examples(examples):
    if not examples:
        return {
            "NumCases": 0,
            "MeanRecoveryUtility": 0.0,
            "MeanGroundedness": 0.0,
            "MeanProblemTargeting": 0.0,
            "MeanMissingSlotRequest": 0.0,
            "MeanClarificationIntent": 0.0,
            "MeanUtteranceLengthTokens": 0.0,
            "MeanTriggerTurn": None,
            "MeanTriggerFraction": None,
        }
    return {
        "NumCases": len(examples),
        "MeanRecoveryUtility": float(np.mean([ex["recovery_utility"] for ex in examples])),
        "MeanGroundedness": float(np.mean([ex["groundedness"] for ex in examples])),
        "MeanProblemTargeting": float(np.mean([ex["problem_targeting"] for ex in examples])),
        "MeanMissingSlotRequest": float(np.mean([ex["missing_slot_request"] for ex in examples])),
        "MeanClarificationIntent": float(np.mean([ex["clarification_intent"] for ex in examples])),
        "MeanUtteranceLengthTokens": float(np.mean([ex["utterance_length_tokens"] for ex in examples])),
        "MeanTriggerTurn": float(np.mean([ex["trigger_turn"] for ex in examples])),
        "MeanTriggerFraction": float(np.mean([ex["trigger_fraction"] for ex in examples])),
    }


def _prefix_summary_vector(prefix_features, turn_position, dialogue_length):
    arr = np.asarray(prefix_features, dtype=np.float32)
    last_feat = arr[-1]
    mean_feat = arr.mean(axis=0)
    max_feat = arr.max(axis=0)
    min_feat = arr.min(axis=0)
    std_feat = arr.std(axis=0)
    progress = np.array(
        [(turn_position + 1) / max(float(dialogue_length), 1.0)], dtype=np.float32
    )
    return np.concatenate([last_feat, mean_feat, max_feat, min_feat, std_feat, progress])


def _last_turn_vector(prefix_features, turn_position, dialogue_length):
    arr = np.asarray(prefix_features, dtype=np.float32)
    last_feat = arr[-1]
    progress = np.array(
        [(turn_position + 1) / max(float(dialogue_length), 1.0)], dtype=np.float32
    )
    return np.concatenate([last_feat, progress])


def build_classical_prefix_dataset(dialogues, features_dict, vectorizer="summary"):
    X, y, dial_ids, turn_positions, dialogue_lengths = [], [], [], [], []
    vectorizer_fn = (
        _prefix_summary_vector if vectorizer == "summary" else _last_turn_vector
    )

    for dial in dialogues:
        feats = features_dict.get(dial.dialogue_id, [])
        if not feats:
            continue
        label = 0.0 if dial.success else 1.0
        max_prefix_turns = min(len(dial.turns), len(feats))
        for turn_idx in range(max_prefix_turns):
            prefix_feats = feats[: turn_idx + 1]
            X.append(vectorizer_fn(prefix_feats, turn_idx, max_prefix_turns))
            y.append(label)
            dial_ids.append(dial.dialogue_id)
            turn_positions.append(turn_idx)
            dialogue_lengths.append(max_prefix_turns)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        dial_ids,
        np.asarray(turn_positions),
        np.asarray(dialogue_lengths),
    )


def evaluate_predictions_table_row(name, preds, labels, dial_ids, turn_positions):
    auc = compute_auc_roc(preds, labels)
    f1m = compute_f1_metrics(preds, labels)
    dial_preds, dial_labels = build_dialogue_prediction_sequences(
        preds, labels, dial_ids, turn_positions
    )
    eds = compute_early_detection_score(dial_preds, dial_labels)
    return {
        "Model": name,
        "AUC-ROC": auc,
        "F1": f1m["f1"],
        "EDS": eds["eds"],
        "MeanDetectionTurn": eds["mean_detection_turn"],
    }


def main():
    parser = argparse.ArgumentParser(description="Full TRACER evaluation")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_predictor.pt")
    parser.add_argument("--dataset", type=str, default="mwoz")
    parser.add_argument(
        "--test_input_suffix",
        type=str,
        default="",
        help="Read unified_test_{suffix}.json and features_test_{suffix}.json for test evaluation.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_recovery", action="store_true", help="Skip recovery and LLM/T5 evaluation")
    parser.add_argument("--skip_cross_domain", action="store_true")
    parser.add_argument("--skip_ablations", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    out_dir = args.output_dir or cfg.output.get("checkpoint_dir", "outputs")
    plot_dir = cfg.output.get("plot_dir", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BeliefStateEncoder.get_tokenizer(cfg.model.stream_b.get("encoder_name", "roberta-base"))
    max_length = cfg.model.stream_b.get("max_length", 512)

    test_dialogues, test_features = load_unified_and_features(
        cache_dir,
        "test",
        args.dataset,
        suffix=args.test_input_suffix,
    )
    if not test_dialogues:
        print("No test data. Run preprocess_all and compute_features first.")
        return

    train_dialogues, train_features = load_unified_and_features(cache_dir, "train", args.dataset)

    test_dataset = PrefixDialogueDataset(
        test_dialogues,
        test_features,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_dialogue_batch,
    )

    resolved_main_checkpoint = args.checkpoint or cfg.model.get("predictor_checkpoint", None)

    results = {
        "evaluation_setup": {
            "dataset_view": "prefix_based",
            "train_dev_test_claim": "forecasting_from_dialogue_prefixes",
            "note": "All forecasting metrics are computed from prefixes ending at turn t rather than whole-dialogue scores.",
            "checkpoints": {
                "tracer_full": resolved_main_checkpoint,
                "features_only": cfg.model.get("features_only_checkpoint", None),
                "text_only": cfg.model.get("text_only_checkpoint", None),
                "t5_recovery": cfg.recovery.get(
                    "t5_checkpoint_dir", "outputs/t5_recovery"
                ),
            },
            "artifacts": {
                "per_turn_auc_plot": os.path.join(plot_dir, "per_turn_auc.pdf"),
                "results_json": os.path.join(out_dir, "evaluate_full_results.json"),
                "error_analysis_json": os.path.join(out_dir, "error_analysis.json"),
            },
        }
    }

    print("Table 3: Failure prediction on MultiWOZ test")
    table3 = []

    if args.checkpoint and os.path.exists(args.checkpoint):
        model = TRACERPredictor(
            stream_a_config=dict(cfg.model.stream_a),
            stream_b_config=dict(cfg.model.stream_b),
            fusion_config=dict(cfg.model.fusion),
        )
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        preds, labels, dial_ids, turn_positions, dialogue_lengths = run_predictor(
            model, test_loader, device, "tracer"
        )
        auc = compute_auc_roc(preds, labels)
        f1m = compute_f1_metrics(preds, labels)
        tracer_dial_preds, tracer_dial_labels = build_dialogue_prediction_sequences(
            preds, labels, dial_ids, turn_positions
        )
        eds = compute_early_detection_score(tracer_dial_preds, tracer_dial_labels)
        table3.append(
            evaluate_predictions_table_row(
                "TRACER (full)", preds, labels, dial_ids, turn_positions
            )
        )
        results["tracer_full"] = {
            "predictions": preds.tolist(),
            "labels": labels.tolist(),
            "turn_positions": turn_positions.tolist(),
            "dialogue_lengths": dialogue_lengths.tolist(),
            "auc": auc,
            "eds": eds,
        }

        try:
            per_turn_auc = compute_and_plot_per_turn_auc(
                preds,
                labels,
                turn_positions,
                os.path.join(plot_dir, "per_turn_auc.pdf"),
            )
            results["tracer_full"]["per_turn_auc"] = per_turn_auc
        except Exception as e:
            print(f"Per-turn AUC computation failed: {e}")

        if not eds["sanity_check_passed"]:
            print(
                "WARNING: all detected failures were triggered at turn 0. "
                "Check forecasting setup or threshold selection."
            )
    else:
        print("No checkpoint provided, skipping TRACER full.")

    for name, baseline in [
        ("No intervention", NoIntervention()),
        ("Fixed-schedule (every 3)", FixedScheduleConfirmation(interval=3)),
        ("Slot-confidence (Osc>0)", SlotConfidenceTrigger(threshold=0.0)),
        ("Feature-threshold ensemble", FeatureThresholdEnsemble()),
    ]:
        preds_list, labels_list, ids_list, turn_list = [], [], [], []
        for batch in test_loader:
            labs = batch["labels"].numpy()
            turns = batch["turn_position"].numpy()
            dial_ids = batch.get("dialogue_id", [f"b_{len(preds_list)+i}" for i in range(len(labs))])
            if name == "No intervention":
                p = baseline.predict(None)
                if np.isscalar(p):
                    p = np.full(len(labs), float(p))
                preds_list.extend(np.atleast_1d(p)[: len(labs)])
            elif name.startswith("Fixed"):
                p = baseline.predict(turns)
                if isinstance(p, torch.Tensor):
                    p = p.numpy()
                preds_list.extend(np.atleast_1d(p)[: len(labs)])
            else:
                feats = batch["features"]
                if feats.dim() == 3:
                    last_indices = batch["traj_mask"].sum(dim=1).long() - 1
                    prefix_last_feats = feats[
                        torch.arange(feats.shape[0]), last_indices, :
                    ]
                else:
                    prefix_last_feats = feats
                p = baseline.predict(
                    prefix_last_feats.numpy() if hasattr(prefix_last_feats, "numpy") else prefix_last_feats.cpu().numpy()
                )
                preds_list.extend(np.array(p).flatten()[: len(labs)])
            labels_list.extend(labs)
            ids_list.extend(dial_ids[: len(labs)])
            turn_list.extend(turns.tolist())
        preds_arr = np.array(preds_list[: len(labels_list)])
        labels_arr = np.array(labels_list)
        table3.append(
            evaluate_predictions_table_row(
                name, preds_arr, labels_arr, ids_list, np.asarray(turn_list)
            )
        )

    feat_ckpt = cfg.model.get("features_only_checkpoint", None)
    if feat_ckpt and os.path.exists(feat_ckpt):
        try:
            feat_model = FeaturesOnlyPredictor(dict(cfg.model.stream_a))
            ckpt = torch.load(feat_ckpt, map_location=device, weights_only=False)
            feat_model.load_state_dict(ckpt["model_state_dict"])
            feat_model.to(device)
            feat_model.eval()
            f_preds, f_labels, f_ids, f_turns, _ = run_predictor(
                feat_model, test_loader, device, "features_only"
            )
            table3.append(
                evaluate_predictions_table_row(
                    "Features-only neural baseline",
                    f_preds,
                    f_labels,
                    f_ids,
                    f_turns,
                )
            )
        except Exception as e:
            print(f"Features-only baseline failed: {e}")

    text_ckpt = cfg.model.get("text_only_checkpoint", None)
    if text_ckpt and os.path.exists(text_ckpt):
        try:
            text_model = TextOnlyPredictor(dict(cfg.model.stream_b))
            ckpt = torch.load(text_ckpt, map_location=device, weights_only=False)
            text_model.load_state_dict(ckpt["model_state_dict"])
            text_model.to(device)
            text_model.eval()
            t_preds, t_labels, t_ids, t_turns, _ = run_predictor(
                text_model, test_loader, device, "text_only"
            )
            table3.append(
                evaluate_predictions_table_row(
                    "Text-only neural baseline",
                    t_preds,
                    t_labels,
                    t_ids,
                    t_turns,
                )
            )
        except Exception as e:
            print(f"Text-only baseline failed: {e}")

    if train_dialogues and train_features:
        try:
            X_train_last, y_train_last, _, _, _ = build_classical_prefix_dataset(
                train_dialogues, train_features, vectorizer="last_turn"
            )
            X_test_last, y_test_last, ids_test_last, turns_test_last, _ = build_classical_prefix_dataset(
                test_dialogues, test_features, vectorizer="last_turn"
            )
            last_turn_logreg = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=cfg.training.get("seed", 42),
                        ),
                    ),
                ]
            )
            last_turn_logreg.fit(X_train_last, y_train_last)
            last_turn_probs = last_turn_logreg.predict_proba(X_test_last)[:, 1]
            table3.append(
                evaluate_predictions_table_row(
                    "Turn-level prefix logreg",
                    last_turn_probs,
                    y_test_last,
                    ids_test_last,
                    turns_test_last,
                )
            )

            X_train_sum, y_train_sum, _, _, _ = build_classical_prefix_dataset(
                train_dialogues, train_features, vectorizer="summary"
            )
            X_test_sum, y_test_sum, ids_test_sum, turns_test_sum, _ = build_classical_prefix_dataset(
                test_dialogues, test_features, vectorizer="summary"
            )
            summary_logreg = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=cfg.training.get("seed", 42),
                        ),
                    ),
                ]
            )
            summary_logreg.fit(X_train_sum, y_train_sum)
            summary_probs = summary_logreg.predict_proba(X_test_sum)[:, 1]
            table3.append(
                evaluate_predictions_table_row(
                    "Classical engineered-prefix logreg",
                    summary_probs,
                    y_test_sum,
                    ids_test_sum,
                    turns_test_sum,
                )
            )
        except Exception as e:
            print(f"Classical baselines failed: {e}")

    print(json.dumps(table3, indent=2))
    results["table3_failure_prediction"] = table3

    dev_dialogues, dev_features = load_unified_and_features(cache_dir, "dev", args.dataset)
    if dev_dialogues and args.checkpoint and os.path.exists(args.checkpoint):
        dev_dataset = PrefixDialogueDataset(
            dev_dialogues,
            dev_features,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_dialogue_batch)
        model.eval()
        dev_preds, dev_labels, dev_ids, dev_turn_positions, _ = run_predictor(
            model, dev_loader, device, "tracer"
        )
        dev_dial_preds, dev_dial_labels = build_dialogue_prediction_sequences(
            dev_preds, dev_labels, dev_ids, dev_turn_positions
        )
        tau_result = find_pareto_threshold(
            dev_dial_preds,
            dev_dial_labels,
            n_thresholds=cfg.recovery.get("threshold_num_steps", 100),
            max_fpr=cfg.recovery.get("max_fpr", 0.15),
        )
        results["optimal_threshold"] = tau_result
        print("Optimal threshold:", tau_result.get("optimal_threshold"))

    if not args.skip_recovery and test_dialogues and "tracer_full" in results:
        print("Table 4: Recovery evaluation (triggered offline proxy)")
        table4 = []
        table4b = []

        trigger_threshold = results.get("optimal_threshold", {}).get("optimal_threshold", 0.5)
        t5_ckpt = cfg.recovery.get(
            "t5_checkpoint_dir", "outputs/t5_recovery"
        )
        llm_enabled = bool(cfg.recovery.get("enable_llm_recovery", False))

        recovery_methods = [("Template", "available")]
        if os.path.exists(t5_ckpt):
            recovery_methods.append(("T5", "available"))
        else:
            recovery_methods.append(("T5", "unavailable"))
        if llm_enabled:
            recovery_methods.append(("LLM", "available"))
        else:
            recovery_methods.append(("LLM", "disabled"))

        t5_generator = None
        llm_generator = None

        if any(name == "T5" and status == "available" for name, status in recovery_methods):
            try:
                t5_generator = T5RecoveryGenerator(model_name_or_path=t5_ckpt)
                t5_generator.load(t5_ckpt)
            except Exception as e:
                print(f"Failed to load T5 recovery model from {t5_ckpt}: {e}")
                t5_generator = None
                recovery_methods = [
                    (name, "unavailable" if name == "T5" else status)
                    for name, status in recovery_methods
                ]

        if any(name == "LLM" and status == "available" for name, status in recovery_methods):
            llm_name = cfg.recovery.get(
                "llm_model_name",
                cfg.recovery.get("llm_model", "meta-llama/Llama-3.1-8B-Instruct"),
            )
            try:
                llm_generator = LLMRecoveryGenerator(model_name=llm_name)
            except Exception as e:
                print(f"Failed to initialize LLM recovery model {llm_name}: {e}")
                llm_generator = None
                recovery_methods = [
                    (name, "unavailable" if name == "LLM" else status)
                    for name, status in recovery_methods
                ]

        triggered_dialogues = []
        for dial in test_dialogues:
            preds = tracer_dial_preds.get(dial.dialogue_id, [])
            if not preds:
                continue
            trigger_turn = _find_first_trigger_turn(preds, trigger_threshold)
            if trigger_turn is None:
                continue
            triggered_dialogues.append((dial, trigger_turn))

        total_dialogues = len(test_dialogues)
        mean_dialogue_length = float(
            np.mean([len(d.turns) for d in test_dialogues]) if test_dialogues else 0.0
        )

        for rec_name, status in recovery_methods:
            entry = {
                "Recovery": rec_name,
                "Status": status,
                "TriggerThreshold": trigger_threshold,
                "InterventionRate": len(triggered_dialogues) / max(total_dialogues, 1),
                "AvgExtraTurnsPerDialogue": len(triggered_dialogues) / max(total_dialogues, 1),
                "AvgRelativeOverhead": (
                    (len(triggered_dialogues) / max(total_dialogues, 1)) / max(mean_dialogue_length, 1.0)
                ),
                "MeanTriggerTurn": (
                    float(np.mean([turn for _, turn in triggered_dialogues]))
                    if triggered_dialogues else float("inf")
                ),
            }

            if status != "available":
                entry["Note"] = "Method not evaluated because the generator is unavailable or disabled."
                table4.append(entry)
                continue

            scored_examples = []
            generation_failures = 0
            bucket_attempts = defaultdict(int)
            bucket_generation_failures = defaultdict(int)

            for dial, trigger_turn in triggered_dialogues[:100]:
                feats = test_features.get(dial.dialogue_id, [])
                if not feats or trigger_turn >= len(feats) or trigger_turn >= len(dial.turns):
                    continue
                trigger_bucket, trigger_fraction = _assign_trigger_quality_bucket(dial, trigger_turn)
                bucket_attempts[trigger_bucket] += 1

                turn = dial.turns[trigger_turn]
                belief_state = turn.belief_state or {}
                domain = turn.domain or (dial.domains[0] if dial.domains else "")
                required_slots = get_required_slots_mwoz(domain, has_booking=True) if domain else []
                utt = None

                try:
                    if rec_name == "Template":
                        utt = generate_template_recovery(
                            feats[trigger_turn],
                            belief_state,
                            domain or "dialogue",
                            required_slots=required_slots,
                        )
                    elif rec_name == "T5" and t5_generator is not None:
                        utt = t5_generator.generate(
                            feats[trigger_turn],
                            belief_state,
                            _build_recovery_turn_context(dial, trigger_turn),
                        )
                    elif rec_name == "LLM" and llm_generator is not None:
                        utt = llm_generator.generate(
                            feats[trigger_turn],
                            belief_state,
                            domain=domain or "dialogue",
                            last_user_utt=turn.user_utterance or "",
                            last_system_utt=turn.system_utterance or "",
                        )
                except Exception:
                    generation_failures += 1
                    bucket_generation_failures[trigger_bucket] += 1
                    continue

                if not utt:
                    generation_failures += 1
                    bucket_generation_failures[trigger_bucket] += 1
                    continue

                scores = _score_recovery_utterance(
                    utt,
                    dial,
                    trigger_turn,
                    required_slots,
                )
                if scores is None:
                    continue

                scored_examples.append(
                    {
                        "dialogue_id": dial.dialogue_id,
                        "label": "success" if dial.success else "fail",
                        "trigger_turn": trigger_turn,
                        "dialogue_length": len(dial.turns),
                        "trigger_fraction": trigger_fraction,
                        "trigger_bucket": trigger_bucket,
                        "utterance": utt,
                        "domain": domain,
                        **scores,
                    }
                )

            entry["GenerationFailureRate"] = generation_failures / max(len(triggered_dialogues), 1)
            summary = _summarize_recovery_examples(scored_examples)
            entry.update(summary)
            if scored_examples:
                top_examples = sorted(scored_examples, key=lambda ex: ex["recovery_utility"], reverse=True)[:3]
                low_examples = sorted(scored_examples, key=lambda ex: ex["recovery_utility"])[:2]
                entry["TopExamples"] = top_examples
                entry["LowExamples"] = low_examples
                entry["ScoredExamples"] = scored_examples

            table4.append(entry)

            for bucket_name in ["Early correct", "Late correct", "False positive"]:
                bucket_examples = [ex for ex in scored_examples if ex["trigger_bucket"] == bucket_name]
                bucket_summary = _summarize_recovery_examples(bucket_examples)
                bucket_entry = {
                    "Recovery": rec_name,
                    "TriggerQuality": bucket_name,
                    **bucket_summary,
                    "AvgExtraTurnsPerDialogue": bucket_attempts[bucket_name] / max(total_dialogues, 1),
                    "AvgRelativeOverhead": (
                        (bucket_attempts[bucket_name] / max(total_dialogues, 1))
                        / max(mean_dialogue_length, 1.0)
                    ),
                    "GenerationFailureRate": (
                        bucket_generation_failures[bucket_name] / max(bucket_attempts[bucket_name], 1)
                    ),
                    "TopExamples": sorted(bucket_examples, key=lambda ex: ex["recovery_utility"], reverse=True)[:2],
                    "LowExamples": sorted(bucket_examples, key=lambda ex: ex["recovery_utility"])[:1],
                }
                table4b.append(bucket_entry)

        results["table4_recovery"] = table4
        results["table4b_recovery_by_trigger_quality"] = table4b
        print(json.dumps(table4, indent=2))

    if not args.skip_cross_domain and args.checkpoint and os.path.exists(args.checkpoint):
        print("Table 5: Cross-domain zero-shot")
        table5 = []
        for target_ds in ["sgd", "abcd"]:
            if target_ds == args.dataset:
                continue
            dials, feats = load_unified_and_features(cache_dir, "test", target_ds)
            if not dials:
                continue
            ds = PrefixDialogueDataset(dials, feats, tokenizer=tokenizer, max_length=max_length)
            dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_dialogue_batch)
            preds, labels, dids, dturns, _ = run_predictor(model, dl, device, "tracer")
            auc = compute_auc_roc(preds, labels)
            f1m = compute_f1_metrics(preds, labels)
            dial_preds, dial_labels = build_dialogue_prediction_sequences(
                preds, labels, dids, dturns
            )
            eds = compute_early_detection_score(dial_preds, dial_labels)
            table5.append(
                {
                    "Dataset": target_ds,
                    "AUC-ROC": auc,
                    "F1": f1m["f1"],
                    "EDS": eds["eds"],
                    "MeanDetectionTurn": eds["mean_detection_turn"],
                }
            )
        results["table5_cross_domain"] = table5
        print(json.dumps(table5, indent=2))

    if (
        not args.skip_ablations
        and resolved_main_checkpoint
        and os.path.exists(resolved_main_checkpoint)
    ):
        print("Table 6: Ablations (Stream A only / Stream B only)")
        table6 = []

        feat_ckpt = cfg.model.get("features_only_checkpoint", None)
        if feat_ckpt and os.path.exists(feat_ckpt):
            try:
                feat_model = FeaturesOnlyPredictor(dict(cfg.model.stream_a))
                ckpt = torch.load(feat_ckpt, map_location=device, weights_only=False)
                feat_model.load_state_dict(ckpt["model_state_dict"])
                feat_model.to(device)
                feat_model.eval()
                f_preds, f_labels, _, _, _ = run_predictor(
                    feat_model, test_loader, device, "features_only"
                )
                f_auc = compute_auc_roc(f_preds, f_labels)
                f_f1 = compute_f1_metrics(f_preds, f_labels)
                table6.append(
                    {
                        "Model": "Stream A only (features)",
                        "AUC-ROC": f_auc,
                        "F1": f_f1["f1"],
                    }
                )
            except Exception as e:
                print(f"Features-only ablation failed: {e}")

        text_ckpt = cfg.model.get("text_only_checkpoint", None)
        if text_ckpt and os.path.exists(text_ckpt):
            try:
                text_model = TextOnlyPredictor(dict(cfg.model.stream_b))
                ckpt = torch.load(text_ckpt, map_location=device, weights_only=False)
                text_model.load_state_dict(ckpt["model_state_dict"])
                text_model.to(device)
                text_model.eval()
                t_preds, t_labels, _, _, _ = run_predictor(
                    text_model, test_loader, device, "text_only"
                )
                t_auc = compute_auc_roc(t_preds, t_labels)
                t_f1 = compute_f1_metrics(t_preds, t_labels)
                table6.append(
                    {
                        "Model": "Stream B only (text)",
                        "AUC-ROC": t_auc,
                        "F1": t_f1["f1"],
                    }
                )
            except Exception as e:
                print(f"Text-only ablation failed: {e}")

        if table6:
            results["table6_ablations"] = table6
            print(json.dumps(table6, indent=2))
        else:
            results["table6_ablations"] = []

    if "tracer_full" in results:
        preds = np.array(results["tracer_full"]["predictions"])
        labels = np.array(results["tracer_full"]["labels"])
        turn_positions = np.array(results["tracer_full"]["turn_positions"])
        dialogue_turns = {dial.dialogue_id: dial.turns for dial in test_dialogues}
        report = run_full_error_analysis(
            preds, labels,
            dialogue_predictions=tracer_dial_preds if "tracer_dial_preds" in locals() else None,
            dialogue_labels=tracer_dial_labels if "tracer_dial_labels" in locals() else None,
            dialogue_ids=dial_ids if "dial_ids" in locals() else None,
            turn_indices=turn_positions.tolist(),
            dialogue_turns=dialogue_turns,
            recovery_results=results.get("table4_recovery"),
            threshold=0.5,
            output_path=os.path.join(out_dir, "error_analysis.json"),
        )
        results["error_analysis"] = {k: v for k, v in report.items() if k != "case_study_report"}

    with open(os.path.join(out_dir, "evaluate_full_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir}/evaluate_full_results.json")


if __name__ == "__main__":
    main()
