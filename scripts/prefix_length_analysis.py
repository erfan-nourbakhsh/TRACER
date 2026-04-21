
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_full import (
    load_config,
    load_unified_and_features,
    run_predictor,
    build_dialogue_prediction_sequences,
    build_classical_prefix_dataset,
)
from tracer.data.unified import PrefixDialogueDataset, collate_dialogue_batch
from tracer.evaluation.metrics import (
    compute_auc_roc,
    compute_f1_metrics,
    compute_early_detection_score,
)
from tracer.models.belief_encoder import BeliefStateEncoder
from tracer.models.dual_stream import (
    TRACERPredictor,
    FeaturesOnlyPredictor,
    TextOnlyPredictor,
)


PREFIX_RATIOS = [0.25, 0.50, 0.75, 1.0]
DEFAULT_MODELS = [
    "tracer",
    "text_only",
    "features_only",
    "logreg_summary",
    "logreg_last_turn",
]
MODEL_DISPLAY_NAMES = {
    "tracer": "TRACER (full)",
    "text_only": "Text-only",
    "features_only": "Features-only",
    "logreg_summary": "Classical engineered-prefix logreg",
    "logreg_last_turn": "Turn-level prefix logreg",
}
MODEL_COLORS = {
    "tracer": "#d62728",
    "text_only": "#328cc1",
    "features_only": "#d9b310",
    "logreg_summary": "#1d2731",
    "logreg_last_turn": "#b95f89",
}
MODEL_MARKERS = {
    "tracer": "o",
    "text_only": "s",
    "features_only": "^",
    "logreg_summary": "D",
    "logreg_last_turn": "P",
}


def build_loader(dialogues, features, tokenizer, max_length, batch_size):
    dataset = PrefixDialogueDataset(
        dialogues,
        features,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dialogue_batch,
    )


def select_prefix_turn(dialogue_length, ratio):
    return max(0, min(dialogue_length - 1, math.ceil(ratio * dialogue_length) - 1))


def filter_flat_prefix_examples(predictions, labels, dialogue_ids, turn_positions, dialogue_lengths, ratio):
    selected_preds = []
    selected_labels = []
    selected_turns = []
    selected_ids = []

    for pred, label, dial_id, turn_pos, dialogue_length in zip(
        predictions, labels, dialogue_ids, turn_positions, dialogue_lengths
    ):
        target_turn = select_prefix_turn(int(dialogue_length), ratio)
        if int(turn_pos) == target_turn:
            selected_preds.append(float(pred))
            selected_labels.append(float(label))
            selected_turns.append(int(turn_pos))
            selected_ids.append(dial_id)

    return (
        np.asarray(selected_preds),
        np.asarray(selected_labels),
        selected_ids,
        np.asarray(selected_turns),
    )


def truncate_dialogue_sequences(dialogue_predictions, ratio):
    truncated = {}
    for dial_id, preds in dialogue_predictions.items():
        if not preds:
            continue
        target_turn = select_prefix_turn(len(preds), ratio)
        truncated[dial_id] = preds[: target_turn + 1]
    return truncated


def evaluate_prefix_ratio(
    ratio,
    predictions,
    labels,
    dialogue_ids,
    turn_positions,
    dialogue_lengths,
    dialogue_predictions,
    dialogue_labels,
    frozen_threshold,
):
    sel_preds, sel_labels, sel_ids, sel_turns = filter_flat_prefix_examples(
        predictions,
        labels,
        dialogue_ids,
        turn_positions,
        dialogue_lengths,
        ratio,
    )
    truncated_dialogues = truncate_dialogue_sequences(dialogue_predictions, ratio)
    eds = compute_early_detection_score(
        truncated_dialogues,
        dialogue_labels,
        threshold=frozen_threshold,
    )
    f1m = compute_f1_metrics(sel_preds, sel_labels, threshold=frozen_threshold)
    return {
        "PrefixVisible": f"{int(ratio * 100)}%",
        "PrefixRatio": ratio,
        "AUC-ROC": float(compute_auc_roc(sel_preds, sel_labels)),
        "Precision": float(f1m["precision"]),
        "Recall": float(f1m["recall"]),
        "F1": float(f1m["f1"]),
        "Accuracy": float(f1m["accuracy"]),
        "EDS": float(eds["eds"]),
        "DetectionRate": float(eds["detection_rate"]),
        "MeanDetectionTurn": (
            float(eds["mean_detection_turn"])
            if np.isfinite(eds["mean_detection_turn"]) else None
        ),
        "MeanPredictionTurn": float(sel_turns.mean()) if len(sel_turns) else None,
        "NumDialogues": int(len(sel_ids)),
    }


def load_existing_rows(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"PrefixVisible", "PrefixRatio", "AUC-ROC", "F1", "EDS", "DetectionRate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Existing CSV {csv_path} is missing columns: {sorted(missing)}")
    rows = df.to_dict(orient="records")
    for row in rows:
        row["ModelKey"] = "tracer"
        row["Model"] = MODEL_DISPLAY_NAMES["tracer"]
    return rows


def _build_loader(dialogues, features, tokenizer, max_length, batch_size):
    return build_loader(dialogues, features, tokenizer, max_length, batch_size)


def evaluate_neural_model(
    model_key,
    model,
    loader,
    device,
    model_type,
    frozen_threshold,
):
    predictions, labels, dialogue_ids, turn_positions, dialogue_lengths = run_predictor(
        model, loader, device, model_type
    )
    dialogue_predictions, dialogue_labels = build_dialogue_prediction_sequences(
        predictions, labels, dialogue_ids, turn_positions
    )

    rows = []
    for ratio in PREFIX_RATIOS:
        row = evaluate_prefix_ratio(
            ratio,
            predictions,
            labels,
            dialogue_ids,
            turn_positions,
            dialogue_lengths,
            dialogue_predictions,
            dialogue_labels,
            frozen_threshold,
        )
        row["ModelKey"] = model_key
        row["Model"] = MODEL_DISPLAY_NAMES[model_key]
        rows.append(row)
    return rows


def evaluate_logreg_model(
    model_key,
    train_dialogues,
    train_features,
    test_dialogues,
    test_features,
    vectorizer,
    frozen_threshold,
    seed,
):
    X_train, y_train, _, _, _ = build_classical_prefix_dataset(
        train_dialogues, train_features, vectorizer=vectorizer
    )
    X_test, y_test, ids_test, turns_test, lengths_test = build_classical_prefix_dataset(
        test_dialogues, test_features, vectorizer=vectorizer
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict_proba(X_test)[:, 1]
    dialogue_predictions, dialogue_labels = build_dialogue_prediction_sequences(
        predictions, y_test, ids_test, turns_test
    )

    rows = []
    for ratio in PREFIX_RATIOS:
        row = evaluate_prefix_ratio(
            ratio,
            predictions,
            y_test,
            ids_test,
            turns_test,
            lengths_test,
            dialogue_predictions,
            dialogue_labels,
            frozen_threshold,
        )
        row["ModelKey"] = model_key
        row["Model"] = MODEL_DISPLAY_NAMES[model_key]
        rows.append(row)
    return rows


def plot_prefix_length_analysis(df, plot_path, metric="AUC-ROC"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(12.0, 5.25))
    ordered_models = [m for m in DEFAULT_MODELS if m in set(df["ModelKey"])]
    for model_key in ordered_models:
        subdf = df[df["ModelKey"] == model_key].sort_values("PrefixRatio")
        x = subdf["PrefixRatio"].astype(float)
        ax.plot(
            x,
            subdf[metric].astype(float),
            marker=MODEL_MARKERS.get(model_key, "o"),
            linewidth=3.4,
            markersize=8,
            color=MODEL_COLORS.get(model_key),
            label=MODEL_DISPLAY_NAMES.get(model_key, model_key),
        )

    x = sorted(df["PrefixRatio"].astype(float).unique().tolist())
    visible_labels = (
        df.sort_values("PrefixRatio")
        .drop_duplicates("PrefixRatio")["PrefixVisible"]
        .tolist()
    )
    ax.set_xlabel("Visible dialogue prefix", fontsize=18, fontweight="bold")
    ax.set_ylabel(metric, fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(visible_labels)
    ax.tick_params(axis="both", which="major", labelsize=16)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(ordered_models), 3),
        frameon=True,
        prop={"weight": "bold", "size": 15},
    )
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Prefix-length analysis for TRACER and baselines")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--checkpoint", type=str, default="outputs/best_predictor.pt")
    parser.add_argument("--dataset", type=str, default="mwoz")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to include in the fixed-prefix analysis.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--plot-dir", type=str, default="plots")
    parser.add_argument(
        "--reuse-existing-tracer-csv",
        type=str,
        default=None,
        help="Optional existing TRACER fixed-prefix CSV to reuse instead of rerunning the TRACER checkpoint.",
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="AUC-ROC",
        choices=["AUC-ROC", "F1", "DetectionRate", "EDS"],
        help="Metric used for the comparison plot.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = cfg.data.get("cache_dir", "cache")
    tokenizer_name = cfg.model.stream_b.get("encoder_name", "roberta-base")
    tokenizer = BeliefStateEncoder.get_tokenizer(tokenizer_name)
    max_length = cfg.model.stream_b.get("max_length", 512)
    batch_size = cfg.training.get("batch_size", 32)

    test_dialogues, test_features = load_unified_and_features(cache_dir, "test", args.dataset)
    test_loader = _build_loader(test_dialogues, test_features, tokenizer, max_length, batch_size)

    results_json = out_dir / "evaluate_full_results.json"
    frozen_threshold = 0.5
    all_results = {}
    if results_json.exists():
        all_results = json.loads(results_json.read_text())
        frozen_threshold = all_results.get("optimal_threshold", {}).get("optimal_threshold", 0.5)

    rows = []
    requested_models = list(dict.fromkeys(args.models))

    if "tracer" in requested_models:
        if args.reuse_existing_tracer_csv:
            rows.extend(load_existing_rows(args.reuse_existing_tracer_csv))
        else:
            model = TRACERPredictor(
                stream_a_config=dict(cfg.model.stream_a),
                stream_b_config=dict(cfg.model.stream_b),
                fusion_config=dict(cfg.model.fusion),
            )
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            rows.extend(
                evaluate_neural_model(
                    "tracer",
                    model,
                    test_loader,
                    device,
                    "tracer",
                    frozen_threshold,
                )
            )

    if "text_only" in requested_models:
        text_model = TextOnlyPredictor(dict(cfg.model.stream_b))
        ckpt = torch.load(
            cfg.model.get("text_only_checkpoint"),
            map_location=device,
            weights_only=False,
        )
        text_model.load_state_dict(ckpt["model_state_dict"])
        text_model.to(device)
        text_model.eval()
        rows.extend(
            evaluate_neural_model(
                "text_only",
                text_model,
                test_loader,
                device,
                "text_only",
                frozen_threshold,
            )
        )

    if "features_only" in requested_models:
        features_model = FeaturesOnlyPredictor(dict(cfg.model.stream_a))
        ckpt = torch.load(
            cfg.model.get("features_only_checkpoint"),
            map_location=device,
            weights_only=False,
        )
        features_model.load_state_dict(ckpt["model_state_dict"])
        features_model.to(device)
        features_model.eval()
        rows.extend(
            evaluate_neural_model(
                "features_only",
                features_model,
                test_loader,
                device,
                "features_only",
                frozen_threshold,
            )
        )

    if "logreg_summary" in requested_models or "logreg_last_turn" in requested_models:
        train_dialogues, train_features = load_unified_and_features(cache_dir, "train", args.dataset)
        seed = cfg.training.get("seed", 42)

        if "logreg_summary" in requested_models:
            rows.extend(
                evaluate_logreg_model(
                    "logreg_summary",
                    train_dialogues,
                    train_features,
                    test_dialogues,
                    test_features,
                    "summary",
                    frozen_threshold,
                    seed,
                )
            )

        if "logreg_last_turn" in requested_models:
            rows.extend(
                evaluate_logreg_model(
                    "logreg_last_turn",
                    train_dialogues,
                    train_features,
                    test_dialogues,
                    test_features,
                    "last_turn",
                    frozen_threshold,
                    seed,
                )
            )

    df = pd.DataFrame(rows)
    csv_stem = "prefix_length_analysis"
    if set(requested_models) != {"tracer"}:
        csv_stem = "prefix_length_analysis_multimodel"
    csv_path = out_dir / f"{csv_stem}.csv"
    df.to_csv(csv_path, index=False)

    plot_name = "prefix_length_analysis"
    if set(requested_models) != {"tracer"}:
        plot_name = "prefix_length_analysis_multimodel"
    plot_path = plot_dir / f"{plot_name}.pdf"
    plot_prefix_length_analysis(df, plot_path, metric=args.plot_metric)

    all_results["prefix_length_analysis"] = {
        "prefixes": PREFIX_RATIOS,
        "frozen_threshold": frozen_threshold,
        "models": requested_models,
        "plot_metric": args.plot_metric,
        "outputs": {
            "csv": str(csv_path),
            "plot": str(plot_path),
        },
        "rows": df.to_dict(orient="records"),
    }
    results_json.write_text(json.dumps(all_results, indent=2))

    print("Prefix-length analysis saved:")
    print(f"- {csv_path}")
    print(f"- {plot_path}")
    print(f"- frozen_threshold={frozen_threshold}")


if __name__ == "__main__":
    main()
