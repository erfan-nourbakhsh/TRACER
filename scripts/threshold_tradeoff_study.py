
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_full import (
    load_config,
    load_unified_and_features,
    run_predictor,
    build_dialogue_prediction_sequences,
)
from tracer.data.unified import PrefixDialogueDataset, collate_dialogue_batch
from tracer.evaluation.metrics import compute_f1_metrics, compute_early_detection_score
from tracer.models.belief_encoder import BeliefStateEncoder
from tracer.models.dual_stream import TRACERPredictor
from tracer.recovery.threshold import find_pareto_threshold


def compute_false_positive_rate(dialogue_predictions, dialogue_labels, threshold):
    n_fp = 0
    n_success = 0
    for dial_id, preds in dialogue_predictions.items():
        if dialogue_labels.get(dial_id, True):
            n_success += 1
            if any(p > threshold for p in preds):
                n_fp += 1
    return n_fp / max(n_success, 1)


def evaluate_threshold_grid(predictions, labels, dial_predictions, dial_labels, thresholds, split_name):
    rows = []
    for threshold in thresholds:
        f1m = compute_f1_metrics(predictions, labels, threshold=threshold)
        eds = compute_early_detection_score(dial_predictions, dial_labels, threshold=threshold)
        rows.append({
            "split": split_name,
            "threshold": float(threshold),
            "precision": float(f1m["precision"]),
            "recall": float(f1m["recall"]),
            "f1": float(f1m["f1"]),
            "accuracy": float(f1m["accuracy"]),
            "eds": float(eds["eds"]),
            "detection_rate": float(eds["detection_rate"]),
            "mean_detection_turn": (
                float(eds["mean_detection_turn"])
                if np.isfinite(eds["mean_detection_turn"]) else None
            ),
            "false_positive_rate": float(
                compute_false_positive_rate(dial_predictions, dial_labels, threshold)
            ),
            "n_failed": int(eds["n_failed"]),
            "n_detected": int(eds["n_detected"]),
            "detected_at_turn_zero": int(eds["detected_at_turn_zero"]),
        })
    return pd.DataFrame(rows)


def select_summary_rows(dev_df, test_df, frozen_threshold, pareto_result):
    def _nearest(df, value):
        idx = (df["threshold"] - value).abs().idxmin()
        return df.loc[idx].copy()

    frozen_dev = _nearest(dev_df, frozen_threshold)
    frozen_dev["selection"] = "frozen_threshold_dev"
    frozen_test = _nearest(test_df, frozen_threshold)
    frozen_test["selection"] = "frozen_threshold_test"

    best_eds_dev = dev_df.loc[dev_df["eds"].idxmax()].copy()
    best_eds_dev["selection"] = "best_eds_dev"

    best_f1_dev = dev_df.loc[dev_df["f1"].idxmax()].copy()
    best_f1_dev["selection"] = "best_f1_dev"

    nontrivial = dev_df[dev_df["detection_rate"] > 0]
    if len(nontrivial) > 0:
        lowest_fpr_dev = nontrivial.loc[nontrivial["false_positive_rate"].idxmin()].copy()
    else:
        lowest_fpr_dev = dev_df.loc[dev_df["false_positive_rate"].idxmin()].copy()
    lowest_fpr_dev["selection"] = "lowest_fpr_nontrivial_dev"

    pareto_dev = _nearest(dev_df, pareto_result.get("optimal_threshold", frozen_threshold))
    pareto_dev["selection"] = "pareto_threshold_dev"

    return pd.DataFrame([frozen_dev, frozen_test, best_eds_dev, best_f1_dev, lowest_fpr_dev, pareto_dev])


def plot_tradeoff(dev_df, frozen_threshold, plot_path):
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
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dev_df["threshold"], dev_df["eds"], label="EDS", linewidth=2.8)
    ax1.plot(dev_df["threshold"], dev_df["f1"], label="F1", linewidth=2.8)
    ax1.plot(dev_df["threshold"], dev_df["detection_rate"], label="Detection rate", linewidth=2.8)
    ax1.plot(dev_df["threshold"], dev_df["false_positive_rate"], label="False positive rate", linewidth=2.8)
    ax1.set_xlabel("Threshold", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Score / Rate", fontsize=18, fontweight="bold")
    ax1.tick_params(axis="both", which="major", labelsize=16)
    for tick_label in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.axvline(frozen_threshold, color="black", linestyle=":", linewidth=2.2, label="Frozen threshold")

    ax2 = ax1.twinx()
    mean_turns = dev_df["mean_detection_turn"].astype(float)
    ax2.plot(dev_df["threshold"], mean_turns, color="gray", linestyle="--", linewidth=2.8, label="Mean detection turn")
    ax2.set_ylabel(
        "Mean detection turn",
        fontsize=18,
        fontweight="bold",
        rotation=270,
        labelpad=24,
    )
    ax2.tick_params(axis="y", which="major", labelsize=16)
    for tick_label in ax2.get_yticklabels():
        tick_label.set_fontweight("bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        prop={"weight": "bold", "size": 13},
    )

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


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


def main():
    parser = argparse.ArgumentParser(description="Threshold tradeoff study for TRACER")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--checkpoint", type=str, default="outputs/best_predictor.pt")
    parser.add_argument("--dataset", type=str, default="mwoz")
    parser.add_argument("--n-thresholds", type=int, default=100)
    parser.add_argument("--threshold-min", type=float, default=0.1)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--plot-dir", type=str, default="plots")
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

    dev_dialogues, dev_features = load_unified_and_features(cache_dir, "dev", args.dataset)
    test_dialogues, test_features = load_unified_and_features(cache_dir, "test", args.dataset)

    model = TRACERPredictor(
        stream_a_config=dict(cfg.model.stream_a),
        stream_b_config=dict(cfg.model.stream_b),
        fusion_config=dict(cfg.model.fusion),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    dev_loader = build_loader(dev_dialogues, dev_features, tokenizer, max_length, batch_size)
    test_loader = build_loader(test_dialogues, test_features, tokenizer, max_length, batch_size)

    dev_preds, dev_labels, dev_ids, dev_turns, _ = run_predictor(model, dev_loader, device, "tracer")
    test_preds, test_labels, test_ids, test_turns, _ = run_predictor(model, test_loader, device, "tracer")

    dev_dial_preds, dev_dial_labels = build_dialogue_prediction_sequences(dev_preds, dev_labels, dev_ids, dev_turns)
    test_dial_preds, test_dial_labels = build_dialogue_prediction_sequences(test_preds, test_labels, test_ids, test_turns)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.n_thresholds)
    dev_df = evaluate_threshold_grid(dev_preds, dev_labels, dev_dial_preds, dev_dial_labels, thresholds, "dev")
    test_df = evaluate_threshold_grid(test_preds, test_labels, test_dial_preds, test_dial_labels, thresholds, "test")

    dev_csv = out_dir / "threshold_tradeoff_dev.csv"
    test_csv = out_dir / "threshold_tradeoff_test.csv"
    dev_df.to_csv(dev_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    results_json = out_dir / "evaluate_full_results.json"
    frozen_threshold = 0.5
    results = {}
    if results_json.exists():
        results = json.loads(results_json.read_text())
        frozen_threshold = results.get("optimal_threshold", {}).get("optimal_threshold", 0.5)

    pareto_result = find_pareto_threshold(
        dev_dial_preds,
        dev_dial_labels,
        n_thresholds=args.n_thresholds,
        max_fpr=cfg.recovery.get("max_fpr", 0.15),
    )
    summary_df = select_summary_rows(dev_df, test_df, frozen_threshold, pareto_result)
    summary_csv = out_dir / "paper_threshold_tradeoff_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plot_path = plot_dir / "threshold_tradeoff.pdf"
    plot_tradeoff(dev_df, frozen_threshold, plot_path)

    summary_block = {
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "n_thresholds": args.n_thresholds,
        "frozen_threshold": frozen_threshold,
        "best_f1_threshold_dev": float(dev_df.loc[dev_df["f1"].idxmax(), "threshold"]),
        "best_eds_threshold_dev": float(dev_df.loc[dev_df["eds"].idxmax(), "threshold"]),
        "pareto_threshold_dev": float(pareto_result.get("optimal_threshold", frozen_threshold)),
        "max_fpr_constraint": float(cfg.recovery.get("max_fpr", 0.15)),
        "outputs": {
            "dev_csv": str(dev_csv),
            "test_csv": str(test_csv),
            "summary_csv": str(summary_csv),
            "plot": str(plot_path),
        },
    }
    results["threshold_tradeoff_summary"] = summary_block
    results_json.write_text(json.dumps(results, indent=2))

    print("Threshold tradeoff study saved:")
    print(f"- {dev_csv}")
    print(f"- {test_csv}")
    print(f"- {summary_csv}")
    print(f"- {plot_path}")
    print(f"- frozen_threshold={frozen_threshold}")


if __name__ == "__main__":
    main()
