
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.unified import UnifiedDialogue, UnifiedTurn
from tracer.data.domain_slots import get_required_slots_mwoz
from tracer.features.trajectory import compute_dialogue_features
from tracer.features.conflict_detector import ConflictDetector
from tracer.features.abcd_features import (
    compute_abcd_dialogue_features,
)
from tracer.data.abcd_loader import load_abcd_kb


def load_config(config_path: str):
    import omegaconf
    return omegaconf.OmegaConf.load(config_path)


def load_unified_from_cache(cache_dir: str, split: str, suffix: str = ""):
    suffix_part = f"_{suffix}" if suffix else ""
    path = os.path.join(cache_dir, f"unified_{split}{suffix_part}.json")
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


def required_slots_fn(domain: str):
    return get_required_slots_mwoz(domain, has_booking=True)


def main():
    parser = argparse.ArgumentParser(description="Compute trajectory features for cached dialogues")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--input_suffix", type=str, default="",
                        help="Read unified_{split}_{input_suffix}.json instead of unified_{split}.json")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Write features_{split}_{output_suffix}.json instead of features_{split}.json")
    parser.add_argument("--use_conflict", action="store_true", default=True,
                        help="Use conflict detector (sentence-transformers)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    conflict_model = cfg.features.get("conflict_model", "sentence-transformers/all-MiniLM-L6-v2")
    conflict_threshold = cfg.features.get("conflict_threshold", 0.3)

    conflict_detector = None
    if args.use_conflict:
        try:
            conflict_detector = ConflictDetector(
                model_name=conflict_model,
                threshold=conflict_threshold,
            )
        except Exception as e:
            print(f"Conflict detector not loaded: {e}. Proceeding without conflict feature.")

    abcd_kb = None
    abcd_kb_path = cfg.data.get("abcd_kb_path")
    if abcd_kb_path and os.path.exists(abcd_kb_path):
        abcd_kb = load_abcd_kb(abcd_kb_path)

    for split in args.splits:
        dialogues = load_unified_from_cache(cache_dir, split, suffix=args.input_suffix)
        if not dialogues:
            print(f"No dialogues for {split}, skipping.")
            continue

        features_dict = {}
        for dial in dialogues:
            if dial.dataset == "abcd":
                if abcd_kb is None:
                    feat_list = [np.zeros(5, dtype=np.float32) for _ in dial.turns]
                else:
                    feat_list = compute_abcd_dialogue_features(dial, abcd_kb)
            else:
                feat_list = compute_dialogue_features(
                    dial,
                    required_slots_fn=required_slots_fn,
                    conflict_detector=conflict_detector,
                )
            features_dict[dial.dialogue_id] = [f.tolist() if hasattr(f, "tolist") else list(f) for f in feat_list]

        suffix_part = f"_{args.output_suffix}" if args.output_suffix else ""
        out_path = os.path.join(cache_dir, f"features_{split}{suffix_part}.json")
        with open(out_path, "w") as f:
            json.dump(features_dict, f)
        print(f"Saved features for {len(features_dict)} dialogues ({split}) to {out_path}")

    print("Feature computation complete.")


if __name__ == "__main__":
    main()
