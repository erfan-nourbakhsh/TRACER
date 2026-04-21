
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.mwoz_loader import build_mwoz_dialogues
from tracer.data.sgd_loader import load_sgd_dialogues
from tracer.data.abcd_loader import load_abcd_dialogues
from tracer.data.unified import (
    convert_mwoz_to_unified,
    convert_sgd_to_unified,
    convert_abcd_to_unified,
    UnifiedDialogue,
)


def _dialogue_to_dict(d: UnifiedDialogue):
    turns = []
    for t in d.turns:
        turns.append({
            "turn_idx": t.turn_idx,
            "user_utterance": t.user_utterance,
            "system_utterance": t.system_utterance,
            "belief_state": t.belief_state,
            "turn_delta": t.turn_delta,
            "domain": t.domain,
        })
    return {
        "dialogue_id": d.dialogue_id,
        "dataset": d.dataset,
        "domains": d.domains,
        "turns": turns,
        "success": d.success,
        "metadata": d.metadata,
    }


def _dict_to_dialogue(d: dict) -> UnifiedDialogue:
    from tracer.data.unified import UnifiedTurn
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
    return UnifiedDialogue(
        dialogue_id=d["dialogue_id"],
        dataset=d["dataset"],
        domains=d["domains"],
        turns=turns,
        success=d["success"],
        metadata=d.get("metadata", {}),
    )


def load_config(config_path: str):
    import omegaconf
    return omegaconf.OmegaConf.load(config_path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MultiWOZ, SGD, ABCD to unified format")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--datasets", type=str, nargs="+", default=["mwoz", "sgd", "abcd"],
                        help="Which datasets to process")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    for split in args.splits:
        all_dialogues = []

        if "mwoz" in args.datasets:
            processed_dir = cfg.data.get("mwoz_processed_dir")
            raw_path = cfg.data.get("mwoz_raw_path")
            if processed_dir and raw_path and os.path.exists(processed_dir):
                mwoz_dialogues = build_mwoz_dialogues(processed_dir, raw_path, split)
                for d in mwoz_dialogues:
                    all_dialogues.append(convert_mwoz_to_unified(d))
                print(f"MultiWOZ {split}: {len(mwoz_dialogues)} dialogues")
            else:
                print(f"MultiWOZ paths not found or not in config, skipping {split}")

        if "sgd" in args.datasets:
            sgd_dir = cfg.data.get("sgd_dir")
            if sgd_dir and os.path.exists(sgd_dir):
                sgd_dialogues = load_sgd_dialogues(sgd_dir, split)
                for d in sgd_dialogues:
                    all_dialogues.append(convert_sgd_to_unified(d))
                print(f"SGD {split}: {len(sgd_dialogues)} dialogues")
            else:
                print(f"SGD path not found, skipping {split}")

        if "abcd" in args.datasets:
            abcd_data = cfg.data.get("abcd_data_path")
            abcd_kb = cfg.data.get("abcd_kb_path")
            if abcd_data and abcd_kb and os.path.exists(abcd_data):
                abcd_dialogues = load_abcd_dialogues(abcd_data, abcd_kb, split)
                for d in abcd_dialogues:
                    all_dialogues.append(convert_abcd_to_unified(d))
                print(f"ABCD {split}: {len(abcd_dialogues)} dialogues")
            else:
                print(f"ABCD paths not found, skipping {split}")

        out_path = os.path.join(cache_dir, f"unified_{split}.json")
        with open(out_path, "w") as f:
            json.dump([_dialogue_to_dict(d) for d in all_dialogues], f, indent=0)
        print(f"Saved {len(all_dialogues)} unified dialogues to {out_path}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
