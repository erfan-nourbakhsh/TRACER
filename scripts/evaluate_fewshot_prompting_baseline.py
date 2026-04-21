
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_prompting_baseline import (
    MODEL_NAME_ALIASES,
    SYSTEM_PROMPT,
    belief_state_to_text,
    build_dialogue_block,
    build_dialogue_prediction_sequences,
    compute_auc_roc,
    compute_auc_roc_per_turn,
    compute_early_detection_score,
    compute_f1_metrics,
    load_config,
    load_unified_dialogues,
    normalize_prediction,
    resolve_model_name,
    slugify_model_name,
    try_parse_json,
)
from tracer.data.unified import UnifiedDialogue


DEFAULT_SUPPORT_IDS = [
    "PMUL1181.json",
    "PMUL0287.json",
    "PMUL1635.json",
]


TARGET_PROMPT_TEMPLATE = """Now classify the target dialogue prefix.

Visible prefix:
{dialogue_block}

Current cumulative belief state:
{belief_text}

Return JSON with exactly this schema:
{{
  "prediction": "FAILURE" or "SUCCESS",
  "confidence": <integer from 0 to 100>,
  "rationale": "<one short sentence>"
}}
"""


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n"


def get_dialogue_by_id(dialogues: List[UnifiedDialogue], dialogue_id: str) -> UnifiedDialogue:
    for dialogue in dialogues:
        if dialogue.dialogue_id == dialogue_id:
            return dialogue
    raise ValueError(f"Support dialogue {dialogue_id!r} was not found in dev split.")


def format_support_example(dialogue: UnifiedDialogue, max_visible_turns: int = None) -> str:
    end_turn_idx = len(dialogue.turns) - 1
    turn = dialogue.turns[end_turn_idx]
    label = "SUCCESS" if dialogue.success else "FAILURE"
    confidence = 95
    dialogue_block = build_dialogue_block(
        dialogue,
        end_turn_idx,
        max_visible_turns=max_visible_turns,
    )
    belief_text = belief_state_to_text(turn.belief_state)
    return f"""Example dialogue prefix:
{dialogue_block}

Current cumulative belief state:
{belief_text}

Answer:
{{
  "prediction": "{label}",
  "confidence": {confidence},
  "rationale": "The visible dialogue state is labeled {label.lower()} in the development example."
}}"""


def build_fewshot_context(
    dev_dialogues: List[UnifiedDialogue],
    support_ids: List[str],
    max_support_turns: int = None,
) -> Tuple[str, List[Dict[str, str]]]:
    examples = []
    metadata = []
    for dialogue_id in support_ids:
        dialogue = get_dialogue_by_id(dev_dialogues, dialogue_id)
        examples.append(format_support_example(dialogue, max_visible_turns=max_support_turns))
        metadata.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "label": "success" if dialogue.success else "fail",
                "num_turns": len(dialogue.turns),
            }
        )

    context = (
        "Use the fixed labeled development examples below as few-shot guidance. "
        "Then classify the target dialogue prefix using only the target prefix.\n\n"
        + "\n\n---\n\n".join(examples)
    )
    return context, metadata


def build_target_prompt(
    dialogue: UnifiedDialogue,
    end_turn_idx: int,
    fewshot_context: str,
    max_visible_turns: int = None,
    cot: bool = False,
) -> str:
    turn = dialogue.turns[end_turn_idx]
    dialogue_block = build_dialogue_block(
        dialogue,
        end_turn_idx,
        max_visible_turns=max_visible_turns,
    )
    belief_text = belief_state_to_text(turn.belief_state)
    prompt = (
        fewshot_context
        + "\n\n---\n\n"
        + TARGET_PROMPT_TEMPLATE.format(
            dialogue_block=dialogue_block,
            belief_text=belief_text,
        ).rstrip()
    )
    if cot:
        prompt = f"{prompt}\n\nThink step by step."
    return prompt


def evaluate_predictions(predictions, labels, dialogue_ids, turn_positions):
    dialogue_predictions, dialogue_labels = build_dialogue_prediction_sequences(
        predictions,
        labels,
        dialogue_ids,
        turn_positions,
    )
    metrics = {
        "auc_roc": compute_auc_roc(predictions, labels),
        "f1_at_0.5": compute_f1_metrics(predictions, labels, threshold=0.5),
        "per_turn_auc": compute_auc_roc_per_turn(predictions, labels, turn_positions),
        "early_detection": compute_early_detection_score(
            dialogue_predictions,
            dialogue_labels,
            threshold=0.5,
        ),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed few-shot prompting baseline with vLLM."
    )
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--support_split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--dataset_filter", default="mwoz")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--support_ids", nargs="+", default=DEFAULT_SUPPORT_IDS)
    parser.add_argument("--max_visible_turns", type=int, default=8)
    parser.add_argument("--max_support_turns", type=int, default=8)
    parser.add_argument("--max_dialogues", type=int, default=None)
    parser.add_argument("--max_prefixes_per_dialogue", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    dataset_filter = None if args.dataset_filter == "all" else args.dataset_filter
    model_name = resolve_model_name(args.model_name)

    dev_dialogues = load_unified_dialogues(
        cache_dir,
        args.support_split,
        dataset_filter=dataset_filter,
    )
    test_dialogues = load_unified_dialogues(
        cache_dir,
        args.split,
        dataset_filter=dataset_filter,
    )
    if args.max_dialogues is not None:
        test_dialogues = test_dialogues[: args.max_dialogues]
    if not dev_dialogues:
        raise RuntimeError("No support dialogues found.")
    if not test_dialogues:
        raise RuntimeError("No target dialogues found.")

    fewshot_context, support_metadata = build_fewshot_context(
        dev_dialogues,
        args.support_ids,
        max_support_turns=args.max_support_turns,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    prompt_items = []
    prompts = []
    for dialogue in test_dialogues:
        max_prefixes = len(dialogue.turns)
        if args.max_prefixes_per_dialogue is not None:
            max_prefixes = min(max_prefixes, args.max_prefixes_per_dialogue)
        for end_turn_idx in range(max_prefixes):
            user_prompt = build_target_prompt(
                dialogue,
                end_turn_idx,
                fewshot_context=fewshot_context,
                max_visible_turns=args.max_visible_turns,
                cot=args.cot,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompts.append(apply_chat_template(tokenizer, messages))
            prompt_items.append((dialogue, end_turn_idx))

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    predictions = []
    labels = []
    dialogue_ids = []
    turn_positions = []
    raw_records = []

    for (dialogue, end_turn_idx), output in zip(prompt_items, outputs):
        response = output.outputs[0].text.strip() if output.outputs else ""
        parsed = try_parse_json(response)
        normalized = normalize_prediction(parsed, response)
        normalized["raw_response"] = response
        label = 0.0 if dialogue.success else 1.0

        predictions.append(normalized["prob_failure"])
        labels.append(label)
        dialogue_ids.append(dialogue.dialogue_id)
        turn_positions.append(end_turn_idx)
        raw_records.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "dataset": dialogue.dataset,
                "turn_position": end_turn_idx,
                "label": label,
                "prediction": normalized["prediction"],
                "confidence": normalized["confidence"],
                "prob_failure": normalized["prob_failure"],
                "rationale": normalized["rationale"],
                "raw_response": normalized["raw_response"],
            }
        )

    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    turn_positions = np.asarray(turn_positions, dtype=np.int64)
    metrics = evaluate_predictions(predictions, labels, dialogue_ids, turn_positions)

    result = {
        "config": {
            "split": args.split,
            "support_split": args.support_split,
            "dataset_filter": args.dataset_filter,
            "requested_model_name": args.model_name,
            "model_name": model_name,
            "support_ids": args.support_ids,
            "support_examples": support_metadata,
            "max_visible_turns": args.max_visible_turns,
            "max_support_turns": args.max_support_turns,
            "max_dialogues": args.max_dialogues,
            "max_prefixes_per_dialogue": args.max_prefixes_per_dialogue,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "cot": args.cot,
        },
        "metrics": metrics,
        "num_dialogues": len(test_dialogues),
        "num_prefixes": len(raw_records),
        "raw_predictions": raw_records,
    }

    output_json = args.output_json
    if output_json is None:
        model_slug = slugify_model_name(model_name)
        output_json = os.path.join(
            cfg.output.get("checkpoint_dir", "outputs"),
            f"fewshot_prompting_baseline_{args.dataset_filter}_{args.split}_{model_slug}.json",
        )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(
        json.dumps(
            {
                "output_json": output_json,
                "support_examples": support_metadata,
                "num_dialogues": len(test_dialogues),
                "num_prefixes": len(raw_records),
                "auc_roc": metrics["auc_roc"],
                "f1_at_0.5": metrics["f1_at_0.5"],
                "early_detection": metrics["early_detection"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
