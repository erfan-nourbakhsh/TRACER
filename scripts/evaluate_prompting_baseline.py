
import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.unified import UnifiedDialogue, UnifiedTurn
from tracer.evaluation.metrics import (
    compute_auc_roc,
    compute_auc_roc_per_turn,
    compute_early_detection_score,
    compute_f1_metrics,
)


SYSTEM_PROMPT = (
    "You are evaluating a task-oriented dialogue prefix. "
    "Your job is to forecast whether the full dialogue will eventually end in "
    "SUCCESS or FAILURE. Use only the visible prefix. "
    "Respond with valid JSON only."
)


MODEL_NAME_ALIASES = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


USER_PROMPT_TEMPLATE = """Given the task-oriented dialogue prefix below, predict whether the full dialogue will eventually succeed or fail.

Rules:
- Predict FAILURE if the dialogue appears likely to end without satisfying the user's full goal.
- Predict SUCCESS if the visible evidence suggests the task is likely to be completed.
- Use the visible prefix only. Do not assume future repair unless strongly supported.
- Return valid JSON only.

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


def load_config(config_path: str):
    import omegaconf

    return omegaconf.OmegaConf.load(config_path)


def resolve_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def slugify_model_name(model_name: str) -> str:
    slug = model_name.strip().lower()
    slug = slug.replace("/", "_")
    slug = re.sub(r"[^a-z0-9._-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_.-")
    return slug or "model"


def load_unified_dialogues(cache_dir: str, split: str, dataset_filter: str = None):
    path = os.path.join(cache_dir, f"unified_{split}.json")
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
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
        dialogues.append(
            UnifiedDialogue(
                dialogue_id=d["dialogue_id"],
                dataset=d["dataset"],
                domains=d["domains"],
                turns=turns,
                success=d["success"],
                metadata=d.get("metadata", {}),
            )
        )
    return dialogues


def belief_state_to_text(belief_state: Dict[str, str]) -> str:
    if not belief_state:
        return "empty state"
    parts = []
    for slot, value in sorted(belief_state.items()):
        slot_text = slot.replace("-", " ").replace("_", " ")
        parts.append(f"{slot_text} is {value}")
    return ", ".join(parts)


def build_dialogue_block(
    dialogue: UnifiedDialogue,
    end_turn_idx: int,
    max_visible_turns: int = None,
) -> str:
    start_idx = 0
    if max_visible_turns is not None and max_visible_turns > 0:
        start_idx = max(0, end_turn_idx + 1 - max_visible_turns)

    lines = []
    for t_idx in range(start_idx, end_turn_idx + 1):
        turn = dialogue.turns[t_idx]
        lines.append(f"Turn {t_idx} User: {turn.user_utterance}")
        if turn.system_utterance:
            lines.append(f"Turn {t_idx} System: {turn.system_utterance}")
    return "\n".join(lines)


def build_messages(
    dialogue: UnifiedDialogue,
    end_turn_idx: int,
    max_visible_turns: int = None,
    cot: bool = False,
) -> List[Dict[str, str]]:
    turn = dialogue.turns[end_turn_idx]
    dialogue_block = build_dialogue_block(
        dialogue, end_turn_idx, max_visible_turns=max_visible_turns
    )
    belief_text = belief_state_to_text(turn.belief_state)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        dialogue_block=dialogue_block,
        belief_text=belief_text,
    )
    if cot:
        user_prompt = f"{user_prompt.rstrip()}\n\nThink step by step."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def try_parse_json(text: str):
    text = (text or "").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def normalize_prediction(parsed: Dict, raw_text: str):
    pred = ""
    confidence = None
    rationale = ""

    if parsed:
        pred = str(parsed.get("prediction", "")).strip().upper()
        confidence = parsed.get("confidence")
        rationale = str(parsed.get("rationale", "")).strip()

    if pred not in {"FAILURE", "SUCCESS"}:
        upper = (raw_text or "").upper()
        if "FAILURE" in upper:
            pred = "FAILURE"
        elif "SUCCESS" in upper:
            pred = "SUCCESS"

    try:
        confidence = int(confidence)
    except Exception:
        confidence = None

    if confidence is None:
        numbers = re.findall(r"\b([0-9]{1,3})\b", raw_text or "")
        confidence = int(numbers[0]) if numbers else 50

    confidence = max(0, min(100, confidence))

    if pred == "FAILURE":
        prob_failure = confidence / 100.0
    elif pred == "SUCCESS":
        prob_failure = 1.0 - (confidence / 100.0)
    else:
        pred = "FAILURE" if confidence >= 50 else "SUCCESS"
        prob_failure = 0.5

    return {
        "prediction": pred,
        "confidence": confidence,
        "prob_failure": float(prob_failure),
        "rationale": rationale,
    }


class PromptingBaseline:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        cot: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cot = cot
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()

    def predict_prefix(self, dialogue: UnifiedDialogue, end_turn_idx: int, max_visible_turns: int):
        messages = build_messages(
            dialogue,
            end_turn_idx,
            max_visible_turns=max_visible_turns,
            cot=self.cot,
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{messages[-1]['content']}\n"
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()
        parsed = try_parse_json(response)
        normalized = normalize_prediction(parsed, response)
        normalized["raw_response"] = response
        return normalized


class VLLMPromptingBaseline:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 2048,
        cot: bool = False,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cot = cot
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    def build_prompt(self, dialogue: UnifiedDialogue, end_turn_idx: int, max_visible_turns: int):
        messages = build_messages(
            dialogue,
            end_turn_idx,
            max_visible_turns=max_visible_turns,
            cot=self.cot,
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SYSTEM_PROMPT}\n\n{messages[-1]['content']}\n"

    def predict_prompts(self, prompts: List[str]):
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
        )
        results = []
        for output in outputs:
            response = output.outputs[0].text.strip() if output.outputs else ""
            parsed = try_parse_json(response)
            normalized = normalize_prediction(parsed, response)
            normalized["raw_response"] = response
            results.append(normalized)
        return results


def build_dialogue_prediction_sequences(predictions, labels, dialogue_ids, turn_positions):
    dialogue_predictions = {}
    dialogue_labels = {}

    for pred, label, dial_id, turn_idx in zip(predictions, labels, dialogue_ids, turn_positions):
        dialogue_predictions.setdefault(dial_id, []).append((int(turn_idx), float(pred)))
        dialogue_labels[dial_id] = bool(label == 0)

    sorted_predictions = {
        dial_id: [pred for _, pred in sorted(seq, key=lambda x: x[0])]
        for dial_id, seq in dialogue_predictions.items()
    }
    return sorted_predictions, dialogue_labels


def evaluate_predictions(predictions, labels, dialogue_ids, turn_positions):
    dialogue_predictions, dialogue_labels = build_dialogue_prediction_sequences(
        predictions, labels, dialogue_ids, turn_positions
    )
    metrics = {
        "auc_roc": compute_auc_roc(predictions, labels),
        "f1_at_0.5": compute_f1_metrics(predictions, labels, threshold=0.5),
        "per_turn_auc": compute_auc_roc_per_turn(predictions, labels, turn_positions),
        "early_detection": compute_early_detection_score(
            dialogue_predictions, dialogue_labels, threshold=0.5
        ),
    }
    return metrics, dialogue_predictions, dialogue_labels


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a simple prompting-based failure forecasting baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
    )
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default="mwoz",
        help="mwoz, sgd, abcd, or all",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Instruction-tuned causal LM used for zero-shot prompting",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Generation backend.",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help='Append "Think step by step." at the end of the user prompt.',
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum model length for vLLM.",
    )
    parser.add_argument(
        "--max_visible_turns",
        type=int,
        default=8,
        help="Limit prompt context to the last N visible turns",
    )
    parser.add_argument(
        "--max_dialogues",
        type=int,
        default=None,
        help="Optional cap on number of dialogues to evaluate",
    )
    parser.add_argument(
        "--max_prefixes_per_dialogue",
        type=int,
        default=None,
        help="Optional cap on number of prefixes per dialogue",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save metrics and raw predictions",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    dataset_filter = None if args.dataset_filter == "all" else args.dataset_filter
    requested_model_name = args.model_name or cfg.recovery.get(
        "llm_model", "meta-llama/Llama-3.1-8B-Instruct"
    )
    model_name = resolve_model_name(requested_model_name)

    dialogues = load_unified_dialogues(cache_dir, args.split, dataset_filter=dataset_filter)
    if args.max_dialogues is not None:
        dialogues = dialogues[: args.max_dialogues]

    if not dialogues:
        raise RuntimeError("No dialogues found for the requested split/filter.")

    if args.backend == "vllm":
        baseline = VLLMPromptingBaseline(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            cot=args.cot,
        )
    else:
        baseline = PromptingBaseline(
            model_name=model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            cot=args.cot,
        )

    predictions = []
    labels = []
    dialogue_ids = []
    turn_positions = []
    raw_records = []

    if args.backend == "vllm":
        prompt_items = []
        for dialogue in dialogues:
            max_prefixes = len(dialogue.turns)
            if args.max_prefixes_per_dialogue is not None:
                max_prefixes = min(max_prefixes, args.max_prefixes_per_dialogue)
            for end_turn_idx in range(max_prefixes):
                prompt_items.append((dialogue, end_turn_idx))

        prompts = [
            baseline.build_prompt(
                dialogue,
                end_turn_idx=end_turn_idx,
                max_visible_turns=args.max_visible_turns,
            )
            for dialogue, end_turn_idx in prompt_items
        ]
        results = baseline.predict_prompts(prompts)
        for (dialogue, end_turn_idx), result in zip(prompt_items, results):
            label = 0.0 if dialogue.success else 1.0
            predictions.append(result["prob_failure"])
            labels.append(label)
            dialogue_ids.append(dialogue.dialogue_id)
            turn_positions.append(end_turn_idx)
            raw_records.append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "dataset": dialogue.dataset,
                    "turn_position": end_turn_idx,
                    "label": label,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "prob_failure": result["prob_failure"],
                    "rationale": result["rationale"],
                    "raw_response": result["raw_response"],
                }
            )
    else:
        progress = tqdm(dialogues, desc="Prompting baseline", unit="dialogue")
        for dialogue in progress:
            max_prefixes = len(dialogue.turns)
            if args.max_prefixes_per_dialogue is not None:
                max_prefixes = min(max_prefixes, args.max_prefixes_per_dialogue)

            for end_turn_idx in range(max_prefixes):
                result = baseline.predict_prefix(
                    dialogue,
                    end_turn_idx=end_turn_idx,
                    max_visible_turns=args.max_visible_turns,
                )
                label = 0.0 if dialogue.success else 1.0

                predictions.append(result["prob_failure"])
                labels.append(label)
                dialogue_ids.append(dialogue.dialogue_id)
                turn_positions.append(end_turn_idx)
                raw_records.append(
                    {
                        "dialogue_id": dialogue.dialogue_id,
                        "dataset": dialogue.dataset,
                        "turn_position": end_turn_idx,
                        "label": label,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "prob_failure": result["prob_failure"],
                        "rationale": result["rationale"],
                        "raw_response": result["raw_response"],
                    }
                )

    predictions = np.array(predictions, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    turn_positions = np.array(turn_positions, dtype=np.int64)

    metrics, dialogue_predictions, dialogue_labels = evaluate_predictions(
        predictions,
        labels,
        dialogue_ids,
        turn_positions,
    )

    result = {
        "config": {
            "split": args.split,
            "dataset_filter": args.dataset_filter,
            "requested_model_name": requested_model_name,
            "model_name": model_name,
            "backend": args.backend,
            "cot": args.cot,
            "max_visible_turns": args.max_visible_turns,
            "max_dialogues": args.max_dialogues,
            "max_prefixes_per_dialogue": args.max_prefixes_per_dialogue,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
        "metrics": metrics,
        "num_dialogues": len(dialogues),
        "num_prefixes": len(raw_records),
        "raw_predictions": raw_records,
    }

    output_json = args.output_json
    if output_json is None:
        model_slug = slugify_model_name(model_name)
        output_json = os.path.join(
            cfg.output.get("checkpoint_dir", "outputs"),
            f"prompting_baseline_{args.dataset_filter}_{args.split}_{model_slug}.json",
        )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(
        {
            "output_json": output_json,
            "num_dialogues": len(dialogues),
            "num_prefixes": len(raw_records),
            "auc_roc": metrics["auc_roc"],
            "f1_at_0.5": metrics["f1_at_0.5"],
            "early_detection": metrics["early_detection"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
