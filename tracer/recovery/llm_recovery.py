
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _format_features(features) -> str:
    if hasattr(features, "__len__") and len(features) >= 5:
        osc, cov, conf, vel, shifts = (
            float(features[0]), float(features[1]), float(features[2]),
            float(features[3]), float(features[4]),
        )
        return (
            f"Oscillation score: {osc:.1f}, Coverage: {cov:.2f}, "
            f"Conflict count: {conf:.0f}, Fill velocity: {vel:.2f}, "
            f"Domain shifts: {shifts:.0f}"
        )
    return "Unknown"


def _belief_to_str(belief_state: Dict[str, str]) -> str:
    if not belief_state:
        return "None"
    return ", ".join(f"{k}: {v}" for k, v in sorted(belief_state.items()))


def _build_system_prompt() -> str:
    return (
        "You are a task-oriented dialogue system assistant. When the system detects "
        "that the conversation may be heading toward failure (e.g., belief state "
        "oscillation, missing information, or user-system conflict), you must generate "
        "a single short recovery utterance. The utterance should clarify or confirm "
        "key information with the user in a natural way. Respond with ONLY the "
        "recovery utterance, no explanation."
    )


def _build_user_prompt(
    features,
    belief_state: Dict[str, str],
    domain: str,
    last_user_utt: str = "",
    last_system_utt: str = "",
) -> str:
    feat_str = _format_features(features)
    belief_str = _belief_to_str(belief_state)
    return (
        "The failure predictor has triggered. Generate a recovery utterance.\n\n"
        f"Trajectory signals: {feat_str}\n"
        f"Current belief state ({domain}): {belief_str}\n"
        f"Last user: {last_user_utt[:200] if last_user_utt else 'N/A'}\n"
        f"Last system: {last_system_utt[:200] if last_system_utt else 'N/A'}\n\n"
        "Recovery utterance:"
    )


class LLMRecoveryGenerator:

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else self.device,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        features,
        belief_state: Dict[str, str],
        domain: str = "dialogue",
        last_user_utt: str = "",
        last_system_utt: str = "",
    ) -> str:
        self._load_model()

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            features, belief_state, domain, last_user_utt, last_system_utt
        )

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\n"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        gen_start = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            outputs[0][gen_start:],
            skip_special_tokens=True,
        ).strip()

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Recovery") and len(line) > 5:
                return line[:300]
        return response[:300] if response else "Let me confirm the details with you."
