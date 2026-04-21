
from typing import Dict, List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def features_to_str(features) -> str:
    if hasattr(features, "__len__") and len(features) >= 5:
        osc, cov, conf, vel, shifts = (
            float(features[0]), float(features[1]), float(features[2]),
            float(features[3]), float(features[4]),
        )
    else:
        osc = cov = conf = vel = shifts = 0.0
    return (
        f"osc={osc:.1f} cov={cov:.2f} conflict={conf:.0f} "
        f"velocity={vel:.2f} domains={shifts:.0f}"
    )


def belief_state_to_str(belief_state: Dict[str, str], max_slots: int = 20) -> str:
    if not belief_state:
        return "empty"
    parts = []
    for slot, value in sorted(belief_state.items())[:max_slots]:
        slot_clean = slot.replace("-", " ").replace("_", " ")
        parts.append(f"{slot_clean} {value}")
    return ", ".join(parts)


def format_context_turns(turns: List[tuple], last_n: int = 3) -> str:
    if not turns:
        return "no context"
    lines = []
    for u, s in turns[-last_n:]:
        if u:
            lines.append(f"user: {u[:100]}")
        if s:
            lines.append(f"sys: {s[:100]}")
    return " | ".join(lines)


def build_t5_input(
    features,
    belief_state: Dict[str, str],
    context_turns: Optional[List[tuple]] = None,
    max_input_length: int = 256,
) -> str:
    feat_str = features_to_str(features)
    belief_str = belief_state_to_str(belief_state)
    ctx_str = format_context_turns(context_turns or [], last_n=3)
    raw = f"features: {feat_str} | belief: {belief_str} | context: {ctx_str}"
    if len(raw) > max_input_length:
        raw = raw[: max_input_length - 3] + "..."
    return raw


class T5RecoveryGenerator:

    def __init__(
        self,
        model_name_or_path: str = "t5-small",
        device: Optional[str] = None,
        max_input_length: int = 256,
        max_output_length: int = 64,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self._model = None
        self._tokenizer = None

    def load(self, checkpoint_path: Optional[str] = None):
        path = checkpoint_path or self.model_name_or_path
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name_or_path if "t5" in path else path
        )
        self._model = T5ForConditionalGeneration.from_pretrained(path)
        self._model.to(self.device)
        self._model.eval()

    def generate(
        self,
        features,
        belief_state: Dict[str, str],
        context_turns: Optional[List[tuple]] = None,
        num_beams: int = 4,
    ) -> str:
        if self._model is None:
            self.load()

        input_str = build_t5_input(
            features, belief_state, context_turns, self.max_input_length
        )
        inputs = self._tokenizer(
            input_str,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def extract_confirmation_utterances_from_mwoz(dialogues) -> List[tuple]:
    pairs = []
    for dial in dialogues:
        if not hasattr(dial, "turns"):
            continue
        for t_idx, turn in enumerate(dial.turns):
            sys_utt = getattr(turn, "system_utterance", "") or ""
            if not sys_utt or len(sys_utt) < 10:
                continue
            if not any(
                phrase in sys_utt.lower()
                for phrase in ("is that", "correct", "right?", "confirm", "just to", "let me")
            ):
                continue
            belief = getattr(turn, "belief_state", {}) or {}
            belief_str = belief_state_to_str(belief)
            context = []
            for i in range(max(0, t_idx - 2), t_idx):
                if i < len(dial.turns):
                    ut = dial.turns[i]
                    context.append((
                        getattr(ut, "user_utterance", "") or "",
                        getattr(ut, "system_utterance", "") or "",
                    ))
            features = [0, 0.5, 0, 0.5, 1]
            input_str = build_t5_input(features, belief, context, max_input_length=256)

            target_utt = sys_utt[:120]

            if input_str and target_utt:
                pairs.append((input_str, target_utt))
    return pairs
