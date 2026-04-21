
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional


class NoIntervention:

    def predict(self, features=None, **kwargs):
        if features is None:
            return 0.0
        if isinstance(features, torch.Tensor):
            return torch.zeros(features.shape[0])
        return np.zeros(len(features))


class FixedScheduleConfirmation:

    def __init__(self, interval=3):
        self.interval = interval

    def predict(self, turn_positions, **kwargs):
        if isinstance(turn_positions, torch.Tensor):
            return ((turn_positions % self.interval) == (self.interval - 1)).float()
        return np.array([(t % self.interval == self.interval - 1)
                         for t in turn_positions], dtype=np.float32)


class SlotConfidenceTrigger:

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def predict(self, features, **kwargs):
        if isinstance(features, torch.Tensor):
            osc_scores = features[:, 0] if features.dim() > 1 else features[0]
            return (osc_scores > self.threshold).float()
        osc_scores = np.array([f[0] for f in features])
        return (osc_scores > self.threshold).astype(np.float32)


class LLMChainOfThought:

    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct",
                 device="cuda", max_new_tokens=256):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def _build_prompt(self, dialogue_context, belief_state):
        bs_text = ", ".join(f"{k}: {v}" for k, v in sorted(belief_state.items()))
        return (
            "You are analyzing a task-oriented dialogue system conversation. "
            "Based on the dialogue history and current belief state, determine if "
            "this dialogue is likely to fail (task not completed).\n\n"
            f"Dialogue so far:\n{dialogue_context}\n\n"
            f"Current belief state: {bs_text}\n\n"
            "Think step by step:\n"
            "1. Is the belief state stable or oscillating?\n"
            "2. Are key slots being filled at a reasonable rate?\n"
            "3. Are there contradictions between user and system?\n\n"
            "Based on your analysis, will this dialogue SUCCEED or FAIL?\n"
            "Answer with just SUCCESS or FAIL on the last line."
        )

    def predict_single(self, dialogue_turns, belief_state):
        self._load_model()

        context = ""
        for t in dialogue_turns:
            context += f"User: {t.user_utterance}\n"
            if t.system_utterance:
                context += f"System: {t.system_utterance}\n"

        prompt = self._build_prompt(context, belief_state)
        inputs = self._tokenizer(prompt, return_tensors="pt",
                                 truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )

        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)

        lines = response.strip().split("\n")
        for line in reversed(lines):
            line_upper = line.strip().upper()
            if "FAIL" in line_upper:
                return 1.0
            if "SUCCESS" in line_upper:
                return 0.0

        return 0.5


class FeatureThresholdEnsemble:

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            "osc_score": 2.0,
            "coverage_rate": 0.4,
            "conflict_count": 2.0,
            "fill_velocity": -0.5,
            "domain_shifts": 3.0,
        }

    def predict(self, features, **kwargs):
        if isinstance(features, torch.Tensor):
            features = features.numpy() if features.device.type == "cpu" else features.cpu().numpy()

        results = []
        for feat in features:
            osc, cov, conf, vel, shifts = feat
            fail = (
                osc > self.thresholds["osc_score"] or
                cov < self.thresholds["coverage_rate"] or
                conf > self.thresholds["conflict_count"] or
                vel < self.thresholds["fill_velocity"] or
                shifts > self.thresholds["domain_shifts"]
            )
            results.append(1.0 if fail else 0.0)

        return np.array(results, dtype=np.float32)
