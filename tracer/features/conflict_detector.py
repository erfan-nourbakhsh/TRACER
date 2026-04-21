
import numpy as np
from typing import Dict, List, Optional


class ConflictDetector:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 threshold=0.3, device=None):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=device)
        self.threshold = threshold
        self._cache = {}

    def _encode(self, texts):
        to_encode = [t for t in texts if t not in self._cache]
        if to_encode:
            embeddings = self.encoder.encode(to_encode, convert_to_numpy=True)
            for text, emb in zip(to_encode, embeddings):
                self._cache[text] = emb
        return np.array([self._cache[t] for t in texts])

    def detect_conflict(self, user_utterance, prev_belief, turn_delta):
        if not prev_belief or not turn_delta:
            return False

        changed_slots = {}
        for slot, new_value in turn_delta.items():
            if slot in prev_belief and prev_belief[slot] != new_value:
                changed_slots[slot] = prev_belief[slot]

        if not changed_slots:
            return False

        user_emb = self._encode([user_utterance])[0]

        for slot, old_value in changed_slots.items():
            slot_text = slot.replace("-", " ").replace("_", " ")
            old_text = f"The {slot_text} is {old_value}"
            old_emb = self._encode([old_text])[0]

            cosine_sim = np.dot(user_emb, old_emb) / (
                np.linalg.norm(user_emb) * np.linalg.norm(old_emb) + 1e-8
            )

            if cosine_sim < self.threshold:
                return True

        return False

    def count_conflicts(self, user_utterances, belief_history, turn_deltas):
        count = 0
        for i in range(len(user_utterances)):
            prev_bs = belief_history[i - 1] if i > 0 else {}
            if self.detect_conflict(user_utterances[i], prev_bs, turn_deltas[i]):
                count += 1
        return count

    def clear_cache(self):
        self._cache = {}
