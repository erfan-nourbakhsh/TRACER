
import argparse
import json
import os
import re
import sys
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.domain_slots import MWOZ_ALL_SLOTS


BASE_MWOZ_SCHEMA = {
    domain: {slot.split("-", 1)[1] for slot in slots}
    for domain, slots in MWOZ_ALL_SLOTS.items()
}

SLOT_ALIASES = {
    "centre": "area",
    "center": "area",
    "cuisine": "food",
    "price": "pricerange",
    "price range": "pricerange",
    "source": "departure",
    "from": "departure",
    "pickup": "departure",
    "pick up": "departure",
    "dropoff": "destination",
    "drop off": "destination",
    "to": "destination",
    "depart": "leaveat",
    "depart time": "leaveat",
    "departure time": "leaveat",
    "leave": "leaveat",
    "leave time": "leaveat",
    "leaving time": "leaveat",
    "arrival": "arriveby",
    "arrival time": "arriveby",
    "arrive": "arriveby",
    "arrive by": "arriveby",
    "people": "book people",
    "party size": "book people",
    "booking people": "book people",
    "book people": "book people",
    "day": "book day",
    "booking day": "book day",
    "book day": "book day",
    "time": "book time",
    "booking time": "book time",
    "book time": "book time",
}


SYSTEM_PROMPT_CONSTRAINED = """You are a dialogue state tracker for task-oriented dialogue.
Predict the cumulative belief state after the latest USER utterance.

Rules:
- Use only the dialogue text shown in the prompt.
- Output only valid JSON. Do not include markdown, explanations, or comments.
- The JSON must have exactly these top-level keys: "domain" and "belief_state".
- "domain" must be the active domain/service after the latest user utterance.
- "belief_state" must be a JSON object mapping slot keys to string values.
- Slot keys must use the format "domain-slot", for example "restaurant-area".
- Use lowercase domain and slot names.
- Include only information explicitly stated or clearly confirmed by the user.
- Start from the previous belief_state given in the prompt.
- Keep all earlier slot values across all domains unless the user corrects or replaces them.
- If the user corrects a value, use the latest corrected value.
- Omit unknown, not mentioned, none, empty, or unsupported slots.
- Values should be short normalized strings, not full sentences.
"""

SYSTEM_PROMPT_OPEN = """You are a dialogue state tracker for task-oriented dialogue.
Predict the cumulative belief state after the latest USER utterance.

Rules:
- Use only the dialogue text shown in the prompt.
- Output only valid JSON. Do not include markdown, explanations, or comments.
- The JSON must have exactly these top-level keys: "domain" and "belief_state".
- "domain" must be the active domain/service after the latest user utterance.
- "belief_state" must be a JSON object mapping slot keys to string values.
- Slot keys must use the format "domain-slot", for example "restaurant-area".
- Infer the relevant domains and slots from the conversation.
- Include only information explicitly stated or clearly confirmed by the user.
- Start from the previous belief_state given in the prompt.
- Keep all earlier slot values across all domains unless the user corrects or replaces them.
- If the user corrects a value, use the latest corrected value.
- Omit unknown, not mentioned, none, empty, or unsupported slots.
- Values should be short normalized strings, not full sentences.
"""

SYSTEM_PROMPT_CONSTRAINED_CAREFUL = """You are a precise dialogue state tracker for MultiWOZ-style task-oriented dialogue.
Predict the cumulative belief state after the latest USER utterance.

Rules:
- Use only the dialogue text shown in the prompt.
- Output only valid JSON. Do not include markdown, explanations, or comments.
- The JSON must have exactly these top-level keys: "domain" and "belief_state".
- "domain" must be the active domain/service after the latest USER utterance.
- "belief_state" must be a JSON object mapping slot keys to string values.
- Slot keys must use the format "domain-slot", for example "restaurant-area".
- Use lowercase domain and slot names.
- Start from the previous belief_state given in the prompt.
- Return the full cumulative belief_state, not only the new changes.
- Keep earlier slot values across domains unless the user explicitly corrects, replaces, or cancels them.
- If the user corrects a domain, move the value to the corrected domain and remove the wrong-domain version.
- Track user constraints such as area, food, price range, type, name, day, time, people, departure, destination, leave time, and arrival time.
- Track booking information when the user gives booking day, time, stay length, or number of people.
- If the system offers a specific entity and the user accepts it or asks follow-up details about it, include that entity as the corresponding "domain-name".
- If the user asks for address, postcode, phone, entrance fee, reference number, rating, or car type, treat these as requested information, not belief-state slots.
- If the user says "same area", copy the relevant earlier area value to the new domain.
- Do not hallucinate values from database results unless the dialogue text makes the user accept or refer to that entity.
- Omit unknown, not mentioned, none, empty, or unsupported slots.
- Values should be short normalized strings, not full sentences.
"""

SYSTEM_PROMPT_OPEN_CAREFUL = """You are a precise dialogue state tracker for task-oriented dialogue.
Predict the cumulative belief state after the latest USER utterance.

Rules:
- Use only the dialogue text shown in the prompt.
- Output only valid JSON. Do not include markdown, explanations, or comments.
- The JSON must have exactly these top-level keys: "domain" and "belief_state".
- "domain" must be the active domain/service after the latest USER utterance.
- "belief_state" must be a JSON object mapping slot keys to string values.
- Slot keys must use the format "domain-slot", for example "restaurant-area".
- Infer the relevant domains and slots from the conversation.
- Start from the previous belief_state given in the prompt.
- Return the full cumulative belief_state, not only the new changes.
- Keep earlier slot values across domains unless the user explicitly corrects, replaces, or cancels them.
- If the user corrects a domain, move the value to the corrected domain and remove the wrong-domain version.
- Track user constraints such as area, food, price range, type, name, day, time, people, departure, destination, leave time, and arrival time.
- Track booking information when the user gives booking day, time, stay length, or number of people.
- If the system offers a specific entity and the user accepts it or asks follow-up details about it, include that entity as the corresponding "domain-name".
- If the user asks for address, postcode, phone, entrance fee, reference number, rating, or car type, treat these as requested information, not belief-state slots.
- If the user says "same area", copy the relevant earlier area value to the new domain.
- Do not hallucinate values from database results unless the dialogue text makes the user accept or refer to that entity.
- Omit unknown, not mentioned, none, empty, or unsupported slots.
- Values should be short normalized strings, not full sentences.
"""

SYSTEM_PROMPT_CONSTRAINED_EXAMPLES = """You are a precise dialogue state tracker for MultiWOZ-style task-oriented dialogue.
Predict the cumulative belief state after the latest USER utterance.

Output requirements:
- Output only valid JSON, with exactly "domain" and "belief_state".
- "belief_state" maps "domain-slot" keys to short lowercase string values.
- Return the full cumulative belief_state after the latest USER utterance.

State tracking policy:
- Start from the previous belief_state in the prompt.
- Keep earlier slot values across domains unless the user corrects, replaces, or cancels them.
- If a user says "not a restaurant, it is an attraction", remove the wrong-domain name and keep the corrected domain name.
- Add a slot only when it is a search/booking constraint or an accepted/referred entity name.
- Do not add requested attributes such as address, postcode, phone, entrance fee, reference number, rating, or car type.
- If a user asks for an address for "saffron brasserie", set "restaurant-name": "saffron brasserie".
- If a user asks for attractions in the same area as a restaurant, copy the restaurant area to "attraction-area".
- Map "leave after 17:15" to "taxi-leaveat" or "train-leaveat".
- Map "arrive by 10:30" to "train-arriveby" or "taxi-arriveby".
- Map "cheap/moderate/expensive" to "pricerange".
- Map "center" to "centre".
"""

SYSTEM_PROMPT_CONSTRAINED_TURN = """You are a precise dialogue state tracker for MultiWOZ-style task-oriented dialogue.
Predict the cumulative belief state after the complete latest turn, using both the latest USER utterance and the following SYSTEM response shown for that turn.

Output requirements:
- Output only valid JSON, with exactly "domain" and "belief_state".
- "domain" is the active domain/service of the latest turn.
- "belief_state" maps "domain-slot" keys to short lowercase string values.
- Return the full cumulative belief_state after the latest complete turn.

State tracking policy:
- Start from the previous belief_state in the prompt.
- Multi-domain belief states are cumulative: keep slots from earlier domains when the user starts a new domain.
- Remove a previous slot only when the user explicitly corrects, replaces, or cancels it.
- If the user corrects a domain, move the value to the corrected domain and remove the wrong-domain version.
- Add user constraints: area, food, pricerange, type, name, booking day/time/people/stay, train/taxi departure, destination, leaveat, and arriveby.
- Use the SYSTEM response to add an entity name when the system recommends, identifies, books, or confirms a specific entity.
- If the user asks follow-up details about an entity mentioned by the system, keep that entity as "domain-name".
- If the user asks for address, postcode, phone, entrance fee, reference number, rating, or car type, these are requestable attributes; do not add them as belief-state slots.
- If the user says "same area", copy the earlier area constraint into the new domain.
- Do not treat "food type", "postcode", "address", or "phone number" requests as slot values.
- Map "center" to "centre".
"""

SYSTEM_PROMPT_OPEN_TURN = """You are a precise dialogue state tracker for task-oriented dialogue.
Predict the cumulative belief state after the complete latest turn, using both the latest USER utterance and the following SYSTEM response shown for that turn.

Output requirements:
- Output only valid JSON, with exactly "domain" and "belief_state".
- "domain" is the active domain/service of the latest turn.
- "belief_state" maps "domain-slot" keys to short lowercase string values.
- Return the full cumulative belief_state after the latest complete turn.

State tracking policy:
- Start from the previous belief_state in the prompt.
- Multi-domain belief states are cumulative: keep slots from earlier domains when the user starts a new domain.
- Remove a previous slot only when the user explicitly corrects, replaces, or cancels it.
- If the user corrects a domain, move the value to the corrected domain and remove the wrong-domain version.
- Infer the relevant domains and slots from the conversation.
- Add user constraints: area, food, pricerange, type, name, booking day/time/people/stay, train/taxi departure, destination, leaveat, and arriveby.
- Use the SYSTEM response to add an entity name when the system recommends, identifies, books, or confirms a specific entity.
- If the user asks follow-up details about an entity mentioned by the system, keep that entity as "domain-name".
- If the user asks for address, postcode, phone, entrance fee, reference number, rating, or car type, these are requestable attributes; do not add them as belief-state slots.
- If the user says "same area", copy the earlier area constraint into the new domain.
- Do not treat "food type", "postcode", "address", or "phone number" requests as slot values.
- Map "center" to "centre".
"""


SYSTEM_PROMPTS = {
    "constrained": SYSTEM_PROMPT_CONSTRAINED,
    "open": SYSTEM_PROMPT_OPEN,
    "constrained_careful": SYSTEM_PROMPT_CONSTRAINED_CAREFUL,
    "open_careful": SYSTEM_PROMPT_OPEN_CAREFUL,
    "constrained_examples": SYSTEM_PROMPT_CONSTRAINED_EXAMPLES,
    "constrained_turn": SYSTEM_PROMPT_CONSTRAINED_TURN,
    "open_turn": SYSTEM_PROMPT_OPEN_TURN,
    "constrained_turn_repair": SYSTEM_PROMPT_CONSTRAINED_TURN,
    "open_turn_repair": SYSTEM_PROMPT_OPEN_TURN,
    "constrained_turn_repair_strict": SYSTEM_PROMPT_CONSTRAINED_TURN,
    "open_turn_repair_strict": SYSTEM_PROMPT_OPEN_TURN,
}


def load_config(config_path: str):
    import omegaconf

    return omegaconf.OmegaConf.load(config_path)


def load_unified(cache_dir: str, split: str) -> List[dict]:
    path = os.path.join(cache_dir, f"unified_{split}.json")
    with open(path, "r") as f:
        return json.load(f)


def save_unified(cache_dir: str, split: str, suffix: str, dialogues: List[dict]):
    path = os.path.join(cache_dir, f"unified_{split}_{suffix}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(dialogues, f, indent=0)
    os.replace(tmp_path, path)
    return path


def build_mwoz_schema(dialogues: List[dict]) -> Dict[str, set]:
    schema = {domain: set(slots) for domain, slots in BASE_MWOZ_SCHEMA.items()}
    for dialogue in dialogues:
        if dialogue.get("dataset") != "mwoz":
            continue
        for turn in dialogue.get("turns", []):
            for key in turn.get("belief_state", {}):
                if "-" not in key:
                    continue
                domain, slot = key.split("-", 1)
                schema.setdefault(domain, set()).add(slot)
    return schema


def schema_text_for_dialogue(dialogue: dict, mwoz_schema: Dict[str, set]) -> str:
    if dialogue.get("dataset") == "mwoz":
        lines = []
        for domain, slots in sorted(mwoz_schema.items()):
            if slots:
                lines.append(f"- {domain}: {', '.join(sorted(slots))}")
            else:
                lines.append(f"- {domain}: no supported slots")
        return "\n".join(lines)

    services = dialogue.get("domains") or []
    if services:
        return "\n".join(f"- {service}: use service-slot keys" for service in services)

    return "- unknown: use domain-slot keys only when supported by the dialogue"


def format_dialogue_prefix(turns: List[dict], turn_idx: int, max_context_turns: int) -> str:
    start = max(0, turn_idx - max_context_turns + 1)
    lines = []
    for local_idx, turn in enumerate(turns[start : turn_idx + 1], start=start):
        user = (turn.get("user_utterance") or "").strip()
        system = (turn.get("system_utterance") or "").strip()
        if user:
            lines.append(f"Turn {local_idx} USER: {user}")
        if system:
            lines.append(f"Turn {local_idx} SYSTEM: {system}")
    return "\n".join(lines)


def build_prompt(
    dialogue: dict,
    turn_idx: int,
    max_context_turns: int,
    prompt_style: str,
    mwoz_schema: Dict[str, set],
    previous_state: Dict[str, str],
) -> str:
    schema = schema_text_for_dialogue(dialogue, mwoz_schema)
    prefix = format_dialogue_prefix(dialogue["turns"], turn_idx, max_context_turns)
    previous_json = json.dumps(previous_state, sort_keys=True)
    if prompt_style.startswith("open"):
        if "_turn" in prompt_style:
            return f"""Dialogue prefix:
{prefix}

Previous belief_state before the latest complete turn:
{previous_json}

Return the belief state after the latest complete turn, including information from the latest USER utterance and the following SYSTEM response.
Required JSON format:
{{"domain": "restaurant", "belief_state": {{"restaurant-area": "north"}}}}"""

        return f"""Dialogue prefix:
{prefix}

Previous belief_state before the latest USER utterance:
{previous_json}

Return the belief state after the latest USER utterance.
Required JSON format:
{{"domain": "restaurant", "belief_state": {{"restaurant-area": "north"}}}}"""

    if "_turn" in prompt_style:
        return f"""Allowed domains and slots:
{schema}

Dialogue prefix:
{prefix}

Previous belief_state before the latest complete turn:
{previous_json}

Return the belief state after the latest complete turn, including information from the latest USER utterance and the following SYSTEM response.
Required JSON format:
{{"domain": "restaurant", "belief_state": {{"restaurant-area": "north"}}}}"""

    return f"""Allowed domains and slots:
{schema}

Dialogue prefix:
{prefix}

Previous belief_state before the latest USER utterance:
{previous_json}

Return the belief state after the latest USER utterance.
Required JSON format:
{{"domain": "restaurant", "belief_state": {{"restaurant-area": "north"}}}}"""


def apply_chat_template(tokenizer, messages: List[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("model output JSON is not an object")
    return parsed


def normalize_key(key: str) -> str:
    key = str(key).strip().lower()
    key = key.replace("_", " ")
    key = re.sub(r"\s*-\s*", "-", key)
    key = re.sub(r"\s+", " ", key)
    return key


def normalize_value(value) -> str:
    normalized = str(value).strip().lower()
    if normalized == "center":
        return "centre"
    return normalized


def _extract_time(text: str) -> str:
    match = re.search(r"\b([0-2]?\d)[:.]([0-5]\d)\b", text)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = match.group(2)
    return f"{hour:02d}:{minute}"


def _clean_entity_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+(?:please|thanks?|thank you)$", "", value)
    return value.strip(" .,!?")


def _repair_prediction(
    belief_state: Dict[str, str],
    previous_state: Dict[str, str],
    turn: dict,
    active_domain: str,
    user_prefix: str,
    strict: bool = False,
) -> Dict[str, str]:
    user = (turn.get("user_utterance") or "").lower()
    system = (turn.get("system_utterance") or "").lower()
    combined = f"{user} {system}"

    repaired = dict(previous_state)

    if "not a restaurant" in user and "attraction" in user:
        repaired.pop("restaurant-name", None)
        name_match = re.search(r"attraction\s*[.,]?\s*([a-z0-9 '&-]+)$", user)
        if name_match:
            repaired["attraction-name"] = _clean_entity_name(name_match.group(1))

    repaired.update(belief_state)

    if active_domain in {"restaurant", "hotel", "attraction"}:
        if "center" in user or "centre" in user or "town" in user:
            repaired[f"{active_domain}-area"] = "centre"
        for area in ("north", "south", "east", "west"):
            if re.search(rf"\b{area}\b", user):
                repaired[f"{active_domain}-area"] = area

    requestable_words = ("address", "postcode", "post code", "phone", "entrance fee", "reference number", "car type")
    if any(word in user for word in requestable_words):
        for key in list(repaired):
            if key.endswith(("-address", "-postcode", "-phone", "-entrance fee", "-reference", "-car type")):
                repaired.pop(key, None)

    if "food type" in user or "postcode" in user or "post code" in user:
        if repaired.get("restaurant-food") in {"multiple sports", "sports"}:
            repaired.pop("restaurant-food", None)

    if "multiple sports" in user and "attraction" in user:
        repaired["attraction-type"] = "multiple sports"
        if "same area" in user and repaired.get("restaurant-area"):
            repaired["attraction-area"] = repaired["restaurant-area"]

    if "architecture attraction" in user or "architecture attractions" in user:
        repaired["attraction-type"] = "architecture"
        if "centre" in user or "center" in user or "town" in user:
            repaired["attraction-area"] = "centre"

    time_value = _extract_time(user)
    if time_value:
        has_taxi = active_domain == "taxi" or any(k.startswith("taxi-") for k in repaired)
        has_train = active_domain == "train" or any(k.startswith("train-") for k in repaired)
        time_domain = "taxi" if has_taxi and not has_train else active_domain
        if time_domain not in {"taxi", "train"}:
            time_domain = "train" if has_train else ("taxi" if has_taxi else "")
        if time_domain:
            if "arrive" in user:
                repaired[f"{time_domain}-arriveby"] = time_value
            elif "leave" in user or "depart" in user:
                repaired[f"{time_domain}-leaveat"] = time_value

    for key in list(repaired):
        slot = key.split("-", 1)[1] if "-" in key else ""
        if slot in {"leaveat", "arriveby", "book time"}:
            if not re.fullmatch(r"[0-2]\d:[0-5]\d", repaired[key]):
                repaired.pop(key, None)

    restaurant_address_match = re.search(r"address for ([a-z0-9 '&-]+)", user)
    if restaurant_address_match:
        repaired["restaurant-name"] = _clean_entity_name(restaurant_address_match.group(1))

    fav_match = re.search(r"my favorite (?:is|it the|it is the)\s+([a-z0-9 '&-]+?)\s+at\b", system)
    if fav_match:
        repaired["restaurant-name"] = _clean_entity_name(fav_match.group(1))

    attraction_match = re.search(r"([a-z0-9 '&-]+?)\s+is an? (?:architectural|architecture|[a-z ]+)?\s*attraction\b", system)
    if attraction_match:
        repaired["attraction-name"] = _clean_entity_name(attraction_match.group(1))

    if ("address" in user or "postcode" in user or "post code" in user) and "attraction-name" in repaired:
        if "attraction-area" not in previous_state and "attraction" not in user:
            repaired.pop("attraction-area", None)
        if "attraction-type" not in previous_state and "attraction" not in user:
            repaired.pop("attraction-type", None)

    if strict:
        user_prefix = user_prefix.lower()
        for key in list(repaired):
            if not key.endswith("-area"):
                continue
            domain = key.split("-", 1)[0]
            value = repaired[key]
            supported = value in user_prefix
            if value == "centre":
                supported = supported or "center" in user_prefix or "town" in user_prefix
            if domain == "attraction" and "same area" in user_prefix and repaired.get("restaurant-area") == value:
                supported = True
            elif domain == "attraction" and "attraction" not in user_prefix:
                supported = False
            if not supported:
                repaired.pop(key, None)

    return repaired


def canonicalize_mwoz_key(key: str, allowed_keys: set) -> str:
    if key in allowed_keys:
        return key
    if "-" not in key:
        return key
    domain, slot = key.split("-", 1)
    slot = SLOT_ALIASES.get(slot, slot)
    candidate = f"{domain}-{slot}"
    if candidate in allowed_keys:
        return candidate
    return key


def normalize_prediction(
    prediction: dict,
    dataset: str,
    prompt_style: str,
    mwoz_allowed_keys: set,
) -> Tuple[str, Dict[str, str]]:
    raw_domain = str(prediction.get("domain", "")).strip().lower()
    raw_state = prediction.get("belief_state", {})
    if not isinstance(raw_state, dict):
        raw_state = {}

    belief_state = {}
    for raw_key, raw_value in raw_state.items():
        key = normalize_key(raw_key)
        if dataset == "mwoz":
            key = canonicalize_mwoz_key(key, mwoz_allowed_keys)
        value = normalize_value(raw_value)
        if not key or not value:
            continue
        if value in {"none", "not mentioned", "unknown", "null", "n/a"}:
            continue
        if dataset == "mwoz" and key not in mwoz_allowed_keys:
            continue
        belief_state[key] = value

    if dataset == "mwoz" and raw_domain not in BASE_MWOZ_SCHEMA:
        raw_domain = ""

    if not raw_domain and belief_state:
        first_key = next(iter(belief_state))
        raw_domain = first_key.split("-", 1)[0] if "-" in first_key else ""

    return raw_domain, belief_state


def compute_delta(current: Dict[str, str], previous: Dict[str, str]) -> Dict[str, str]:
    return {
        slot: value
        for slot, value in current.items()
        if slot not in previous or previous[slot] != value
    }


def _build_chat_prompts(tokenizer, prompts: List[str], system_prompt: str) -> List[str]:
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat_prompts.append(apply_chat_template(tokenizer, messages))
    return chat_prompts


def generate_texts_batch(
    llm: LLM,
    tokenizer,
    prompts: List[str],
    system_prompt: str,
    max_new_tokens: int,
) -> List[str]:
    if not prompts:
        return []
    chat_prompts = _build_chat_prompts(tokenizer, prompts, system_prompt)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    outputs = llm.generate(chat_prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def iter_selected_dialogues(
    dialogues: Iterable[dict],
    datasets: set,
    start_index: int,
    max_dialogues: int,
):
    selected = [(idx, dial) for idx, dial in enumerate(dialogues)]
    if start_index:
        selected = selected[start_index:]
    if max_dialogues > 0:
        selected = selected[:max_dialogues]
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompted belief states with Qwen and save a parallel unified cache."
    )
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--datasets", nargs="+", default=["mwoz"])
    parser.add_argument("--suffix", default="qwen_generated")
    parser.add_argument("--max_context_turns", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_dialogues", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--prompt_style", choices=sorted(SYSTEM_PROMPTS), default="constrained",
                        help="Use the schema-constrained prompt or an open prompt without allowed slots.")
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (default: all available GPUs).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory vllm may use (default: 0.90).")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum sequence length (prompt + generation tokens).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    datasets = set(args.datasets or [])

    system_prompt = SYSTEM_PROMPTS[args.prompt_style]

    tp_size = args.tensor_parallel_size or max(1, torch.cuda.device_count())

    print(f"Loading {args.model} with vllm (tensor_parallel_size={tp_size}, "
          f"gpu_memory_utilization={args.gpu_memory_utilization}, "
          f"max_model_len={args.max_model_len}, dtype={args.torch_dtype}) ...")

    llm = LLM(
        model=args.model,
        dtype=args.torch_dtype,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()

    for split in args.splits:
        oracle_dialogues = load_unified(cache_dir, split)
        mwoz_schema = build_mwoz_schema(oracle_dialogues)
        mwoz_allowed_keys = {
            f"{domain}-{slot}"
            for domain, slots in mwoz_schema.items()
            for slot in slots
        }
        generated_dialogues = [
            deepcopy(dial)
            for dial in oracle_dialogues
            if not datasets or dial.get("dataset") in datasets
        ]
        selected = iter_selected_dialogues(
            generated_dialogues,
            datasets=set(),
            start_index=args.start_index,
            max_dialogues=args.max_dialogues,
        )
        if args.start_index or args.max_dialogues > 0:
            generated_dialogues = [dialogue for _, dialogue in selected]
            selected = list(enumerate(generated_dialogues))


        n_dialogues = len(selected)
        max_turns = max((len(dial["turns"]) for _, dial in selected), default=0)

        prev_states: Dict[int, Dict[str, str]] = {
            idx: {} for idx, _ in selected
        }
        prev_domains: Dict[int, str] = {
            idx: (dial["turns"][0].get("domain", "") if dial["turns"] else "")
            for idx, dial in selected
        }
        failures = []
        completed_dialogues = 0

        for turn_idx in tqdm(range(max_turns), desc=f"Generating {split} belief states (turn batches)"):
            active: List[Tuple[int, dict]] = [
                (idx, dial) for idx, dial in selected if turn_idx < len(dial["turns"])
            ]
            if not active:
                break

            user_prompts: List[str] = []
            for idx, dialogue in active:
                prompt = build_prompt(
                    dialogue,
                    turn_idx,
                    args.max_context_turns,
                    prompt_style=args.prompt_style,
                    mwoz_schema=mwoz_schema,
                    previous_state=prev_states[idx],
                )
                user_prompts.append(prompt)

            raw_outputs = generate_texts_batch(
                llm,
                tokenizer,
                user_prompts,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
            )

            for (idx, dialogue), raw_output in zip(active, raw_outputs):
                turn = dialogue["turns"][turn_idx]
                previous_state = prev_states[idx]
                previous_domain = prev_domains[idx]
                try:
                    parsed = extract_json_object(raw_output)
                    domain, belief_state = normalize_prediction(
                        parsed,
                        dataset=dialogue.get("dataset", ""),
                        prompt_style=args.prompt_style,
                        mwoz_allowed_keys=mwoz_allowed_keys,
                    )
                    if args.prompt_style.endswith("_repair"):
                        belief_state = _repair_prediction(
                            belief_state,
                            previous_state=previous_state,
                            turn=turn,
                            active_domain=domain,
                            user_prefix=" ".join(
                                (pt.get("user_utterance") or "")
                                for pt in dialogue["turns"][: turn_idx + 1]
                            ),
                            strict=False,
                        )
                    elif args.prompt_style.endswith("_repair_strict"):
                        belief_state = _repair_prediction(
                            belief_state,
                            previous_state=previous_state,
                            turn=turn,
                            active_domain=domain,
                            user_prefix=" ".join(
                                (pt.get("user_utterance") or "")
                                for pt in dialogue["turns"][: turn_idx + 1]
                            ),
                            strict=True,
                        )
                except Exception as exc:
                    failures.append({
                        "dialogue_id": dialogue.get("dialogue_id"),
                        "turn_idx": turn_idx,
                        "error": str(exc),
                    })
                    domain = previous_domain
                    belief_state = dict(previous_state)

                turn["belief_state"] = belief_state
                turn["turn_delta"] = compute_delta(belief_state, previous_state)
                turn["domain"] = domain or previous_domain
                prev_states[idx] = dict(belief_state)
                prev_domains[idx] = turn["domain"]

            completed_dialogues += sum(
                1 for idx, dial in active if turn_idx == len(dial["turns"]) - 1
            )

            if args.save_every > 0 and completed_dialogues % args.save_every == 0:
                save_unified(cache_dir, split, args.suffix, generated_dialogues)

        for _, dialogue in selected:
            dialogue.setdefault("metadata", {})
            dialogue["metadata"]["belief_state_source"] = args.model
            dialogue["metadata"]["belief_state_generation"] = "prompted"
            dialogue["metadata"]["belief_state_prompt_style"] = args.prompt_style

        output_path = save_unified(cache_dir, split, args.suffix, generated_dialogues)
        if failures:
            failure_path = os.path.join(cache_dir, f"belief_generation_failures_{split}_{args.suffix}.json")
            with open(failure_path, "w") as f:
                json.dump(failures, f, indent=2)
            print(f"Saved {len(failures)} generation failures to {failure_path}")

        print(f"Saved generated belief-state cache to {output_path}")


if __name__ == "__main__":
    main()
