
import argparse
import ast
import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


PRIMARY_LABELS = [
    "Information Incompleteness",
    "Contradiction / Misalignment",
    "State Instability",
    "Task Drift / Domain Confusion",
]

SYSTEM_PROMPT = (
    "You are an expert dialogue systems researcher doing qualitative analysis of "
    "task-oriented dialogue failures. You are assisting a researcher with "
    "interpretive annotation. Always respond with valid JSON only."
)

USER_PROMPT_TEMPLATE = """You are performing LLM-assisted qualitative annotation for failed task-oriented dialogues.

Important framing:
- These annotations are interpretive research notes.
- They are NOT ground-truth labels.
- Choose the most plausible PRIMARY failure pattern from the available evidence.

Primary label definitions:
1. Information Incompleteness
   Use when the task fails because required information is never stably assembled or the dialogue ends without the needed constraints.
2. Contradiction / Misalignment
   Use when the system and user are misaligned on facts or constraints, leading to correction, contradiction, or unresolved repair.
3. State Instability
   Use when slot values or dialogue state flip repeatedly and never stabilize.
4. Task Drift / Domain Confusion
   Use when task/domain switching derails progress or causes the system to respond to the wrong goal.

Auxiliary flags:
- mixed_signal=yes if a second failure pattern is clearly present.
- insufficient_evidence=yes if the visible context is too limited for a high-confidence call.

Dialogue metadata:
- dataset: {dataset}
- dialogue_id: {dialogue_id}
- number_of_turns: {num_turns}

Heuristic summary for analyst context only:
- heuristic_primary_label: {heuristic_primary_label}
- heuristic_secondary_label: {heuristic_secondary_label}
- heuristic_mixed_signal: {heuristic_mixed_signal}
- heuristic_insufficient_evidence: {heuristic_insufficient_evidence}
- heuristic_rationale: {heuristic_rationale}

Conversation context:
{dialogue_block}

Return valid JSON only:
{{
  "primary_label": "<one of the 4 primary labels>",
  "secondary_label": "<one of the 4 primary labels or empty string>",
  "mixed_signal": "<yes or no>",
  "insufficient_evidence": "<yes or no>",
  "confidence": <integer 1-5>,
  "evidence_span": "<short quote-like paraphrase of the key evidence, <= 20 words>",
  "annotation_note": "<2-4 concise sentences explaining the failure pattern>",
  "paper_safe_note": "<1-2 sentences phrased for a paper's qualitative analysis section>"
}}"""


def normalize_yes_no(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "yes" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "no"


def parse_snippet(snippet_str):
    if not isinstance(snippet_str, str) or not snippet_str.strip():
        return None
    try:
        return json.loads(snippet_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(snippet_str)
        except Exception:
            return None


def format_turn_snippets(snippets):
    lines = []
    for i, snippet in enumerate(snippets, start=1):
        if not snippet:
            continue
        lines.append(
            f"Snippet {i} (turn {snippet.get('turn_idx', 'N/A')}, domain={snippet.get('domain', 'unknown')}): "
            f"[User] {snippet.get('user', '')} "
            f"[System] {snippet.get('system', '')}"
        )
    return "\n".join(lines) if lines else "No snippets available."


def build_prompt(row):
    snippets = row.get("turn_snippets", [])
    if not isinstance(snippets, list):
        snippets = []
    return USER_PROMPT_TEMPLATE.format(
        dataset=row.get("dataset", "unknown"),
        dialogue_id=row.get("dialogue_id", "unknown"),
        num_turns=row.get("num_turns", "unknown"),
        heuristic_primary_label=row.get("primary_label", ""),
        heuristic_secondary_label=row.get("secondary_label", ""),
        heuristic_mixed_signal=normalize_yes_no(row.get("mixed_signal", "no")),
        heuristic_insufficient_evidence=normalize_yes_no(row.get("insufficient_evidence", "no")),
        heuristic_rationale=row.get("rationale", ""),
        dialogue_block=format_turn_snippets(snippets[:5]),
    )


def validate_response(parsed):
    primary_label = parsed.get("primary_label", "").strip()
    secondary_label = parsed.get("secondary_label", "").strip()
    if primary_label not in PRIMARY_LABELS:
        primary_label = "Information Incompleteness"
    if secondary_label and secondary_label not in PRIMARY_LABELS:
        secondary_label = ""
    if secondary_label == primary_label:
        secondary_label = ""
    try:
        confidence = int(parsed.get("confidence", 3))
    except Exception:
        confidence = 3
    confidence = max(1, min(5, confidence))
    return {
        "primary_label": primary_label,
        "secondary_label": secondary_label,
        "mixed_signal": normalize_yes_no(parsed.get("mixed_signal", "no")),
        "insufficient_evidence": normalize_yes_no(parsed.get("insufficient_evidence", "no")),
        "confidence": confidence,
        "evidence_span": str(parsed.get("evidence_span", "")).strip(),
        "annotation_note": str(parsed.get("annotation_note", "")).strip(),
        "paper_safe_note": str(parsed.get("paper_safe_note", "")).strip(),
    }


def call_model(client, model, prompt, retry_limit=3, retry_delay=5):
    error_msg = ""
    for attempt in range(1, retry_limit + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            validated = validate_response(parsed)
            validated["raw_response"] = raw
            validated["error"] = ""
            return validated
        except Exception as exc:
            error_msg = str(exc)
            logging.warning("Attempt %d/%d failed for %s: %s", attempt, retry_limit, model, exc)
            if attempt < retry_limit:
                time.sleep(retry_delay)
    return {
        "primary_label": "Information Incompleteness",
        "secondary_label": "",
        "mixed_signal": "no",
        "insufficient_evidence": "yes",
        "confidence": 1,
        "evidence_span": "",
        "annotation_note": "",
        "paper_safe_note": "",
        "raw_response": "",
        "error": error_msg,
    }


def sample_records(records_df, sample_size, seed, balanced=False):
    rng = random.Random(seed)
    if not balanced:
        idxs = list(records_df.index)
        rng.shuffle(idxs)
        return records_df.loc[idxs[:sample_size]].reset_index(drop=True)

    per_label = max(1, sample_size // len(PRIMARY_LABELS))
    rows = []
    for label in PRIMARY_LABELS:
        subset = records_df[records_df["primary_label"] == label]
        idxs = list(subset.index)
        rng.shuffle(idxs)
        rows.append(subset.loc[idxs[:per_label]])
    sampled = pd.concat(rows, axis=0).reset_index(drop=True)
    if len(sampled) < sample_size:
        remaining = records_df[~records_df["dialogue_id"].isin(sampled["dialogue_id"])]
        idxs = list(remaining.index)
        rng.shuffle(idxs)
        sampled = pd.concat([sampled, remaining.loc[idxs[: sample_size - len(sampled)]]], axis=0).reset_index(drop=True)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="LLM-assisted qualitative annotation for dialogue failure taxonomy")
    parser.add_argument("--input_json", type=str, default="plots/table2_failure_taxonomy_details.json")
    parser.add_argument("--output_csv", type=str, default="outputs/llm_assisted_taxonomy_annotations.csv")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--sample_size", type=int, default=60)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--request_pause", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    with open(args.input_json, "r") as f:
        records = json.load(f)
    records_df = pd.DataFrame(records)
    records_df = records_df[records_df["dataset"] == "mwoz"].copy()
    records_df = sample_records(records_df, args.sample_size, args.seed, balanced=args.balanced)

    logging.info("Loaded %d sampled dialogues for LLM-assisted annotation.", len(records_df))
    client = OpenAI(api_key=api_key)

    results = []
    pbar = tqdm(records_df.iterrows(), total=len(records_df), desc="Annotating", unit="dialogue")
    for idx, row in pbar:
        dialogue_id = row.get("dialogue_id", f"row_{idx}")
        pbar.set_postfix({"id": dialogue_id, "heuristic": row.get("primary_label", "")[:18]})
        prompt = build_prompt(row)
        result = call_model(client, args.model, prompt)
        time.sleep(args.request_pause)
        results.append({
            "dialogue_id": dialogue_id,
            "dataset": row.get("dataset", ""),
            "heuristic_primary_label": row.get("primary_label", ""),
            "heuristic_secondary_label": row.get("secondary_label", ""),
            "heuristic_mixed_signal": normalize_yes_no(row.get("mixed_signal", "no")),
            "heuristic_insufficient_evidence": normalize_yes_no(row.get("insufficient_evidence", "no")),
            "heuristic_rationale": row.get("rationale", ""),
            "llm_model": args.model,
            "llm_primary_label": result["primary_label"],
            "llm_secondary_label": result["secondary_label"],
            "llm_mixed_signal": result["mixed_signal"],
            "llm_insufficient_evidence": result["insufficient_evidence"],
            "llm_confidence": result["confidence"],
            "llm_evidence_span": result["evidence_span"],
            "llm_annotation_note": result["annotation_note"],
            "llm_paper_safe_note": result["paper_safe_note"],
            "llm_error": result["error"],
        })

    out_df = pd.DataFrame(results)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    logging.info("Saved LLM-assisted qualitative annotations to %s", args.output_csv)

    print("\n" + "=" * 60)
    print("LLM-ASSISTED QUALITATIVE ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"Total dialogues annotated : {len(out_df)}")
    print("\nHeuristic primary-label distribution:")
    print(out_df["heuristic_primary_label"].value_counts().to_string())
    print("\nLLM primary-label distribution:")
    print(out_df["llm_primary_label"].value_counts().to_string())
    print("\nHeuristic vs LLM primary-label confusion:")
    print(pd.crosstab(out_df["heuristic_primary_label"], out_df["llm_primary_label"]).to_string())
    print("\nPaper-safe framing:")
    print("These outputs are LLM-assisted qualitative annotations for analysis, not ground-truth taxonomy labels.")


if __name__ == "__main__":
    main()
