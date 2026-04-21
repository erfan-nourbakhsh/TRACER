
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_JSON = OUTPUT_DIR / "evaluate_full_results.json"
ERROR_JSON = OUTPUT_DIR / "error_analysis.json"
TAXONOMY_SUMMARY = OUTPUT_DIR / "llm_assisted_taxonomy_summary.md"
TAXONOMY_CSV = OUTPUT_DIR / "llm_assisted_taxonomy_annotations.csv"


def load_results():
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_JSON}")
    return json.loads(RESULTS_JSON.read_text())


def export_table(rows, out_path):
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def summarize_cross_domain(results):
    rows = results.get("table5_cross_domain", [])
    if not rows:
        return "Cross-domain transfer summary unavailable."

    notes = []
    for row in rows:
        dataset = row.get("Dataset", "unknown")
        auc = row.get("AUC-ROC", "n/a")
        f1 = row.get("F1", "n/a")
        notes.append(f"- {dataset}: AUC-ROC `{auc}`, F1 `{f1}`")
    return "\n".join(notes)


def build_notes(results):
    evaluation_setup = results.get("evaluation_setup", {})
    checkpoints = evaluation_setup.get("checkpoints", {})
    artifacts = evaluation_setup.get("artifacts", {})
    threshold = results.get("optimal_threshold", {}).get("optimal_threshold", "n/a")

    per_turn_plot = artifacts.get("per_turn_auc_plot", "")
    per_turn_exists = Path(per_turn_plot).exists() if per_turn_plot else False
    error_exists = ERROR_JSON.exists()
    taxonomy_summary_exists = TAXONOMY_SUMMARY.exists()
    taxonomy_csv_exists = TAXONOMY_CSV.exists()

    return f"""# Paper Submission Notes

## Final Run Metadata

- Frozen threshold from dev tuning: `{threshold}`
- Prefix-based forecasting setup: `{evaluation_setup.get("dataset_view", "unknown")}`
- Train/dev/test claim: `{evaluation_setup.get("train_dev_test_claim", "unknown")}`

### Checkpoints used

- TRACER full: `{checkpoints.get("tracer_full", "missing")}`
- Features-only: `{checkpoints.get("features_only", "missing")}`
- Text-only: `{checkpoints.get("text_only", "missing")}`
- T5 recovery: `{checkpoints.get("t5_recovery", "missing")}`

### Artifact verification

- Results JSON: `{RESULTS_JSON}` -> `{RESULTS_JSON.exists()}`
- Error analysis JSON: `{ERROR_JSON}` -> `{error_exists}`
- Per-turn AUC plot: `{per_turn_plot}` -> `{per_turn_exists}`
- LLM-assisted taxonomy summary: `{TAXONOMY_SUMMARY}` -> `{taxonomy_summary_exists}`
- LLM-assisted taxonomy annotations: `{TAXONOMY_CSV}` -> `{taxonomy_csv_exists}`

## Required Paper Framing

- The failure taxonomy is a **4-core qualitative taxonomy**:
  - `Information Incompleteness`
  - `Contradiction / Misalignment`
  - `State Instability`
  - `Task Drift / Domain Confusion`
- Taxonomy outputs must be described as **LLM-assisted qualitative analysis**, not as ground-truth labels.
- Public datasets used in this project do **not** provide taxonomy labels.
- Recovery evaluation is an **offline proxy evaluation** and should not be described as direct proof of downstream task-success improvement unless additional evidence is added.
- The cross-domain transfer story is mixed and should be framed honestly.

## Suggested Limitations Paragraph

TRACER's recovery results are based on an offline proxy evaluation rather than a live interactive study, so they should be interpreted as evidence about intervention quality and cost rather than definitive downstream task-success gains. The failure taxonomy is also not a dataset-provided gold annotation scheme; instead, it is used as an LLM-assisted qualitative analysis framework for interpreting failure patterns. Finally, cross-domain transfer remains mixed, with more encouraging behavior on some datasets than others, so the generalization story should be presented as partial rather than universal.

## Cross-Domain Snapshot

{summarize_cross_domain(results)}

## Taxonomy Support

Use these files as supporting evidence for the taxonomy section:

- `{TAXONOMY_SUMMARY}`
- `{TAXONOMY_CSV}`

## Recovery Claim Constraint

Safe wording:

> We evaluate recovery strategies with an offline proxy that measures grounding, problem targeting, clarification intent, and intervention cost under a shared trigger policy.

Trigger-quality ablation wording:

> We further stratify triggered cases into early correct triggers, late correct triggers, and false positives to analyze whether recovery quality depends on the timing and correctness of the trigger.

Interpretation note:

- Early correct triggers test whether recovery has enough lead time to be useful.
- Late correct triggers test whether recovery can still help after delayed warning.
- False positives quantify the quality and cost of interventions on dialogues that would otherwise have succeeded.

Avoid unless you add stronger evidence:

> TRACER's recovery policy improves downstream task success.
"""


def main():
    results = load_results()

    table3 = export_table(
        results.get("table3_failure_prediction", []),
        OUTPUT_DIR / "paper_table3_failure_prediction.csv",
    )
    table4 = export_table(
        results.get("table4_recovery", []),
        OUTPUT_DIR / "paper_table4_recovery.csv",
    )
    table4b = export_table(
        results.get("table4b_recovery_by_trigger_quality", []),
        OUTPUT_DIR / "paper_table4b_recovery_by_trigger_quality.csv",
    )
    table6 = export_table(
        results.get("table6_ablations", []),
        OUTPUT_DIR / "paper_table6_ablations.csv",
    )

    notes = build_notes(results)
    notes_path = OUTPUT_DIR / "paper_submission_notes.md"
    notes_path.write_text(notes)

    print("Exported paper artifacts:")
    print(f"- {OUTPUT_DIR / 'paper_table3_failure_prediction.csv'} ({len(table3)} rows)")
    print(f"- {OUTPUT_DIR / 'paper_table4_recovery.csv'} ({len(table4)} rows)")
    print(f"- {OUTPUT_DIR / 'paper_table4b_recovery_by_trigger_quality.csv'} ({len(table4b)} rows)")
    print(f"- {OUTPUT_DIR / 'paper_table6_ablations.csv'} ({len(table6)} rows)")
    print(f"- {notes_path}")


if __name__ == "__main__":
    main()
