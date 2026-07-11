#!/usr/bin/env python3
"""Compare completed grouped-student experiments without loading checkpoints."""

import argparse
import json
from pathlib import Path

from .grouped_student import EXPERIMENT_SPECS


def _number(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _millions(value):
    if value is None:
        return "-"
    return f"{float(value) / 1_000_000:.1f}"


def load_rows(root: Path):
    rows = []
    missing = []
    for name in EXPERIMENT_SPECS:
        summary_path = root / name / "summary.json"
        if not summary_path.exists():
            missing.append(name)
            continue
        summary = json.loads(summary_path.read_text())
        rows.append({
            "experiment": summary.get("experiment", name),
            "description": summary.get(
                "description", EXPERIMENT_SPECS[name]["description"]
            ),
            "summary_path": str(summary_path),
            "audio_dir": str(summary_path.parent / "audio")
            if (summary_path.parent / "audio").exists()
            else None,
            "student_parameters": summary.get("student_parameters"),
            "local_steps_per_frame": summary.get("local_steps_per_frame"),
            "best_step": summary.get("best_step"),
            "best_validation_loss": summary.get("best_validation_loss"),
            "best_validation_selection_loss": summary.get(
                "best_validation_selection_loss"
            ),
            "best_validation_rollout_loss": summary.get(
                "best_validation_rollout_loss"
            ),
            "best_validation_ground_truth_loss": summary.get(
                "best_validation_ground_truth_loss"
            ),
            "best_validation_distillation_loss": summary.get(
                "best_validation_distillation_loss"
            ),
            "teacher_token_accuracy": summary.get("teacher_token_accuracy"),
            "free_token_accuracy": summary.get("free_token_accuracy"),
            "free_coarse_accuracy_by_position_quartile": summary.get(
                "free_coarse_accuracy_by_position_quartile"
            ),
            "student_teacher_agreement": summary.get("student_teacher_agreement"),
            "student_free_teacher_agreement": summary.get(
                "student_free_teacher_agreement"
            ),
            "rollout_weight": summary.get("rollout_weight", 0.0),
        })
    return rows, missing


def print_table(rows, missing):
    print(
        "experiment       status  params(M) steps  val_loss  rollout  val_gt  "
        "val_kd  teacher_acc  free_acc  student_teacher"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['experiment']:<16} {'done':<7} "
            f"{_millions(row['student_parameters']):>8} "
            f"{str(row['local_steps_per_frame'] or '-'):>5} "
            f"{_number(row['best_validation_loss']):>9} "
            f"{_number(row['best_validation_rollout_loss']):>7} "
            f"{_number(row['best_validation_ground_truth_loss']):>7} "
            f"{_number(row['best_validation_distillation_loss']):>7} "
            f"{_number(row['teacher_token_accuracy']):>11} "
            f"{_number(row['free_token_accuracy']):>9} "
            f"{_number(row['student_teacher_agreement']):>16}"
        )
    for name in missing:
        print(f"{name:<16} {'missing':<7}")
    if not rows:
        print("No completed experiment summaries found.")
    elif missing:
        print(f"Missing summaries: {', '.join(missing)}")
    print()
    print("Audio directories:")
    for row in rows:
        print(f"  {row['experiment']}: {row['audio_dir'] or '-'}")
    print()
    ranked = sorted(
        (row for row in rows if row["free_token_accuracy"] is not None),
        key=lambda row: row["free_token_accuracy"],
        reverse=True,
    )
    if ranked:
        print(
            "Highest free token accuracy: "
            f"{ranked[0]['experiment']} ({ranked[0]['free_token_accuracy']:.4f})"
        )
    print(
        "Note: combined validation loss is only directly comparable within the "
        "same objective weights. Use free accuracy, agreement, and listening tests "
        "together when choosing a variant."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/moss_experiments")
    parser.add_argument(
        "--output",
        default="evaluation/moss_experiments_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.root)
    rows, missing = load_rows(root)
    result = {
        "root": str(root),
        "expected_experiments": list(EXPERIMENT_SPECS),
        "completed": len(rows),
        "missing": missing,
        "experiments": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print_table(rows, missing)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
