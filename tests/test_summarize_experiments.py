import json

from minimoss.summarize_experiments import load_rows


def test_load_rows_reports_completed_and_missing_variants(tmp_path):
    output_dir = tmp_path / "baseline11"
    output_dir.mkdir()
    (output_dir / "audio").mkdir()
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "baseline11",
                "student_parameters": 10,
                "local_steps_per_frame": 11,
                "best_validation_loss": 1.0,
                "free_token_accuracy": 0.5,
            }
        )
    )

    rows, missing = load_rows(tmp_path)

    assert [row["experiment"] for row in rows] == ["baseline11"]
    assert rows[0]["audio_dir"].endswith("baseline11/audio")
    assert "large11" in missing
