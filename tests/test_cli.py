import json
import subprocess
import sys


def test_cli_no_plots_smoke_writes_csv_and_report(tmp_path):
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "out"
    config_path.write_text(
        json.dumps(
            {
                "ris_x_points": 4,
                "ris_z_points": 4,
                "pd_grid_points": 4,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ris_vlc_sim.cli",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Figure generation skipped" in result.stdout
    assert (output_dir / "results" / "scenario_results.csv").exists()
    assert (output_dir / "results" / "ris_position_optimization.csv").exists()
    assert (output_dir / "results" / "simulation_summary.md").exists()
