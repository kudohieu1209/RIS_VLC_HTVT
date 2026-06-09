from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .plotting import (
    plot_rate_comparison,
    plot_ris_optimization_3d,
    plot_ris_optimization_heatmap,
    plot_room_top_view,
    plot_snr_comparison,
    plot_snr_heatmap,
    plot_system_geometry_3d,
)
from .report import write_summary_report
from .simulation import run_pd_snr_grid, run_ris_position_optimization, run_scenarios
from .utils import ensure_directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RIS-assisted indoor VLC simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a flat JSON file that overrides SimulationConfig values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where figures/ and results/ folders will be created.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation and only write CSV/report outputs.",
    )
    parser.add_argument(
        "--open-figures",
        action="store_true",
        help="Open the figures folder after the simulation finishes. Useful for classroom demo.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_simulation(config, generate_plots: bool = True, open_figures: bool = False) -> dict[str, Path]:
    ensure_directories(config.figures_dir, config.results_dir)

    scenario_df = run_scenarios(config)
    scenario_csv = config.results_dir / "scenario_results.csv"
    scenario_df.to_csv(scenario_csv, index=False)

    optimization_df, best_row = run_ris_position_optimization(config)
    optimization_csv = config.results_dir / "ris_position_optimization.csv"
    optimization_df.to_csv(optimization_csv, index=False)

    best_ris_position = (
        float(best_row["x_RIS_m"]),
        float(best_row["y_RIS_m"]),
        float(best_row["z_RIS_m"]),
    )

    if generate_plots:
        plot_snr_comparison(scenario_df, config.figures_dir / "snr_comparison.png")
        plot_rate_comparison(scenario_df, config.figures_dir / "data_rate_comparison.png")
        plot_ris_optimization_3d(
            optimization_df,
            best_row,
            config.figures_dir / "ris_position_optimization_3d.png",
        )
        plot_ris_optimization_heatmap(
            optimization_df,
            best_row,
            config.figures_dir / "ris_position_optimization_heatmap.png",
        )
        plot_system_geometry_3d(
            config,
            best_ris_position,
            config.figures_dir / "system_geometry_3d.png",
        )
        plot_room_top_view(
            config,
            best_ris_position,
            config.figures_dir / "room_top_view.png",
        )

        x_no_ris, y_no_ris, snr_no_ris = run_pd_snr_grid(config, with_ris=False, use_obstacle=True)
        plot_snr_heatmap(
            x_no_ris,
            y_no_ris,
            snr_no_ris,
            config.figures_dir / "snr_heatmap_without_ris.png",
            "SNR heatmap with obstacle, without RIS",
            config.snr_plot_floor_db,
        )

        x_with_ris, y_with_ris, snr_with_ris = run_pd_snr_grid(
            config,
            with_ris=True,
            ris_position=best_ris_position,
            use_obstacle=True,
        )
        plot_snr_heatmap(
            x_with_ris,
            y_with_ris,
            snr_with_ris,
            config.figures_dir / "snr_heatmap_with_ris.png",
            "SNR heatmap with obstacle, with optimized RIS",
            config.snr_plot_floor_db,
        )

    obstacle_without_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 2: Obstacle, without RIS",
        "data_rate_Mbps",
    ].iloc[0]
    obstacle_with_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 3: Obstacle, with RIS",
        "data_rate_Mbps",
    ].iloc[0]
    delta_rate_mbps = obstacle_with_ris - obstacle_without_ris

    summary_report = config.results_dir / "simulation_summary.md"
    write_summary_report(config, scenario_df, best_row, summary_report)

    print("RIS-assisted VLC indoor simulation completed.")
    print(f"Scenario results saved to: {scenario_csv}")
    print(f"RIS sweep results saved to: {optimization_csv}")
    print("")
    print("Optimal RIS position on wall y = 0:")
    print(f"  x_opt = {best_row['x_RIS_m']:.3f} m")
    print(f"  z_opt = {best_row['z_RIS_m']:.3f} m")
    print(f"  max SNR = {best_row['SNR_dB']:.3f} dB")
    print(f"  max data rate = {best_row['data_rate_Mbps']:.3f} Mbps")
    print("")
    print("Scenario summary:")
    for _, row in scenario_df.iterrows():
        snr_text = "-inf" if row["SNR_linear"] <= 0.0 else f"{row['SNR_dB']:.3f} dB"
        print(f"  {row['scenario']}: SNR = {snr_text}, Rate = {row['data_rate_Mbps']:.3f} Mbps")
    print(f"  Delta rate for obstacle case with RIS: {delta_rate_mbps:.3f} Mbps")
    print(f"Summary report saved to: {summary_report}")

    if generate_plots:
        print("")
        print("Generated figures:")
        for figure_path in sorted(config.figures_dir.glob("*.png")):
            print(f"  {figure_path}")
    else:
        print("Figure generation skipped because --no-plots was used.")

    if open_figures and generate_plots:
        open_folder(config.figures_dir)
    elif open_figures:
        print("Figures folder was not opened because --no-plots was used.")

    return {
        "scenario_csv": scenario_csv,
        "optimization_csv": optimization_csv,
        "summary_report": summary_report,
    }


def open_folder(path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except OSError as exc:
        print(f"Could not open folder automatically: {exc}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    run_simulation(config, generate_plots=not args.no_plots, open_figures=args.open_figures)


if __name__ == "__main__":
    main()
