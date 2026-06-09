from dataclasses import replace

from ris_vlc_sim.config import SimulationConfig
from ris_vlc_sim.simulation import run_ris_position_optimization, run_scenarios


def small_config() -> SimulationConfig:
    return replace(SimulationConfig(), ris_x_points=5, ris_z_points=5, pd_grid_points=5)


def test_run_scenarios_returns_expected_four_rows():
    df = run_scenarios(small_config())

    assert len(df) == 4
    assert list(df["scenario"]) == [
        "Scenario 1: No obstacle, without RIS",
        "Scenario 2: Obstacle, without RIS",
        "Scenario 3: Obstacle, with RIS",
        "Scenario 4: No obstacle, with RIS",
    ]


def test_ris_improves_obstacle_case_over_blocked_los_only():
    df = run_scenarios(small_config())
    without_ris = df.loc[df["scenario"] == "Scenario 2: Obstacle, without RIS"].iloc[0]
    with_ris = df.loc[df["scenario"] == "Scenario 3: Obstacle, with RIS"].iloc[0]

    assert without_ris["data_rate_Mbps"] == 0.0
    assert with_ris["data_rate_Mbps"] > without_ris["data_rate_Mbps"]


def test_optimization_best_position_stays_within_sweep_bounds():
    config = small_config()
    df, best_row = run_ris_position_optimization(config)

    assert len(df) == config.ris_x_points * config.ris_z_points
    assert config.ris_x_min <= best_row["x_RIS_m"] <= config.ris_x_max
    assert config.ris_z_min <= best_row["z_RIS_m"] <= config.ris_z_max
