import numpy as np
import pandas as pd

from .ris_channel import ris_channel_gain
from .utils import blockage_factor, calculate_link_metrics
from .vlc_channel import los_channel_gain


def _scenario_row(scenario: str, h_los: float, h_ris: float, h_total: float, config) -> dict:
    metrics = calculate_link_metrics(h_total, config)
    return {
        "scenario": scenario,
        "H_LoS": h_los,
        "H_RIS": h_ris,
        "H_total": h_total,
        **metrics,
    }


def run_scenarios(config) -> pd.DataFrame:
    ap = config.ap_position
    pd_position = config.pd_position
    ris = config.ris_default_position

    h_los_raw = los_channel_gain(ap, pd_position, config)
    h_ris = ris_channel_gain(ap, ris, pd_position, config)
    blockage = blockage_factor(ap, pd_position, config)
    h_los_blocked = blockage * h_los_raw

    rows = [
        _scenario_row(
            "Scenario 1: No obstacle, without RIS",
            h_los_raw,
            0.0,
            h_los_raw,
            config,
        ),
        _scenario_row(
            "Scenario 2: Obstacle, without RIS",
            h_los_blocked,
            0.0,
            h_los_blocked,
            config,
        ),
        _scenario_row(
            "Scenario 3: Obstacle, with RIS",
            h_los_blocked,
            h_ris,
            h_los_blocked + h_ris,
            config,
        ),
        _scenario_row(
            "Scenario 4: No obstacle, with RIS",
            h_los_raw,
            h_ris,
            h_los_raw + h_ris,
            config,
        ),
    ]
    df = pd.DataFrame(rows)
    df["LoS_blocked"] = [
        False,
        blockage <= 0.0,
        blockage <= 0.0,
        False,
    ]
    return df


def run_ris_position_optimization(config) -> tuple[pd.DataFrame, pd.Series]:
    x_values = np.linspace(config.ris_x_min, config.ris_x_max, config.ris_x_points)
    z_values = np.linspace(config.ris_z_min, config.ris_z_max, config.ris_z_points)
    rows = []

    for x_ris in x_values:
        for z_ris in z_values:
            ris_position = (float(x_ris), config.ris_wall_y, float(z_ris))
            h_ris = ris_channel_gain(config.ap_position, ris_position, config.pd_position, config)
            metrics = calculate_link_metrics(h_ris, config)
            rows.append(
                {
                    "x_RIS_m": float(x_ris),
                    "y_RIS_m": config.ris_wall_y,
                    "z_RIS_m": float(z_ris),
                    "H_RIS": h_ris,
                    "H_total": h_ris,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    best_idx = df["data_rate_bps"].idxmax()
    return df, df.loc[best_idx]


def run_pd_snr_grid(config, with_ris: bool, ris_position=None, use_obstacle: bool = True):
    x_values = np.linspace(
        config.pd_grid_margin,
        config.room_length - config.pd_grid_margin,
        config.pd_grid_points,
    )
    y_values = np.linspace(
        config.pd_grid_margin,
        config.room_width - config.pd_grid_margin,
        config.pd_grid_points,
    )
    snr_db_grid = np.empty((len(y_values), len(x_values)), dtype=float)

    if ris_position is None:
        ris_position = config.ris_default_position

    for y_index, y_pd in enumerate(y_values):
        for x_index, x_pd in enumerate(x_values):
            pd_position = (float(x_pd), float(y_pd), config.user_plane_z)
            blockage = blockage_factor(config.ap_position, pd_position, config) if use_obstacle else 1.0
            h_los_eff = blockage * los_channel_gain(config.ap_position, pd_position, config)
            h_ris = 0.0
            if with_ris:
                h_ris = ris_channel_gain(config.ap_position, ris_position, pd_position, config)
            h_total = h_los_eff + h_ris
            snr_db_grid[y_index, x_index] = calculate_link_metrics(h_total, config)["SNR_dB"]

    return x_values, y_values, snr_db_grid
