from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_3D_FIELDS = {
    "ap_position",
    "pd_position",
    "obstacle_min",
    "obstacle_max",
    "ris_default_position",
    "ris_normal",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Central configuration for the RIS-assisted VLC simulation."""

    # Project paths
    base_dir: Path = PROJECT_ROOT
    figures_dir: Path = PROJECT_ROOT / "figures"
    results_dir: Path = PROJECT_ROOT / "results"

    # Room geometry, in meters
    room_length: float = 5.0
    room_width: float = 5.0
    room_height: float = 3.0

    # Main node positions, in meters
    ap_position: tuple[float, float, float] = (2.5, 2.5, 3.0)
    pd_position: tuple[float, float, float] = (2.5, 1.0, 0.85)
    user_plane_z: float = 0.85

    # Rectangular obstacle modeled as an axis-aligned box, in meters.
    # It is placed between AP and the default PD so the basic LoS link is blocked.
    obstacle_min: tuple[float, float, float] = (2.15, 1.65, 0.85)
    obstacle_max: tuple[float, float, float] = (2.85, 1.95, 2.45)

    # RIS is fixed on the wall y = 0 and faces the room along +y.
    ris_wall_y: float = 0.0
    ris_default_position: tuple[float, float, float] = (2.5, 0.0, 1.6)
    ris_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ris_effective_area: float = 2.0
    ris_reflection_coefficient: float = 0.8
    ris_alignment_gain: float = 1.0

    # Optical and communication parameters
    led_transmit_power_w: float = 1.0
    pd_area_m2: float = 1e-4
    led_half_power_angle_deg: float = 60.0
    pd_fov_deg: float = 70.0
    optical_filter_gain: float = 1.0
    optical_concentrator_index: float = 1.5
    pd_responsivity_a_per_w: float = 0.53
    modulation_bandwidth_hz: float = 20e6
    noise_variance: float = 1e-14
    snr_threshold_db: float = 10.0

    # Sweep settings
    ris_x_min: float = 0.5
    ris_x_max: float = 4.5
    ris_z_min: float = 0.5
    ris_z_max: float = 2.8
    ris_x_points: int = 61
    ris_z_points: int = 61

    pd_grid_points: int = 70
    pd_grid_margin: float = 0.1

    # Plotting only: replace -inf SNR with this value so heatmaps remain readable.
    snr_plot_floor_db: float = -60.0


def load_config(config_path: str | Path | None = None, output_dir: str | Path | None = None) -> SimulationConfig:
    """Load a simulation config with optional JSON overrides and output directory."""

    config = SimulationConfig()
    if config_path is not None:
        config = apply_json_overrides(config, Path(config_path))

    if output_dir is not None:
        base_dir = Path(output_dir).expanduser().resolve()
        config = replace(
            config,
            base_dir=base_dir,
            figures_dir=base_dir / "figures",
            results_dir=base_dir / "results",
        )

    validate_config(config)
    return config


def apply_json_overrides(config: SimulationConfig, config_path: Path) -> SimulationConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw_overrides = json.load(handle)

    if not isinstance(raw_overrides, dict):
        raise ValueError("Config JSON must contain a flat object of SimulationConfig keys.")

    known_fields = {field.name: field for field in fields(SimulationConfig)}
    unknown_keys = sorted(set(raw_overrides) - set(known_fields))
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config key(s): {joined}")

    converted: dict[str, Any] = {}
    for key, value in raw_overrides.items():
        current_value = getattr(config, key)
        converted[key] = _convert_config_value(key, value, current_value)

    return replace(config, **converted)


def validate_config(config: SimulationConfig) -> None:
    positive_fields = [
        "room_length",
        "room_width",
        "room_height",
        "led_transmit_power_w",
        "pd_area_m2",
        "optical_filter_gain",
        "optical_concentrator_index",
        "pd_responsivity_a_per_w",
        "modulation_bandwidth_hz",
        "noise_variance",
        "ris_effective_area",
    ]
    for field_name in positive_fields:
        _require_positive(field_name, getattr(config, field_name))

    if not (0.0 < config.led_half_power_angle_deg < 90.0):
        raise ValueError("led_half_power_angle_deg must be between 0 and 90 degrees.")
    if not (0.0 < config.pd_fov_deg < 90.0):
        raise ValueError("pd_fov_deg must be between 0 and 90 degrees.")
    if not (0.0 <= config.ris_reflection_coefficient <= 1.0):
        raise ValueError("ris_reflection_coefficient must be between 0 and 1.")
    if not (0.0 <= config.ris_alignment_gain <= 1.0):
        raise ValueError("ris_alignment_gain must be between 0 and 1.")

    for field_name in ("ris_x_points", "ris_z_points", "pd_grid_points"):
        value = getattr(config, field_name)
        if value < 2:
            raise ValueError(f"{field_name} must be at least 2.")

    if config.ris_x_min >= config.ris_x_max:
        raise ValueError("ris_x_min must be smaller than ris_x_max.")
    if config.ris_z_min >= config.ris_z_max:
        raise ValueError("ris_z_min must be smaller than ris_z_max.")
    if config.pd_grid_margin < 0.0:
        raise ValueError("pd_grid_margin must be non-negative.")
    if config.pd_grid_margin * 2.0 >= min(config.room_length, config.room_width):
        raise ValueError("pd_grid_margin is too large for the room dimensions.")

    for field_name in VECTOR_3D_FIELDS:
        _convert_vector3(field_name, getattr(config, field_name))

    for min_value, max_value, axis in zip(config.obstacle_min, config.obstacle_max, "xyz"):
        if min_value >= max_value:
            raise ValueError(f"obstacle_min.{axis} must be smaller than obstacle_max.{axis}.")


def _convert_config_value(key: str, value: Any, current_value: Any) -> Any:
    if key in VECTOR_3D_FIELDS:
        return _convert_vector3(key, value)
    if isinstance(current_value, Path):
        return Path(value).expanduser()
    if isinstance(current_value, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean.")
        return value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} must be an integer.")
        return value
    if isinstance(current_value, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} must be a number.")
        return float(value)
    return value


def _convert_vector3(key: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-element list or tuple.")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in value):
        raise ValueError(f"{key} must contain only numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require_positive(field_name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
