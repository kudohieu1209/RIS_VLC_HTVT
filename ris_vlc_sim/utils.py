import math
from pathlib import Path

import numpy as np


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def as_vector(point: tuple[float, float, float]) -> np.ndarray:
    return np.array(point, dtype=float)


def distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_b - point_a))


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def linear_to_db(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return float("-inf")
    return 10.0 * math.log10(value)


def db_for_plot(values: np.ndarray, floor_db: float) -> np.ndarray:
    values = np.array(values, dtype=float)
    return np.where(np.isfinite(values), values, floor_db)


def calculate_link_metrics(h_total: float, config) -> dict[str, float]:
    h_total = max(float(h_total), 0.0)
    pr_w = config.led_transmit_power_w * h_total
    snr_linear = ((config.pd_responsivity_a_per_w * pr_w) ** 2) / config.noise_variance
    snr_db = linear_to_db(snr_linear)
    data_rate_bps = config.modulation_bandwidth_hz * math.log2(1.0 + snr_linear)
    return {
        "Pr_W": pr_w,
        "SNR_linear": snr_linear,
        "SNR_dB": snr_db,
        "data_rate_bps": data_rate_bps,
        "data_rate_Mbps": data_rate_bps / 1e6,
    }


def segment_intersects_box(start, end, box_min, box_max) -> bool:
    """Return True when a line segment intersects an axis-aligned 3D box."""

    start = as_vector(start)
    end = as_vector(end)
    box_min = as_vector(box_min)
    box_max = as_vector(box_max)
    direction = end - start
    t_min = 0.0
    t_max = 1.0

    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < box_min[axis] or start[axis] > box_max[axis]:
                return False
            continue

        inv_direction = 1.0 / direction[axis]
        t1 = (box_min[axis] - start[axis]) * inv_direction
        t2 = (box_max[axis] - start[axis]) * inv_direction
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)

        if t_min > t_max:
            return False

    return True


def blockage_factor(ap_position, pd_position, config) -> float:
    return 0.0 if segment_intersects_box(
        ap_position,
        pd_position,
        config.obstacle_min,
        config.obstacle_max,
    ) else 1.0
