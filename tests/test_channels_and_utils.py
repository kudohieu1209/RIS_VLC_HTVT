import math

import pytest

from ris_vlc_sim.config import SimulationConfig
from ris_vlc_sim.utils import blockage_factor, calculate_link_metrics, segment_intersects_box
from ris_vlc_sim.vlc_channel import lambertian_order, optical_concentrator_gain


def test_lambertian_order_for_60_degrees_is_one():
    assert lambertian_order(60.0) == pytest.approx(1.0)


def test_optical_concentrator_gain_is_zero_outside_fov():
    config = SimulationConfig()
    psi_rad = math.radians(config.pd_fov_deg + 1.0)

    assert optical_concentrator_gain(psi_rad, config) == 0.0


def test_segment_intersects_box_for_crossing_segment():
    assert segment_intersects_box(
        start=(0.0, 0.0, 0.0),
        end=(2.0, 2.0, 2.0),
        box_min=(0.5, 0.5, 0.5),
        box_max=(1.5, 1.5, 1.5),
    )


def test_segment_intersects_box_for_miss():
    assert not segment_intersects_box(
        start=(0.0, 0.0, 0.0),
        end=(0.0, 2.0, 0.0),
        box_min=(0.5, 0.5, 0.5),
        box_max=(1.5, 1.5, 1.5),
    )


def test_segment_intersects_box_for_axis_parallel_hit():
    assert segment_intersects_box(
        start=(0.0, 0.0, 0.0),
        end=(0.0, 2.0, 0.0),
        box_min=(-0.1, 0.5, -0.1),
        box_max=(0.1, 1.5, 0.1),
    )


def test_default_obstacle_blocks_default_los_path():
    config = SimulationConfig()

    assert blockage_factor(config.ap_position, config.pd_position, config) == 0.0


def test_zero_channel_has_zero_metrics():
    metrics = calculate_link_metrics(0.0, SimulationConfig())

    assert metrics["SNR_linear"] == 0.0
    assert math.isinf(metrics["SNR_dB"])
    assert metrics["SNR_dB"] < 0.0
    assert metrics["data_rate_bps"] == 0.0
