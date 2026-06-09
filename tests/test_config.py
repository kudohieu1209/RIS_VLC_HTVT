import json

import pytest

from ris_vlc_sim.config import SimulationConfig, load_config


def test_load_config_applies_flat_json_overrides(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "pd_position": [2.4, 1.1, 0.85],
                "ris_x_points": 7,
                "noise_variance": 2e-14,
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.pd_position == (2.4, 1.1, 0.85)
    assert config.ris_x_points == 7
    assert config.noise_variance == 2e-14


def test_load_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"unknown_key": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(config_path)


def test_load_config_rejects_invalid_vector(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"pd_position": [1.0, 2.0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="pd_position"):
        load_config(config_path)


def test_load_config_rejects_invalid_grid_size(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ris_x_points": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="ris_x_points"):
        load_config(config_path)


def test_output_dir_overrides_default_result_locations(tmp_path):
    config = load_config(output_dir=tmp_path)

    assert config.base_dir == tmp_path.resolve()
    assert config.figures_dir == tmp_path.resolve() / "figures"
    assert config.results_dir == tmp_path.resolve() / "results"


def test_default_config_is_valid():
    assert isinstance(load_config(), SimulationConfig)
