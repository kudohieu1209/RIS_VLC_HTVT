import pytest

from app import RoomFigureOptions, make_room_figure
from ris_vlc_sim.config import SimulationConfig


def trace_names(fig):
    return [trace.name for trace in fig.data if trace.name]


def test_room_figure_includes_core_markers_by_default():
    config = SimulationConfig()
    fig = make_room_figure(config, config.ris_default_position)

    names = trace_names(fig)

    assert "LED AP" in names
    assert "PD" in names
    assert "Tâm RIS" in names


def test_room_figure_can_hide_los_layer():
    config = SimulationConfig()
    options = RoomFigureOptions(show_los_path=False)
    fig = make_room_figure(config, config.ris_default_position, options)

    names = trace_names(fig)

    assert "LoS bị chắn" not in names
    assert "LoS trực tiếp" not in names


def test_room_figure_applies_selected_camera_view():
    config = SimulationConfig()
    options = RoomFigureOptions(view="Nhìn từ trên")
    fig = make_room_figure(config, config.ris_default_position, options)

    camera_eye = fig.layout.scene.camera.eye

    assert camera_eye.x == pytest.approx(0.0)
    assert camera_eye.y == pytest.approx(0.0)
    assert camera_eye.z == pytest.approx(2.8)


def test_room_figure_treats_hidden_obstacle_as_clear_los():
    config = SimulationConfig()
    options = RoomFigureOptions(show_obstacle=False)
    fig = make_room_figure(config, config.ris_default_position, options)

    names = trace_names(fig)
    los_trace = next(trace for trace in fig.data if trace.name == "LoS trực tiếp")

    assert "LoS bị chắn" not in names
    assert los_trace.line.dash == "solid"
