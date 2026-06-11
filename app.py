from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ris_vlc_sim.config import SimulationConfig, validate_config
from ris_vlc_sim.simulation import run_pd_snr_grid, run_ris_position_optimization, run_scenarios
from ris_vlc_sim.utils import blockage_factor, db_for_plot


st.set_page_config(
    page_title="Mô phỏng RIS/VLC",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


SCENARIO_LABELS = {
    "Scenario 1: No obstacle, without RIS": "S1: LoS",
    "Scenario 2: Obstacle, without RIS": "S2: Bị chắn",
    "Scenario 3: Obstacle, with RIS": "S3: Có RIS",
    "Scenario 4: No obstacle, with RIS": "S4: LoS + RIS",
}

SCENARIO_COLORS = {
    "S1: LoS": "#1b7f5f",
    "S2: Bị chắn": "#b42318",
    "S3: Có RIS": "#2563eb",
    "S4: LoS + RIS": "#7c3aed",
}

ACCENT_COLORS = {
    "rate": "#2563eb",
    "snr": "#0f766e",
    "baseline": "#7c3aed",
    "risk": "#b42318",
}

APPLE_FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif'

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


ROOM_VIEW_CAMERAS = {
    "Góc 3D": dict(eye=dict(x=1.55, y=-1.95, z=1.15), center=dict(x=0, y=0, z=-0.08)),
    "Nhìn từ trên": dict(eye=dict(x=0.0, y=0.0, z=2.8), center=dict(x=0, y=0, z=0)),
    "Nhìn tường RIS": dict(eye=dict(x=0.0, y=-2.65, z=0.82), center=dict(x=0, y=0, z=0)),
    "Nhìn từ phía PD": dict(eye=dict(x=0.0, y=2.55, z=0.95), center=dict(x=0, y=0, z=0)),
}


@dataclass(frozen=True)
class RoomFigureOptions:
    view: str = "Góc 3D"
    show_room_surfaces: bool = True
    show_room_edges: bool = True
    show_user_plane: bool = True
    show_obstacle: bool = True
    show_los_path: bool = True
    show_ris_path: bool = True
    show_markers: bool = True
    show_technical_details: bool = True


def main() -> None:
    apply_theme()
    config = sidebar_config()

    try:
        validate_config(config)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    scenario_df, optimization_df, best_row = run_cached_core(config)
    best_ris_position = (
        float(best_row["x_RIS_m"]),
        float(best_row["y_RIS_m"]),
        float(best_row["z_RIS_m"]),
    )

    render_hero(config)
    render_metrics(scenario_df, best_row)
    render_dashboard_tabs(config, scenario_df, optimization_df, best_row, best_ris_position)


def render_hero(config: SimulationConfig) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-copy">
                <div class="eyebrow">RIS / VLC Indoor Link Simulator</div>
                <h1>Tối ưu hiệu suất hệ thống VLC trong nhà có hỗ trợ RIS</h1>
                <div class="hero-meta">
                    <span>Phòng {config.room_length:.0f} x {config.room_width:.0f} x {config.room_height:.0f} m</span>
                    <span>4 kịch bản truyền dẫn</span>
                    <span>RIS trên tường y = {config.ris_wall_y:.0f}</span>
                    <span>Ngưỡng SNR {config.snr_threshold_db:.0f} dB</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_tabs(
    config: SimulationConfig,
    scenario_df: pd.DataFrame,
    optimization_df: pd.DataFrame,
    best_row: pd.Series,
    best_ris_position: tuple[float, float, float],
) -> None:
    overview_tab, optimization_tab, room_tab, snr_tab = st.tabs(
        ["Tổng quan", "Tối ưu RIS", "Mô hình 3D", "Bản đồ SNR"]
    )

    with overview_tab:
        render_overview_panel(scenario_df)
    with optimization_tab:
        with st.container(border=True):
            render_optimization_panel(optimization_df, best_row)
    with room_tab:
        with st.container(border=True):
            render_section_header(
                "Hình học phòng 3D",
                "",
            )
            render_room_3d_panel(config, best_ris_position, best_row)
    with snr_tab:
        render_snr_maps_panel(config, best_ris_position)


def render_overview_panel(scenario_df: pd.DataFrame) -> None:
    with st.container(border=True):
        render_section_header(
            "So sánh các kịch bản truyền dẫn",
            "Đối chiếu ảnh hưởng của vật cản lên đường LoS và mức cải thiện khi bổ sung đường phản xạ qua RIS.",
        )
        render_scenario_table(scenario_df)
        rate_col, snr_col = st.columns(2, gap="large")
        with rate_col:
            st.plotly_chart(make_rate_bar(scenario_df), use_container_width=True, config=PLOTLY_CONFIG)
        with snr_col:
            st.plotly_chart(make_snr_bar(scenario_df), use_container_width=True, config=PLOTLY_CONFIG)


def render_snr_maps_panel(config: SimulationConfig, best_ris_position: tuple[float, float, float]) -> None:
    with st.container(border=True):
        render_section_header(
            "Bản đồ SNR trên mặt phẳng người dùng",
            "Hai bản đồ cho thấy phân bố SNR tại mặt phẳng người dùng trước và sau khi sử dụng vị trí RIS tối ưu.",
        )
        map_left, map_right = st.columns(2, gap="large")
        with map_left:
            x_no_ris, y_no_ris, snr_no_ris = run_cached_grid(config, False, best_ris_position)
            st.plotly_chart(
                make_snr_heatmap(
                    x_no_ris,
                    y_no_ris,
                    snr_no_ris,
                    config.snr_plot_floor_db,
                    "Có vật cản, không RIS",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with map_right:
            x_with_ris, y_with_ris, snr_with_ris = run_cached_grid(config, True, best_ris_position)
            st.plotly_chart(
                make_snr_heatmap(
                    x_with_ris,
                    y_with_ris,
                    snr_with_ris,
                    config.snr_plot_floor_db,
                    "Có vật cản, dùng RIS tối ưu",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config() -> SimulationConfig:
    base = SimulationConfig()
    resolution = (61, 61, 70)

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">
                <span>Bài tập lớn</span>
                <h1>Hệ thống viễn thông</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="team-card">
                <div>
                    <span>Giảng viên hướng dẫn</span>
                    <strong>Nguyễn Thành Chuyên</strong>
                </div>
                <div>
                    <span>Sinh viên thực hiện</span>
                    <strong>Mai Duy Hiếu - 20223967</strong>
                    <strong>Nguyễn Kim Đạt - 20223762</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            """
            <div class="control-heading">
                <span>Điều chỉnh mô phỏng</span>
                <strong>Tham số đầu vào</strong>
                <p>Các thanh trượt này điều khiển KPI, SNR, data rate, tối ưu RIS và mô hình 3D.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Bộ thu PD", expanded=True):
            st.caption("Điều chỉnh vị trí người dùng trên mặt phẳng thu.")
            pd_x = st.slider("Tọa độ PD x (m)", 0.1, base.room_length - 0.1, base.pd_position[0], 0.05)
            pd_y = st.slider("Tọa độ PD y (m)", 0.1, base.room_width - 0.1, base.pd_position[1], 0.05)
            pd_z = st.slider("Tọa độ PD z (m)", 0.2, base.room_height - 0.1, base.pd_position[2], 0.05)

        with st.expander("RIS", expanded=True):
            st.caption("Tham số phản xạ của bề mặt thông minh trên tường y = 0.")
            ris_area = st.slider("Diện tích (m²)", 0.2, 5.0, base.ris_effective_area, 0.1)
            reflection = st.slider("Hệ số phản xạ", 0.0, 1.0, base.ris_reflection_coefficient, 0.05)
            alignment = st.slider("Độ căn chỉnh", 0.0, 1.0, base.ris_alignment_gain, 0.05)

        with st.expander("Liên kết quang", expanded=True):
            st.caption("Cấu hình công suất phát, FoV, băng thông và nhiễu.")
            power = st.slider("Công suất LED (W)", 0.1, 5.0, base.led_transmit_power_w, 0.1)
            fov = st.slider("Góc FoV của PD (độ)", 20.0, 85.0, base.pd_fov_deg, 1.0)
            bandwidth_mhz = st.slider("Băng thông (MHz)", 1.0, 100.0, base.modulation_bandwidth_hz / 1e6, 1.0)
            noise_exp = st.slider("Số mũ phương sai nhiễu", -16, -10, -14, 1)

    return replace(
        base,
        pd_position=(float(pd_x), float(pd_y), float(pd_z)),
        user_plane_z=float(pd_z),
        ris_effective_area=float(ris_area),
        ris_reflection_coefficient=float(reflection),
        ris_alignment_gain=float(alignment),
        led_transmit_power_w=float(power),
        pd_fov_deg=float(fov),
        modulation_bandwidth_hz=float(bandwidth_mhz) * 1e6,
        noise_variance=10.0 ** int(noise_exp),
        ris_x_points=resolution[0],
        ris_z_points=resolution[1],
        pd_grid_points=resolution[2],
    )


@st.cache_data(show_spinner=False)
def run_cached_core(config: SimulationConfig):
    scenario_df = run_scenarios(config)
    optimization_df, best_row = run_ris_position_optimization(config)
    return scenario_df, optimization_df, best_row


@st.cache_data(show_spinner=False)
def run_cached_grid(config: SimulationConfig, with_ris: bool, ris_position: tuple[float, float, float]):
    return run_pd_snr_grid(
        config,
        with_ris=with_ris,
        ris_position=ris_position,
        use_obstacle=True,
    )


def render_metrics(scenario_df: pd.DataFrame, best_row: pd.Series) -> None:
    obstacle_without_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 2: Obstacle, without RIS"
    ].iloc[0]
    obstacle_with_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 3: Obstacle, with RIS"
    ].iloc[0]
    baseline = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 1: No obstacle, without RIS"
    ].iloc[0]
    delta_rate = obstacle_with_ris["data_rate_Mbps"] - obstacle_without_ris["data_rate_Mbps"]

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card metric-risk">
                <span>Trạng thái LoS</span>
                <strong>{'Bị chắn' if obstacle_without_ris['LoS_blocked'] else 'Thông suốt'}</strong>
                <small>Kịch bản có vật cản, không RIS</small>
            </div>
            <div class="metric-card metric-rate">
                <span>Mức tăng khi dùng RIS</span>
                <strong>{delta_rate:.2f} Mbps</strong>
                <small>So với trường hợp LoS bị chắn</small>
            </div>
            <div class="metric-card metric-rate">
                <span>Data rate RIS tối ưu</span>
                <strong>{best_row['data_rate_Mbps']:.2f} Mbps</strong>
                <small>Vị trí RIS tốt nhất trong lưới quét</small>
            </div>
            <div class="metric-card metric-snr">
                <span>SNR tại RIS tối ưu</span>
                <strong>{best_row['SNR_dB']:.2f} dB</strong>
                <small>Ngưỡng tham chiếu: 10 dB</small>
            </div>
            <div class="metric-card metric-baseline">
                <span>Data rate LoS chuẩn</span>
                <strong>{baseline['data_rate_Mbps']:.2f} Mbps</strong>
                <small>Kịch bản không có vật cản</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_table(scenario_df: pd.DataFrame) -> None:
    display_df = scenario_df.copy()
    display_df["scenario"] = display_df["scenario"].map(SCENARIO_LABELS)
    display_df = display_df[
        [
            "scenario",
            "H_LoS",
            "H_RIS",
            "SNR_dB",
            "data_rate_Mbps",
        ]
    ].rename(
        columns={
            "scenario": "Kịch bản",
            "H_LoS": "H LoS",
            "H_RIS": "H RIS",
            "SNR_dB": "SNR dB",
            "data_rate_Mbps": "Data rate Mbps",
        }
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "H LoS": st.column_config.NumberColumn(format="%.3e"),
            "H RIS": st.column_config.NumberColumn(format="%.3e"),
            "SNR dB": st.column_config.NumberColumn(format="%.2f"),
            "Data rate Mbps": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def render_optimization_panel(optimization_df: pd.DataFrame, best_row: pd.Series) -> None:
    render_section_header(
        "Tối ưu vị trí RIS",
        "RIS được quét trên mặt phẳng tường y=0; tiêu chí tối ưu là data rate của tuyến AP-RIS-PD khi đường LoS bị vật cản che.",
    )
    st.markdown(
        f"""
        <div class="explain-box">
            <b>Ý nghĩa bản đồ tối ưu RIS:</b>
            mỗi điểm trên heatmap tương ứng với một vị trí đặt RIS trên tường <code>y=0</code>.
            Trục x biểu diễn vị trí ngang của RIS, trục z biểu diễn độ cao của RIS.
            Thang màu biểu diễn data rate thu được; vùng xanh đậm cho thấy vị trí đặt RIS hiệu quả hơn.
            Với cấu hình hiện tại, vị trí tối ưu là
            <b>x={best_row['x_RIS_m']:.2f} m</b>, <b>z={best_row['z_RIS_m']:.2f} m</b>,
            đạt <b>{best_row['data_rate_Mbps']:.2f} Mbps</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_optimization_heatmap(optimization_df, best_row), use_container_width=True, config=PLOTLY_CONFIG)


def make_rate_bar(scenario_df: pd.DataFrame) -> go.Figure:
    df = scenario_df.copy()
    df["label"] = df["scenario"].map(SCENARIO_LABELS)
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["data_rate_Mbps"],
            marker_color=[SCENARIO_COLORS[label] for label in df["label"]],
            marker_line=dict(color="rgba(255,255,255,0.78)", width=1),
            text=[f"{value:.1f}" for value in df["data_rate_Mbps"]],
            textposition="outside",
            hovertemplate="%{x}<br>Data rate=%{y:.3f} Mbps<extra></extra>",
        )
    )
    fig.update_layout(
        title="Data rate theo kịch bản",
        yaxis_title="Mbps",
        height=310,
        margin=dict(l=8, r=8, t=42, b=12),
    )
    return style_figure(fig)


def make_snr_bar(scenario_df: pd.DataFrame) -> go.Figure:
    df = scenario_df.copy()
    df["label"] = df["scenario"].map(SCENARIO_LABELS)
    values = db_for_plot(df["SNR_dB"].to_numpy(), floor_db=-60.0)
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=values,
            marker_color=[SCENARIO_COLORS[label] for label in df["label"]],
            marker_line=dict(color="rgba(255,255,255,0.78)", width=1),
            text=["-inf" if not np.isfinite(value) else f"{value:.1f}" for value in df["SNR_dB"]],
            textposition="outside",
            hovertemplate="%{x}<br>SNR=%{text} dB<extra></extra>",
        )
    )
    fig.add_hline(y=10.0, line_dash="dash", line_color="#475569", annotation_text="10 dB")
    fig.update_layout(
        title="SNR theo kịch bản",
        yaxis_title="dB",
        height=330,
        margin=dict(l=8, r=8, t=42, b=12),
    )
    return style_figure(fig)


def make_optimization_heatmap(optimization_df: pd.DataFrame, best_row: pd.Series) -> go.Figure:
    pivot = optimization_df.pivot(index="z_RIS_m", columns="x_RIS_m", values="data_rate_Mbps")
    best_x = float(best_row["x_RIS_m"])
    best_z = float(best_row["z_RIS_m"])
    best_rate = float(best_row["data_rate_Mbps"])
    fig = go.Figure(
        go.Heatmap(
            x=pivot.columns,
            y=pivot.index,
            z=pivot.to_numpy(),
            colorscale=[
                [0.0, "#f8fafc"],
                [0.22, "#bfdbfe"],
                [0.5, "#60a5fa"],
                [0.78, "#2563eb"],
                [1.0, "#172554"],
            ],
            colorbar=dict(title="Data rate<br>(Mbps)"),
            hovertemplate="Đặt RIS tại:<br>x=%{x:.2f} m<br>z=%{y:.2f} m<br>Data rate=%{z:.3f} Mbps<extra></extra>",
        )
    )
    fig.add_vline(
        x=best_x,
        line_width=1.5,
        line_dash="dot",
        line_color=ACCENT_COLORS["risk"],
        opacity=0.72,
    )
    fig.add_hline(
        y=best_z,
        line_width=1.5,
        line_dash="dot",
        line_color=ACCENT_COLORS["risk"],
        opacity=0.72,
    )
    fig.add_trace(
        go.Scatter(
            x=[best_x],
            y=[best_z],
            mode="markers",
            marker=dict(color=ACCENT_COLORS["risk"], size=14, symbol="x", line=dict(width=2)),
            hovertemplate=(
                "Vị trí tối ưu<br>"
                "x=%{x:.2f} m<br>"
                "z=%{y:.2f} m<br>"
                f"Data rate={best_rate:.2f} Mbps"
                "<extra></extra>"
            ),
        )
    )
    fig.add_annotation(
        x=best_x,
        y=best_z,
        text=f"<b>Vị trí tối ưu</b><br>{best_rate:.2f} Mbps",
        showarrow=False,
        xshift=12,
        yshift=24,
        xanchor="left",
        yanchor="bottom",
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(184,35,24,0.45)",
        borderwidth=1,
        font=dict(size=12, color="#17212b"),
    )
    fig.update_layout(
        title="Heatmap data rate khi quét vị trí RIS trên tường y=0",
        xaxis_title="Tọa độ x của RIS trên tường y=0 (m)",
        yaxis_title="Tọa độ z của RIS (m)",
        height=405,
        margin=dict(l=8, r=8, t=42, b=12),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        showarrow=False,
        align="left",
        text="<b>Xanh đậm:</b> data rate cao<br><b>Xanh nhạt:</b> data rate thấp",
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="#d8e0ea",
        borderwidth=1,
        font=dict(size=12, color="#17212b"),
    )
    return style_figure(fig)


def make_snr_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    snr_grid: np.ndarray,
    floor_db: float,
    title: str,
) -> go.Figure:
    display_grid = db_for_plot(snr_grid, floor_db=floor_db)
    fig = go.Figure(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=display_grid,
            zmin=floor_db,
            colorscale=[
                [0.0, "#111827"],
                [0.25, "#7f1d1d"],
                [0.5, "#dc2626"],
                [0.75, "#f59e0b"],
                [1.0, "#fef3c7"],
            ],
            colorbar=dict(title="dB"),
            hovertemplate="PD x=%{x:.2f} m<br>PD y=%{y:.2f} m<br>SNR=%{z:.2f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tọa độ PD x (m)",
        yaxis_title="Tọa độ PD y (m)",
        height=390,
        margin=dict(l=8, r=8, t=42, b=12),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return style_figure(fig)


def render_room_3d_panel(
    config: SimulationConfig,
    best_ris_position: tuple[float, float, float],
    best_row: pd.Series,
) -> None:
    control_left, control_right = st.columns([2.2, 1.0], gap="large")
    with control_left:
        view = st.radio(
            "Góc nhìn",
            list(ROOM_VIEW_CAMERAS),
            horizontal=True,
            key="room_3d_view",
        )
    with control_right:
        show_technical_details = st.checkbox(
            "Hiển thị chi tiết kỹ thuật",
            value=True,
            key="room_3d_technical_details",
        )

    layer_specs = [
        ("Mặt sàn/tường", "show_room_surfaces", True),
        ("Mặt phẳng PD", "show_user_plane", True),
        ("Vật cản", "show_obstacle", True),
        ("LoS", "show_los_path", True),
        ("AP-RIS-PD", "show_ris_path", True),
    ]
    layer_columns = st.columns(len(layer_specs), gap="small")
    selected_layers: dict[str, bool] = {
        "show_room_edges": True,
        "show_markers": True,
    }
    for column, (label, option_name, default_value) in zip(layer_columns, layer_specs):
        with column:
            selected_layers[option_name] = st.checkbox(
                label,
                value=default_value,
                key=f"room_3d_{option_name}",
            )

    options = RoomFigureOptions(
        view=view,
        show_room_surfaces=selected_layers["show_room_surfaces"],
        show_room_edges=selected_layers["show_room_edges"],
        show_user_plane=selected_layers["show_user_plane"],
        show_obstacle=selected_layers["show_obstacle"],
        show_los_path=selected_layers["show_los_path"],
        show_ris_path=selected_layers["show_ris_path"],
        show_markers=selected_layers["show_markers"],
        show_technical_details=show_technical_details,
    )

    los_blocked = is_los_blocked_in_room_view(config, options)
    st.markdown(
        f"""
        <div class="room-summary">
            <div><span>AP</span><strong>{format_point(config.ap_position)}</strong></div>
            <div><span>PD</span><strong>{format_point(config.pd_position)}</strong></div>
            <div><span>RIS tối ưu</span><strong>{format_point(best_ris_position)}</strong></div>
            <div><span>LoS</span><strong>{'Bị chắn' if los_blocked else 'Thông suốt'}</strong></div>
            <div><span>Data rate / SNR</span><strong>{best_row['data_rate_Mbps']:.2f} Mbps · {best_row['SNR_dB']:.2f} dB</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_room_figure(config, best_ris_position, options),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )


def format_point(point: tuple[float, float, float]) -> str:
    return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) m"


def camera_for_view(view: str) -> dict:
    return ROOM_VIEW_CAMERAS.get(view, ROOM_VIEW_CAMERAS["Góc 3D"])


def make_room_figure(
    config: SimulationConfig,
    ris_position: tuple[float, float, float],
    options: RoomFigureOptions | None = None,
) -> go.Figure:
    if options is None:
        options = RoomFigureOptions()

    ap = np.array(config.ap_position, dtype=float)
    pd_pos = np.array(config.pd_position, dtype=float)
    ris = np.array(ris_position, dtype=float)
    fig = go.Figure()

    if options.show_room_surfaces:
        add_room_surfaces(fig, config, options.show_technical_details)
    if options.show_user_plane:
        add_user_plane(fig, config, options.show_technical_details)
    if options.show_room_edges:
        add_room_edges(fig, config)
    if options.show_obstacle:
        add_obstacle_mesh(fig, config, options.show_technical_details)
        add_obstacle_edges(fig, config)

    add_ris_panel(fig, ris, config.ris_effective_area, options.show_technical_details)

    los_blocked = is_los_blocked_in_room_view(config, options)
    if options.show_los_path:
        add_path(
            fig,
            ap,
            pd_pos,
            "#d00000" if los_blocked else "#0f766e",
            "LoS bị chắn" if los_blocked else "LoS trực tiếp",
            "Tuyến trực tiếp LED AP - PD đi qua vùng vật cản."
            if los_blocked
            else "Tuyến trực tiếp LED AP - PD không cắt vật cản.",
            dash="dash" if los_blocked else "solid",
            technical_details=options.show_technical_details,
        )
    if options.show_ris_path:
        add_ris_path_segments(fig, ap, ris, pd_pos, options.show_technical_details)
    if options.show_markers:
        add_marker(fig, ap, "LED AP", "#f4a261", "circle", "Nguồn phát quang đặt trên trần phòng", options.show_technical_details)
        add_marker(fig, pd_pos, "PD", "#2a9d8f", "diamond", "Bộ thu tại mặt phẳng người dùng", options.show_technical_details)
        add_marker(fig, ris, "Tâm RIS", "#4361ee", "square", "Vị trí RIS tối ưu theo data rate", options.show_technical_details)

    fig.update_layout(
        title=dict(text="Mô hình phòng 3D tương tác", x=0.02, xanchor="left"),
        scene=dict(
            xaxis=dict(title="x (m)", range=[0, config.room_length], backgroundcolor="#f8fafc"),
            yaxis=dict(title="y (m)", range=[0, config.room_width], backgroundcolor="#f8fafc"),
            zaxis=dict(title="z (m)", range=[0, config.room_height], backgroundcolor="#ffffff"),
            aspectmode="data",
            camera=camera_for_view(options.view),
            dragmode="orbit",
        ),
        height=620,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, x=0.0, bgcolor="rgba(255,255,255,0.82)"),
        uirevision="room-geometry",
    )
    return style_figure(fig)


def is_los_blocked_in_room_view(config: SimulationConfig, options: RoomFigureOptions) -> bool:
    if not options.show_obstacle:
        return False
    return blockage_factor(config.ap_position, config.pd_position, config) <= 0.0


def add_room_surfaces(fig: go.Figure, config: SimulationConfig, technical_details: bool) -> None:
    x0, x1 = 0.0, config.room_length
    y0, y1 = 0.0, config.room_width
    z0, z1 = 0.0, config.room_height

    surfaces = [
        (
            "Sàn phòng",
            np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),
            "#dbeafe",
            0.28,
            "Mặt sàn 5 m x 5 m",
        ),
        (
            "Tường đặt RIS",
            np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
            "#bfdbfe",
            0.18,
            "Mặt phẳng quét RIS: y = 0",
        ),
    ]
    for name, vertices, color, opacity, hover in surfaces:
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=opacity,
                name=name,
                hovertemplate=f"{hover}<extra></extra>" if technical_details else f"{name}<extra></extra>",
                showlegend=True,
            )
        )


def add_user_plane(fig: go.Figure, config: SimulationConfig, technical_details: bool) -> None:
    z = float(config.user_plane_z)
    vertices = np.array(
        [
            [0.0, 0.0, z],
            [config.room_length, 0.0, z],
            [config.room_length, config.room_width, z],
            [0.0, config.room_width, z],
        ]
    )
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#99f6e4",
            opacity=0.16,
            name="Mặt phẳng PD",
            hovertemplate=(
                f"Mặt phẳng người dùng<br>z = {z:.2f} m<extra></extra>"
                if technical_details
                else "Mặt phẳng PD<extra></extra>"
            ),
            showlegend=True,
        )
    )


def add_room_edges(fig: go.Figure, config: SimulationConfig) -> None:
    x0, x1 = 0.0, config.room_length
    y0, y1 = 0.0, config.room_width
    z0, z1 = 0.0, config.room_height
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for idx, (start, end) in enumerate(edges):
        points = corners[[start, end]]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color="#64748b", width=3),
                name="Khung phòng" if idx == 0 else None,
                showlegend=idx == 0,
                hoverinfo="skip",
            )
        )


def add_ris_panel(fig: go.Figure, center: np.ndarray, effective_area: float, technical_details: bool) -> None:
    panel_width = min(1.25, max(0.55, np.sqrt(effective_area) * 0.55))
    panel_height = min(1.15, max(0.45, np.sqrt(effective_area) * 0.5))
    x0, x1 = center[0] - panel_width / 2.0, center[0] + panel_width / 2.0
    z0, z1 = center[2] - panel_height / 2.0, center[2] + panel_height / 2.0
    y = center[1] + 0.015
    vertices = np.array(
        [
            [x0, y, z0],
            [x1, y, z0],
            [x1, y, z1],
            [x0, y, z1],
        ]
    )
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#4361ee",
            opacity=0.54,
            name="Bề mặt RIS",
            hovertemplate=(
                "Bề mặt RIS<br>"
                f"Kích thước hiển thị: {panel_width:.2f} m x {panel_height:.2f} m<br>"
                "Tường y = 0<br>Pháp tuyến: +y<extra></extra>"
                if technical_details
                else "Bề mặt RIS<extra></extra>"
            ),
        )
    )
    add_rectangle_edges(fig, vertices, "#1d4ed8", "Viền RIS", 5)
    normal_end = center + np.array([0.0, 0.56, 0.0])
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], normal_end[0]],
            y=[center[1], normal_end[1]],
            z=[center[2], normal_end[2]],
            mode="lines+markers",
            line=dict(color="#1d4ed8", width=6),
            marker=dict(size=[1, 5], color="#1d4ed8"),
            name="Pháp tuyến RIS +y",
            hovertemplate=(
                "Hướng phản xạ chuẩn của RIS<br>Pháp tuyến +y<extra></extra>"
                if technical_details
                else "Pháp tuyến RIS +y<extra></extra>"
            ),
            showlegend=True,
        )
    )


def add_obstacle_mesh(fig: go.Figure, config: SimulationConfig, technical_details: bool) -> None:
    mn = np.array(config.obstacle_min, dtype=float)
    mx = np.array(config.obstacle_max, dtype=float)
    vertices, triangles = box_vertices_and_triangles(mn, mx)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color="#9b2226",
            opacity=0.42,
            name="Vật cản",
            hovertemplate=(
                "Vật cản<br>"
                f"x: {mn[0]:.2f} - {mx[0]:.2f} m<br>"
                f"y: {mn[1]:.2f} - {mx[1]:.2f} m<br>"
                f"z: {mn[2]:.2f} - {mx[2]:.2f} m<extra></extra>"
                if technical_details
                else "Vật cản<extra></extra>"
            ),
        )
    )


def add_obstacle_edges(fig: go.Figure, config: SimulationConfig) -> None:
    vertices, _ = box_vertices_and_triangles(
        np.array(config.obstacle_min, dtype=float),
        np.array(config.obstacle_max, dtype=float),
    )
    add_box_edges(fig, vertices, "#7f1d1d", "Viền vật cản", 4)


def box_vertices_and_triangles(mn: np.ndarray, mx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ]
    )
    return vertices, triangles


def add_rectangle_edges(fig: go.Figure, vertices: np.ndarray, color: str, name: str, width: int) -> None:
    edge_order = [0, 1, 2, 3, 0]
    points = vertices[edge_order]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            hoverinfo="skip",
            showlegend=False,
        )
    )


def add_box_edges(fig: go.Figure, vertices: np.ndarray, color: str, name: str, width: int) -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for idx, (start, end) in enumerate(edges):
        points = vertices[[start, end]]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color=color, width=width),
                name=name if idx == 0 else None,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_marker(
    fig: go.Figure,
    point: np.ndarray,
    name: str,
    color: str,
    symbol: str,
    note: str,
    technical_details: bool,
) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers+text",
            marker=dict(size=9, color=color, symbol=symbol, line=dict(color="#ffffff", width=2)),
            text=[name],
            textposition="top center",
            name=name,
            hovertemplate=(
                f"{name}<br>{note}<br>"
                "x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>"
                if technical_details
                else f"{name}<extra></extra>"
            ),
        )
    )


def add_path(
    fig: go.Figure,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    name: str,
    note: str,
    dash: str = "solid",
    technical_details: bool = True,
) -> None:
    distance = float(np.linalg.norm(end - start))
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=7, dash=dash),
            name=name,
            hovertemplate=(
                f"{name}<br>{note}<br>Chiều dài: {distance:.2f} m<extra></extra>"
                if technical_details
                else f"{name}<extra></extra>"
            ),
        )
    )


def add_ris_path_segments(fig: go.Figure, ap: np.ndarray, ris: np.ndarray, pd_pos: np.ndarray, technical_details: bool) -> None:
    segments = [
        ("AP → RIS", ap, ris, "#0ea5e9"),
        ("RIS → PD", ris, pd_pos, "#0369a1"),
    ]
    total_distance = float(np.linalg.norm(ris - ap) + np.linalg.norm(pd_pos - ris))
    for idx, (name, start, end, color) in enumerate(segments):
        segment_distance = float(np.linalg.norm(end - start))
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=color, width=8),
                name=name,
                legendgroup="ris-path",
                showlegend=True,
                hovertemplate=(
                    f"{name}<br>"
                    f"Đoạn: {segment_distance:.2f} m<br>"
                    f"Tổng AP-RIS-PD: {total_distance:.2f} m<extra></extra>"
                    if technical_details
                    else f"{name}<extra></extra>"
                ),
            )
        )
        if idx == 0:
            fig.add_trace(
                go.Scatter3d(
                    x=[ris[0]],
                    y=[ris[1]],
                    z=[ris[2]],
                    mode="markers",
                    marker=dict(size=5, color="#0f172a"),
                    name="Điểm phản xạ RIS",
                    legendgroup="ris-path",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family=APPLE_FONT_FAMILY, size=13, color="#17212b"),
        title_font=dict(color="#0f172a", size=17),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font_color="#ffffff"),
        legend=dict(font=dict(color="#17212b"), bgcolor="rgba(255,255,255,0.65)"),
        title=dict(x=0.02, xanchor="left"),
    )
    fig.update_xaxes(
        title_font=dict(color="#344054"),
        tickfont=dict(color="#344054"),
        gridcolor="#e2e8f0",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
    )
    fig.update_yaxes(
        title_font=dict(color="#344054"),
        tickfont=dict(color="#344054"),
        gridcolor="#e2e8f0",
        zerolinecolor="#cbd5e1",
        linecolor="#cbd5e1",
    )
    if "scene" in fig.layout:
        fig.update_scenes(
            xaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#e2e8f0"),
            yaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#e2e8f0"),
            zaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#e2e8f0"),
        )
    return fig


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-color: #17212b;
            --background-color: #f5f7fb;
            --secondary-background-color: #ffffff;
            --primary-color: #2563eb;
            --success-color: #0f766e;
            --warning-color: #b45309;
            --risk-color: #b42318;
            --muted-color: #64748b;
            --border-color: #e2e8f0;
            --panel-shadow: 0 18px 42px rgba(15, 23, 42, 0.07);
        }
        .block-container {
            padding-top: 0.15rem !important;
            padding-bottom: 2.75rem !important;
            max-width: 1480px;
        }
        header[data-testid="stHeader"] {
            background: rgba(244,247,251,0.9) !important;
            backdrop-filter: blur(16px);
            height: 2.65rem !important;
        }
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stActionButton"],
        [data-testid="manage-app-button"],
        a[href*="github.com"][target="_blank"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        [data-testid="stToolbar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarExpandButton"],
        button[kind="header"],
        button[aria-label*="sidebar" i],
        button[title*="sidebar" i] {
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: #17212b !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebar"] [data-testid="collapsedControl"],
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] button[aria-label*="close sidebar" i],
        [data-testid="stSidebar"] button[title*="close sidebar" i] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
        .stApp {
            background: #f5f7fb;
            color: #17212b;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
        }
        .stApp,
        .stApp p,
        .stApp label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] {
            color: #17212b !important;
        }
        .stApp p,
        .stApp label,
        .stApp button,
        .stApp input,
        .stApp textarea,
        [data-testid="stMarkdownContainer"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.96);
            border-right: 1px solid var(--border-color);
            min-width: 20rem;
            width: 20rem;
            z-index: 999998;
        }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"] {
            background: rgba(255,255,255,0.98);
            min-width: 20rem;
            width: 20rem;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.25rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stWidgetLabel"] {
            color: #17212b !important;
            opacity: 1 !important;
        }
        [data-testid="stHeader"] {
            background: #f5f7fb;
        }
        .sidebar-title {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.055);
            padding: 1.08rem 1rem 1.12rem 1rem;
            margin: -0.35rem 0 0.95rem 0;
        }
        .sidebar-title span {
            color: #2563eb !important;
            display: block;
            font-size: 0.84rem;
            font-weight: 820;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.34rem;
        }
        .sidebar-title h1 {
            color: #0f172a !important;
            font-size: 1.55rem;
            line-height: 1.08;
            font-weight: 840;
            margin: 0 !important;
            white-space: nowrap;
        }
        .team-card {
            background: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            margin: 0.4rem 0 0.9rem 0;
        }
        .team-card div + div {
            border-top: 1px solid #e5edf7;
            margin-top: 0.75rem;
            padding-top: 0.75rem;
        }
        .team-card span {
            color: #64748b !important;
            display: block;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-bottom: 0.25rem;
        }
        .team-card strong {
            color: #0f172a !important;
            display: block;
            font-size: 0.92rem;
            line-height: 1.35;
            font-weight: 720;
        }
        .control-heading {
            margin: 0.25rem 0 0.7rem 0;
            padding: 0.78rem 0.86rem;
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.045);
        }
        .control-heading span {
            color: #64748b !important;
            display: block;
            font-size: 0.75rem;
            font-weight: 760;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .control-heading strong {
            color: #0f172a !important;
            display: block;
            font-size: 1.15rem;
            font-weight: 820;
            line-height: 1.25;
        }
        .control-heading p {
            color: #64748b !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.3rem 0 0 0;
        }
        [data-testid="stExpander"] {
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 8px !important;
            background: #ffffff !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.035) !important;
            margin-bottom: 0.65rem !important;
        }
        [data-testid="stExpander"] summary {
            color: #0f172a !important;
            font-weight: 780 !important;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
            color: #64748b !important;
            font-size: 0.82rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 96px;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #17212b !important;
            opacity: 1 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 750;
        }
        h1, h2, h3 {
            color: #17212b !important;
            letter-spacing: 0;
        }
        h1 {
            margin-top: 0 !important;
            padding-top: 0 !important;
            margin-bottom: 0.35rem !important;
        }
        .hero {
            border-bottom: 1px solid var(--border-color);
            overflow: visible;
            padding: 0 0 0.85rem 0;
            margin-bottom: 0.65rem;
        }
        .hero .eyebrow {
            color: #2563eb !important;
            font-size: 0.78rem;
            font-weight: 780;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .hero h1 {
            color: #0f172a !important;
            font-size: 2.35rem;
            line-height: 1.28;
            font-weight: 780;
            margin: 0;
            max-width: 980px;
            overflow: visible;
            padding: 0.35rem 0 0.08rem 0 !important;
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.82rem;
        }
        .hero-meta span {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            color: #334155 !important;
            display: inline-flex;
            font-size: 0.86rem;
            font-weight: 720;
            line-height: 1.2;
            padding: 0.45rem 0.62rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.95rem 0 0.85rem 0;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            min-height: 112px;
            padding: 0.86rem 0.92rem;
        }
        .metric-card span {
            color: #64748b !important;
            display: block;
            font-size: 0.82rem;
            font-weight: 720;
            line-height: 1.25;
            margin-bottom: 0.52rem;
        }
        .metric-card strong {
            color: #0f172a !important;
            display: block;
            font-size: 1.55rem;
            font-weight: 790;
            line-height: 1.05;
            margin-bottom: 0.48rem;
        }
        .metric-card small {
            color: #64748b !important;
            display: block;
            font-size: 0.8rem;
            line-height: 1.3;
        }
        .metric-rate {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        }
        .metric-snr {
            background: linear-gradient(180deg, #ffffff 0%, #f7fdfb 100%);
        }
        .metric-baseline {
            background: linear-gradient(180deg, #ffffff 0%, #fbf9ff 100%);
        }
        .metric-risk {
            background: linear-gradient(180deg, #ffffff 0%, #fffafa 100%);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            box-shadow: var(--panel-shadow);
            padding: 1rem 1.05rem 0.9rem 1.05rem;
            margin: 0.7rem 0 0.85rem 0;
        }
        .section-heading {
            border-bottom: 1px solid #e8eef6;
            padding-bottom: 0.7rem;
            margin-bottom: 0.78rem;
        }
        .section-heading h2 {
            color: #0f172a !important;
            font-size: 1.18rem;
            line-height: 1.2;
            font-weight: 760;
            margin: 0;
        }
        .section-heading p {
            color: #64748b !important;
            font-size: 0.94rem;
            line-height: 1.45;
            margin: 0.35rem 0 0 0;
        }
        .room-summary {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.35rem 0 0.9rem 0;
        }
        .room-summary div {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            min-height: 66px;
            padding: 0.65rem 0.72rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.035);
        }
        .room-summary span {
            color: #64748b !important;
            display: block;
            font-size: 0.76rem;
            font-weight: 760;
            line-height: 1.2;
            margin-bottom: 0.3rem;
        }
        .room-summary strong {
            color: #0f172a !important;
            display: block;
            font-size: 0.92rem;
            font-weight: 780;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .stRadio label,
        .stSlider label,
        .stSlider [data-testid="stMarkdownContainer"] p {
            color: #17212b !important;
            opacity: 1 !important;
        }
        button[data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: #334155 !important;
            font-weight: 720;
            height: 2.5rem;
            margin-right: 0.35rem;
            padding: 0 1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            border-color: rgba(37, 99, 235, 0.28);
            color: #1d4ed8 !important;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.12);
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin: 0.3rem 0 0.45rem 0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }
        .explain-box {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            padding: 11px 14px;
            margin: 0 0 12px 0;
            color: #17212b;
            line-height: 1.45;
            font-size: 0.94rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.035);
        }
        .explain-box code {
            color: #1d4ed8;
            background: #eff6ff;
            border-radius: 4px;
            padding: 1px 4px;
        }
        .stDownloadButton button {
            border-radius: 8px;
            border-color: #cbd5e1;
            color: #0f172a;
            font-weight: 700;
        }
        @media (max-width: 1100px) {
            .hero h1 {
                font-size: 2.15rem;
            }
            .metric-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
            .room-summary {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 720px) {
            .block-container {
                padding-left: 0.9rem !important;
                padding-right: 0.9rem !important;
            }
            .hero h1 {
                font-size: 1.85rem;
            }
            .hero-meta {
                display: grid;
                grid-template-columns: 1fr;
            }
            .metric-grid {
                grid-template-columns: 1fr;
            }
            .room-summary {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
