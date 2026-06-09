from __future__ import annotations

from dataclasses import replace
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ris_vlc_sim.config import SimulationConfig, validate_config
from ris_vlc_sim.simulation import run_pd_snr_grid, run_ris_position_optimization, run_scenarios
from ris_vlc_sim.utils import db_for_plot


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


def main() -> None:
    apply_theme()
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">RIS / VLC Indoor Link Simulator</div>
            <h1>Mô phỏng VLC hỗ trợ RIS</h1>
            <p>Điều chỉnh tham số, quan sát tác động của vật cản và tìm vị trí RIS tối ưu theo data rate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    render_metrics(scenario_df, best_row)

    left, right = st.columns((1.05, 1.25), gap="large")
    with left:
        with st.container(border=True):
            render_section_header(
                "So sánh các kịch bản truyền dẫn",
                "Bảng và biểu đồ thể hiện ảnh hưởng của vật cản đến đường LoS và mức cải thiện khi bổ sung đường phản xạ qua RIS.",
            )
            render_scenario_table(scenario_df)
            st.plotly_chart(make_rate_bar(scenario_df), use_container_width=True)

    with right:
        with st.container(border=True):
            render_optimization_panel(optimization_df, best_row)

    mid_left, mid_right = st.columns((1.0, 1.15), gap="large")
    with mid_left:
        with st.container(border=True):
            render_section_header(
                "Ngưỡng chất lượng liên kết",
                "SNR được đối chiếu với ngưỡng tham chiếu 10 dB để đánh giá khả năng duy trì liên kết tin cậy.",
            )
            st.plotly_chart(make_snr_bar(scenario_df), use_container_width=True)
    with mid_right:
        with st.container(border=True):
            render_section_header(
                "Hình học phòng 3D",
                "Mô hình không gian biểu diễn vị trí AP, PD, vật cản, RIS và hai tuyến truyền LoS/AP-RIS-PD.",
            )
            st.plotly_chart(make_room_figure(config, best_ris_position), use_container_width=True)

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
            )

    with st.container(border=True):
        render_section_header(
            "Xuất dữ liệu",
            "Các bảng kết quả có thể tải về để đối chiếu số liệu, lập phụ lục hoặc tiếp tục xử lý bằng công cụ khác.",
        )
        render_downloads(scenario_df, optimization_df)


def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_config() -> SimulationConfig:
    base = SimulationConfig()
    with st.sidebar:
        st.title("Bảo vệ bài tập lớn hệ thống viễn thông")
        st.markdown(
            """
            **Giảng viên hướng dẫn**  
            Nguyễn Thành Chuyên

            **Sinh viên thực hiện**  
            Mai Duy Hiếu - 20223967  
            Nguyễn Kim Đạt - 20223762
            """
        )
        st.divider()
        st.header("Bảng điều khiển")
        resolution = (61, 61, 70)

        st.subheader("Bộ thu PD")
        pd_x = st.slider("Tọa độ PD x (m)", 0.1, base.room_length - 0.1, base.pd_position[0], 0.05)
        pd_y = st.slider("Tọa độ PD y (m)", 0.1, base.room_width - 0.1, base.pd_position[1], 0.05)
        pd_z = st.slider("Tọa độ PD z (m)", 0.2, base.room_height - 0.1, base.pd_position[2], 0.05)

        st.subheader("RIS")
        ris_area = st.slider("Diện tích (m²)", 0.2, 5.0, base.ris_effective_area, 0.1)
        reflection = st.slider("Hệ số phản xạ", 0.0, 1.0, base.ris_reflection_coefficient, 0.05)
        alignment = st.slider("Độ căn chỉnh", 0.0, 1.0, base.ris_alignment_gain, 0.05)

        st.subheader("Liên kết quang")
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mức tăng data rate khi có RIS", f"{delta_rate:.2f} Mbps")
    col2.metric("Data rate tại vị trí RIS tối ưu", f"{best_row['data_rate_Mbps']:.2f} Mbps")
    col3.metric("SNR tại vị trí RIS tối ưu", f"{best_row['SNR_dB']:.2f} dB")
    col4.metric("Data rate LoS chuẩn", f"{baseline['data_rate_Mbps']:.2f} Mbps")


def render_scenario_table(scenario_df: pd.DataFrame) -> None:
    display_df = scenario_df.copy()
    display_df["scenario"] = display_df["scenario"].map(SCENARIO_LABELS)
    display_df = display_df[
        [
            "scenario",
            "LoS_blocked",
            "H_LoS",
            "H_RIS",
            "SNR_dB",
            "data_rate_Mbps",
        ]
    ].rename(
        columns={
            "scenario": "Kịch bản",
            "LoS_blocked": "Bị chắn",
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
            Thang màu biểu diễn data rate thu được; vùng màu sáng cho thấy vị trí đặt RIS hiệu quả hơn.
            Với cấu hình hiện tại, vị trí tối ưu là
            <b>x={best_row['x_RIS_m']:.2f} m</b>, <b>z={best_row['z_RIS_m']:.2f} m</b>,
            đạt <b>{best_row['data_rate_Mbps']:.2f} Mbps</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_optimization_heatmap(optimization_df, best_row), use_container_width=True)


def make_rate_bar(scenario_df: pd.DataFrame) -> go.Figure:
    df = scenario_df.copy()
    df["label"] = df["scenario"].map(SCENARIO_LABELS)
    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["data_rate_Mbps"],
            marker_color=["#2f6f91", "#9b2226", "#3973ac", "#8a5a44"],
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
            marker_color=["#2f6f91", "#9b2226", "#3973ac", "#8a5a44"],
            text=["-inf" if not np.isfinite(value) else f"{value:.1f}" for value in df["SNR_dB"]],
            textposition="outside",
            hovertemplate="%{x}<br>SNR=%{text} dB<extra></extra>",
        )
    )
    fig.add_hline(y=10.0, line_dash="dash", line_color="#444", annotation_text="10 dB")
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
            colorscale="Viridis",
            colorbar=dict(title="Data rate<br>(Mbps)"),
            hovertemplate="Đặt RIS tại:<br>x=%{x:.2f} m<br>z=%{y:.2f} m<br>Data rate=%{z:.3f} Mbps<extra></extra>",
        )
    )
    fig.add_vline(
        x=best_x,
        line_width=2,
        line_dash="dot",
        line_color="#ff3131",
    )
    fig.add_hline(
        y=best_z,
        line_width=2,
        line_dash="dot",
        line_color="#ff3131",
    )
    fig.add_trace(
        go.Scatter(
            x=[best_x],
            y=[best_z],
            mode="markers+text",
            marker=dict(color="#ff3131", size=13, symbol="x"),
            text=[f"vị trí tối ưu<br>{best_rate:.1f} Mbps"],
            textposition="top center",
            hovertemplate="Tối ưu x=%{x:.2f} m<br>Tối ưu z=%{y:.2f} m<extra></extra>",
        )
    )
    fig.update_layout(
        title="Phân bố data rate theo vị trí RIS",
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
        text="Màu tối: data rate thấp<br>Màu sáng: data rate cao",
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
            colorscale="Magma",
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


def make_room_figure(config: SimulationConfig, ris_position: tuple[float, float, float]) -> go.Figure:
    ap = np.array(config.ap_position, dtype=float)
    pd_pos = np.array(config.pd_position, dtype=float)
    ris = np.array(ris_position, dtype=float)
    fig = go.Figure()

    add_room_edges(fig, config)
    add_obstacle_mesh(fig, config)
    add_path(fig, ap, pd_pos, "#d00000", "LoS bị chắn")
    add_polyline(fig, np.vstack([ap, ris, pd_pos]), "#0077b6", "AP-RIS-PD")
    add_marker(fig, ap, "LED AP", "#f4a261", "circle")
    add_marker(fig, pd_pos, "PD", "#2a9d8f", "diamond")
    add_marker(fig, ris, "RIS", "#4361ee", "square")

    fig.update_layout(
        title="Mô hình phòng 3D tương tác",
        scene=dict(
            xaxis=dict(title="x (m)", range=[0, config.room_length]),
            yaxis=dict(title="y (m)", range=[0, config.room_width]),
            zaxis=dict(title="z (m)", range=[0, config.room_height]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=-1.75, z=1.05)),
        ),
        height=420,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.0),
    )
    return style_figure(fig)


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
                line=dict(color="#777", width=3),
                name="Phòng" if idx == 0 else None,
                showlegend=idx == 0,
                hoverinfo="skip",
            )
        )


def add_obstacle_mesh(fig: go.Figure, config: SimulationConfig) -> None:
    mn = np.array(config.obstacle_min, dtype=float)
    mx = np.array(config.obstacle_max, dtype=float)
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
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            color="#9b2226",
            opacity=0.38,
            name="Vật cản",
            hovertemplate="Vật cản<extra></extra>",
        )
    )


def add_marker(fig: go.Figure, point: np.ndarray, name: str, color: str, symbol: str) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol=symbol),
            text=[name],
            textposition="top center",
            name=name,
            hovertemplate=f"{name}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m<br>z=%{{z:.2f}} m<extra></extra>",
        )
    )


def add_path(fig: go.Figure, start: np.ndarray, end: np.ndarray, color: str, name: str) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=5),
            name=name,
            hovertemplate=f"{name}<extra></extra>",
        )
    )


def add_polyline(fig: go.Figure, points: np.ndarray, color: str, name: str) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            line=dict(color=color, width=6),
            name=name,
            hovertemplate=f"{name}<extra></extra>",
        )
    )


def render_downloads(scenario_df: pd.DataFrame, optimization_df: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    col1.download_button(
        "Tải CSV scenario",
        data=to_csv(scenario_df),
        file_name="scenario_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "Tải CSV quét RIS",
        data=to_csv(optimization_df),
        file_name="ris_position_optimization.csv",
        mime="text/csv",
        use_container_width=True,
    )


def to_csv(df: pd.DataFrame) -> str:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Arial", size=13, color="#17212b"),
        title_font=dict(color="#17212b", size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfd",
        hoverlabel=dict(bgcolor="#17212b", font_color="#ffffff"),
        legend=dict(font=dict(color="#17212b")),
    )
    fig.update_xaxes(
        title_font=dict(color="#344054"),
        tickfont=dict(color="#344054"),
        gridcolor="#dde5ef",
        zerolinecolor="#c8d3df",
    )
    fig.update_yaxes(
        title_font=dict(color="#344054"),
        tickfont=dict(color="#344054"),
        gridcolor="#dde5ef",
        zerolinecolor="#c8d3df",
    )
    if "scene" in fig.layout:
        fig.update_scenes(
            xaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#dde5ef"),
            yaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#dde5ef"),
            zaxis=dict(title_font=dict(color="#344054"), tickfont=dict(color="#344054"), gridcolor="#dde5ef"),
        )
    return fig


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-color: #17212b;
            --background-color: #f6f8fb;
            --secondary-background-color: #ffffff;
            --primary-color: #e63946;
        }
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1500px;
        }
        header[data-testid="stHeader"] {
            background: rgba(245,245,247,0.86) !important;
            backdrop-filter: blur(16px);
            height: 2.65rem !important;
        }
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        button[kind="header"] {
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            color: #17212b !important;
        }
        .stApp {
            background: #f5f5f7;
            color: #17212b;
        }
        .stApp, .stApp p, .stApp span, .stApp label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: #17212b !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.94);
            border-right: 1px solid #e5e5ea;
        }
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: translateX(0) !important;
            margin-left: 0 !important;
            min-width: 360px !important;
            width: 360px !important;
            max-width: 360px !important;
        }
        section[data-testid="stSidebar"] > div {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            width: 360px !important;
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
            background: #f6f8fb;
        }
        [data-testid="stToolbar"] {
            color: #17212b;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e5ea;
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 96px;
            box-shadow: 0 8px 28px rgba(17, 24, 39, 0.045);
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
            padding: 0.25rem 0 1.1rem 0;
        }
        .hero .eyebrow {
            color: #667085 !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .hero h1 {
            color: #111827 !important;
            font-size: clamp(2.1rem, 4vw, 3.35rem);
            line-height: 1.04;
            font-weight: 780;
            margin: 0;
        }
        .hero p {
            color: #667085 !important;
            max-width: 760px;
            font-size: 1.02rem;
            line-height: 1.45;
            margin: 0.65rem 0 0 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid #e5e5ea;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(17, 24, 39, 0.045);
            padding: 1.05rem 1.1rem 0.8rem 1.1rem;
            margin: 0.85rem 0;
        }
        .section-heading {
            border-bottom: 1px solid #eef1f5;
            padding-bottom: 0.75rem;
            margin-bottom: 0.85rem;
        }
        .section-heading h2 {
            color: #111827 !important;
            font-size: 1.18rem;
            line-height: 1.2;
            font-weight: 760;
            margin: 0;
        }
        .section-heading p {
            color: #667085 !important;
            font-size: 0.94rem;
            line-height: 1.45;
            margin: 0.35rem 0 0 0;
        }
        .stRadio label,
        .stSlider label,
        .stSlider [data-testid="stMarkdownContainer"] p {
            color: #17212b !important;
            opacity: 1 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e1e7ef;
            border-radius: 8px;
            overflow: hidden;
        }
        .explain-box {
            background: #f9fafb;
            border: 1px solid #eef1f5;
            border-radius: 8px;
            padding: 12px 14px;
            margin: 0 0 12px 0;
            color: #17212b;
            line-height: 1.45;
            font-size: 0.94rem;
        }
        .explain-box code {
            color: #9b2226;
            background: #f3f6fa;
            border-radius: 4px;
            padding: 1px 4px;
        }
        .stDownloadButton button {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
