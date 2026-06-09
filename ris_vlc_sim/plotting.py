import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .utils import db_for_plot


def _save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_snr_comparison(scenario_df, output_path):
    labels = [f"S{i + 1}" for i in range(len(scenario_df))]
    values = db_for_plot(scenario_df["SNR_dB"].to_numpy(), floor_db=-60.0)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#2f6f91", "#9b2226", "#3973ac", "#8a5a44"])
    plt.axhline(10.0, color="#333333", linestyle="--", linewidth=1.2, label="Threshold 10 dB")
    plt.ylabel("SNR (dB)")
    plt.xlabel("Scenario")
    plt.title("SNR comparison between VLC scenarios")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    for bar, original in zip(bars, scenario_df["SNR_dB"]):
        label = "-inf" if not np.isfinite(original) else f"{original:.1f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")

    _save_current_figure(output_path)


def plot_rate_comparison(scenario_df, output_path):
    labels = [f"S{i + 1}" for i in range(len(scenario_df))]
    values = scenario_df["data_rate_Mbps"].to_numpy()

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#2f6f91", "#9b2226", "#3973ac", "#8a5a44"])
    plt.ylabel("Data rate (Mbps)")
    plt.xlabel("Scenario")
    plt.title("Data rate comparison between VLC scenarios")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom")

    _save_current_figure(output_path)


def plot_ris_optimization_3d(optimization_df, best_row, output_path):
    pivot = optimization_df.pivot(index="z_RIS_m", columns="x_RIS_m", values="data_rate_Mbps")
    x_grid, z_grid = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
    rate_grid = pivot.to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x_grid, z_grid, rate_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.scatter(
        [best_row["x_RIS_m"]],
        [best_row["z_RIS_m"]],
        [best_row["data_rate_Mbps"]],
        color="#d00000",
        s=70,
        marker="o",
        label="Optimal RIS position",
    )
    ax.text(
        best_row["x_RIS_m"],
        best_row["z_RIS_m"],
        best_row["data_rate_Mbps"],
        f"  x={best_row['x_RIS_m']:.2f}, z={best_row['z_RIS_m']:.2f}\n  R={best_row['data_rate_Mbps']:.2f} Mbps",
        color="#111111",
    )
    ax.set_xlabel("RIS x position on wall y=0 (m)")
    ax.set_ylabel("RIS z position (m)")
    ax.set_zlabel("Data rate (Mbps)")
    ax.set_title("RIS position optimization")
    ax.legend()
    fig.colorbar(surface, shrink=0.6, aspect=12, label="Data rate (Mbps)")

    _save_current_figure(output_path)


def plot_ris_optimization_heatmap(optimization_df, best_row, output_path):
    pivot = optimization_df.pivot(index="z_RIS_m", columns="x_RIS_m", values="data_rate_Mbps")
    x_values = pivot.columns.to_numpy(dtype=float)
    z_values = pivot.index.to_numpy(dtype=float)

    plt.figure(figsize=(9, 6))
    plt.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        extent=[x_values.min(), x_values.max(), z_values.min(), z_values.max()],
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="Data rate (Mbps)")
    plt.scatter(best_row["x_RIS_m"], best_row["z_RIS_m"], color="#d00000", marker="x", s=90, linewidths=2.5)
    plt.text(
        best_row["x_RIS_m"],
        best_row["z_RIS_m"],
        f"  opt: ({best_row['x_RIS_m']:.2f}, {best_row['z_RIS_m']:.2f})",
        color="white",
        weight="bold",
    )
    plt.xlabel("RIS x position on wall y=0 (m)")
    plt.ylabel("RIS z position (m)")
    plt.title("Data rate heatmap for RIS position sweep")
    _save_current_figure(output_path)


def plot_snr_heatmap(x_values, y_values, snr_db_grid, output_path, title, floor_db):
    display_grid = db_for_plot(snr_db_grid, floor_db=floor_db)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        display_grid,
        origin="lower",
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        aspect="equal",
        cmap="magma",
        vmin=floor_db,
    )
    plt.colorbar(label="SNR (dB)")
    plt.xlabel("PD x position (m)")
    plt.ylabel("PD y position (m)")
    plt.title(title)
    _save_current_figure(output_path)


def plot_system_geometry_3d(config, ris_position, output_path):
    ap = np.array(config.ap_position, dtype=float)
    pd = np.array(config.pd_position, dtype=float)
    ris = np.array(ris_position, dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

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
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        ax.plot(*zip(corners[start], corners[end]), color="#666666", linewidth=0.9, alpha=0.6)

    obstacle_min = np.array(config.obstacle_min, dtype=float)
    obstacle_max = np.array(config.obstacle_max, dtype=float)
    obstacle_x = np.array([obstacle_min[0], obstacle_max[0], obstacle_max[0], obstacle_min[0]])
    obstacle_y = np.array([obstacle_min[1], obstacle_min[1], obstacle_max[1], obstacle_max[1]])
    obstacle_z = np.array([obstacle_min[2], obstacle_min[2], obstacle_max[2], obstacle_max[2]])
    ax.plot_trisurf(obstacle_x, obstacle_y, obstacle_z, color="#9b2226", alpha=0.42)

    ax.scatter(*ap, color="#f4a261", s=120, marker="o", label="AP LED")
    ax.scatter(*pd, color="#2a9d8f", s=110, marker="^", label="PD")
    ax.scatter(*ris, color="#4361ee", s=130, marker="s", label="RIS")

    ax.plot(
        [ap[0], pd[0]],
        [ap[1], pd[1]],
        [ap[2], pd[2]],
        color="#d00000",
        linestyle="--",
        linewidth=2.0,
        label="Blocked LoS",
    )
    ax.plot(
        [ap[0], ris[0], pd[0]],
        [ap[1], ris[1], pd[1]],
        [ap[2], ris[2], pd[2]],
        color="#0077b6",
        linewidth=2.5,
        label="AP-RIS-PD path",
    )

    ax.text(*ap, " AP", color="#111111")
    ax.text(*pd, " PD", color="#111111")
    ax.text(*ris, " RIS", color="#111111")
    ax.text(2.9, 1.7, 2.15, "Obstacle", color="#9b2226")

    ax.set_xlim(0, config.room_length)
    ax.set_ylim(0, config.room_width)
    ax.set_zlim(0, config.room_height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("RIS-assisted indoor VLC geometry")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    _save_current_figure(output_path)


def plot_room_top_view(config, ris_position, output_path):
    ap = np.array(config.ap_position, dtype=float)
    pd = np.array(config.pd_position, dtype=float)
    ris = np.array(ris_position, dtype=float)
    obstacle_min = np.array(config.obstacle_min, dtype=float)
    obstacle_max = np.array(config.obstacle_max, dtype=float)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    room = plt.Rectangle((0, 0), config.room_length, config.room_width, fill=False, linewidth=2.0, color="#333333")
    obstacle = plt.Rectangle(
        (obstacle_min[0], obstacle_min[1]),
        obstacle_max[0] - obstacle_min[0],
        obstacle_max[1] - obstacle_min[1],
        color="#9b2226",
        alpha=0.45,
        label="Obstacle footprint",
    )
    ax.add_patch(room)
    ax.add_patch(obstacle)

    plt.scatter(ap[0], ap[1], s=130, color="#f4a261", label="AP LED")
    plt.scatter(pd[0], pd[1], s=120, color="#2a9d8f", marker="^", label="PD")
    plt.scatter(ris[0], ris[1], s=130, color="#4361ee", marker="s", label="RIS")

    plt.plot([ap[0], pd[0]], [ap[1], pd[1]], color="#d00000", linestyle="--", linewidth=2.0, label="Blocked LoS")
    plt.plot([ap[0], ris[0], pd[0]], [ap[1], ris[1], pd[1]], color="#0077b6", linewidth=2.5, label="AP-RIS-PD")

    plt.text(ap[0] + 0.08, ap[1] + 0.08, "AP")
    plt.text(pd[0] + 0.08, pd[1] + 0.08, "PD")
    plt.text(ris[0] + 0.08, ris[1] + 0.08, "RIS")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Top view of indoor VLC scenario")
    plt.xlim(0, config.room_length)
    plt.ylim(0, config.room_width)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    _save_current_figure(output_path)
