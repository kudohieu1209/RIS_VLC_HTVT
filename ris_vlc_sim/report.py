from pathlib import Path


def write_summary_report(config, scenario_df, best_row, output_path: Path) -> None:
    obstacle_without_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 2: Obstacle, without RIS",
        "data_rate_Mbps",
    ].iloc[0]
    obstacle_with_ris = scenario_df.loc[
        scenario_df["scenario"] == "Scenario 3: Obstacle, with RIS",
        "data_rate_Mbps",
    ].iloc[0]
    delta_rate_mbps = obstacle_with_ris - obstacle_without_ris

    lines = [
        "# RIS-assisted VLC Simulation Summary",
        "",
        "## Geometry",
        "",
        f"- Room size: {config.room_length} m x {config.room_width} m x {config.room_height} m",
        f"- AP LED position: {config.ap_position} m",
        f"- Default PD position: {config.pd_position} m",
        f"- RIS wall: y = {config.ris_wall_y} m",
        f"- Obstacle min corner: {config.obstacle_min} m",
        f"- Obstacle max corner: {config.obstacle_max} m",
        "",
        "## Optimal RIS Position",
        "",
        f"- x_opt: {best_row['x_RIS_m']:.3f} m",
        f"- z_opt: {best_row['z_RIS_m']:.3f} m",
        f"- max SNR: {best_row['SNR_dB']:.3f} dB",
        f"- max data rate: {best_row['data_rate_Mbps']:.3f} Mbps",
        "",
        "## Scenario Results",
        "",
        "| Scenario | LoS blocked | SNR (dB) | Data rate (Mbps) |",
        "|---|---:|---:|---:|",
    ]

    for _, row in scenario_df.iterrows():
        snr_text = "-inf" if row["SNR_linear"] <= 0.0 else f"{row['SNR_dB']:.3f}"
        lines.append(
            f"| {row['scenario']} | {row['LoS_blocked']} | {snr_text} | {row['data_rate_Mbps']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Improvement",
            "",
            f"- Delta rate in obstacle case: {delta_rate_mbps:.3f} Mbps",
            "",
            "The obstacle is modeled as a 3D rectangular box. For each PD point, the simulation checks whether the AP-PD LoS segment intersects the box. If it intersects, the LoS component is blocked; otherwise the LoS component remains available.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
