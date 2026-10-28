#!/usr/bin/env python3
"""Combine the eight pure-water CNT structure/state analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT")
OUT = ROOT / "pure_water_phase_state_i6_i40"
OLD_RATIO_CSV = (
    ROOT / "pure_water_deficit_length_i6_i30"
    / "pure_water_Ccnt_Cbulk_ratio_i6_i30.csv"
)
RADIAL = OUT / "radial_preparation"
INDICES = (6, 8, 9, 10, 12, 22, 30, 40)
COLORS = (
    "#2878b5", "#e76f51", "#f4a261", "#d4a017",
    "#2a9d8f", "#7b2cbf", "#96345b", "#4d4d4d",
)
LAYER_REGIMES = {
    6: "Single-file",
    8: "Monolayer shell",
    9: "Monolayer shell",
    10: "Axial core + outer shell",
    12: "Two radial layers",
    22: "Multilayer",
    30: "Bulk-like core + wall layers",
    40: "Bulk-like core + wall layers",
}


def load_ratios() -> dict[int, dict[str, float]]:
    ratios: dict[int, dict[str, float]] = {}
    with OLD_RATIO_CSV.open(encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            index = int(source["CNT"].strip('()\"').split(",")[0])
            ratios[index] = {
                "diameter_nm": float(source["diameter_nm"]),
                "ratio": float(source["Cw_CNT_over_Cw_bulk"]),
                "ratio_sem": float(source["ratio_SEM"]),
            }
    for index in (9, 10, 40):
        source = json.loads(
            (RADIAL / f"cnt_{index}_{index}_radial_profile_300_500ns.json")
            .read_text()
        )
        ratios[index] = {
            "diameter_nm": float(source["diameter_nm"]),
            "ratio": float(source["ratio"]),
            "ratio_sem": float(source["ratio_SEM"]),
        }
    return ratios


def load_rows() -> list[dict[str, object]]:
    ratios = load_ratios()
    rows: list[dict[str, object]] = []
    for index in INDICES:
        summary = json.loads(
            (OUT / f"cnt_{index}_{index}" / "summary.json").read_text()
        )
        raw = summary["MSD"]["raw"]
        corrected = summary["MSD"]["collective_corrected"]
        state = (
            "Crystal/solid-like"
            if summary["binary_state_from_structure_factor"] == "Crystal-like"
            else "Liquid"
        )
        rows.append(
            {
                "index": index,
                "cnt": f"({index},{index})",
                **ratios[index],
                "layer_regime": LAYER_REGIMES[index],
                "selection": summary["layer_selection"]["kind"],
                "r_min_nm": float(summary["layer_selection"]["r_min_nm"]),
                "r_max_nm": float(summary["layer_selection"]["r_max_nm"]),
                "mean_selected_ow": float(summary["mean_selected_OW_per_frame"]),
                "s_max": float(summary["S_max"]),
                "q_peak": float(summary["q_peak_nm_inv"]),
                "peak_direction": summary["peak_direction"],
                "raw_d": float(raw["block_mean_D_z_1e-9_m2_per_s"]),
                "raw_d_sem": float(raw["block_sem_D_z_1e-9_m2_per_s"]),
                "corrected_d": float(
                    corrected["block_mean_D_z_1e-9_m2_per_s"]
                ),
                "corrected_d_sem": float(
                    corrected["block_sem_D_z_1e-9_m2_per_s"]
                ),
                "state": state,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> Path:
    path = OUT / "pure_water_liquid_crystal_state_i6_i40.csv"
    fields = [
        "CNT", "diameter_nm", "Cw_CNT_over_Cw_bulk", "ratio_SEM",
        "radial_layer_regime", "structure_selection", "r_min_nm", "r_max_nm",
        "mean_selected_OW_per_frame", "S_max", "q_peak_nm_inv",
        "peak_direction", "Hansen_Verlet_reference",
        "Dz_raw_1e-9_m2_per_s", "Dz_raw_SEM_1e-9_m2_per_s",
        "Dz_collective_corrected_1e-9_m2_per_s",
        "Dz_collective_corrected_SEM_1e-9_m2_per_s", "binary_water_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "CNT": row["cnt"],
                    "diameter_nm": f"{row['diameter_nm']:.9f}",
                    "Cw_CNT_over_Cw_bulk": f"{row['ratio']:.9f}",
                    "ratio_SEM": f"{row['ratio_sem']:.9f}",
                    "radial_layer_regime": row["layer_regime"],
                    "structure_selection": row["selection"],
                    "r_min_nm": f"{row['r_min_nm']:.3f}",
                    "r_max_nm": f"{row['r_max_nm']:.3f}",
                    "mean_selected_OW_per_frame": f"{row['mean_selected_ow']:.3f}",
                    "S_max": f"{row['s_max']:.6f}",
                    "q_peak_nm_inv": f"{row['q_peak']:.6f}",
                    "peak_direction": row["peak_direction"],
                    "Hansen_Verlet_reference": "2.85",
                    "Dz_raw_1e-9_m2_per_s": f"{row['raw_d']:.9f}",
                    "Dz_raw_SEM_1e-9_m2_per_s": f"{row['raw_d_sem']:.9f}",
                    "Dz_collective_corrected_1e-9_m2_per_s": f"{row['corrected_d']:.9f}",
                    "Dz_collective_corrected_SEM_1e-9_m2_per_s": f"{row['corrected_d_sem']:.9f}",
                    "binary_water_state": row["state"],
                }
            )
    return path


def plot_structure_factor(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.0))
    for ax, row, color in zip(axes.ravel(), rows, COLORS):
        index = int(row["index"])
        data = np.load(OUT / f"cnt_{index}_{index}" / "structure_factor_blocks.npz")
        qz, sz, sz_sem = data["q_z_nm_inv"], data["S_z_mean"], data["S_z_sem"]
        use_z = (qz >= 10.0) & (qz <= 50.0)
        ax.plot(qz[use_z], sz[use_z], color=color, lw=2.0)
        ax.fill_between(qz[use_z], sz[use_z]-sz_sem[use_z], sz[use_z]+sz_sem[use_z], color=color, alpha=.16, linewidth=0)
        qu = data["q_u_nm_inv"]
        if qu.size:
            su, su_sem = data["S_u_mean"], data["S_u_sem"]
            use_u = (qu >= 10.0) & (qu <= 50.0)
            ax.plot(qu[use_u], su[use_u], color="#333333", lw=1.4, ls="--")
            ax.fill_between(qu[use_u], su[use_u]-su_sem[use_u], su[use_u]+su_sem[use_u], color="#777777", alpha=.10, linewidth=0)
        ax.axhline(2.85, color="#b2182b", lw=1.2, ls=":")
        ax.scatter([row["q_peak"]], [row["s_max"]], color="#b2182b", s=24, zorder=4)
        ax.set_title(
            f"CNT{row['cnt']} — {row['state']}\n"
            f"$S_{{max}}={row['s_max']:.2f}$, {row['peak_direction']}",
            color=color, fontweight="bold", fontsize=10,
        )
        ax.set_xlim(10, 50)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$q$ (nm$^{-1}$)")
        ax.set_ylabel(r"$S(q)$")
        ax.grid(alpha=.2)
    fig.subplots_adjust(left=.06, right=.99, bottom=.14, top=.89, wspace=.28, hspace=.38)
    fig.suptitle("Pure water in CNTs: static structure factor and water-state screening (300–500 ns)", y=.97, fontsize=15, fontweight="bold")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#2878b5", lw=2, label="Axial"),
            Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="Circumferential"),
            Line2D([0], [0], color="#b2182b", lw=1.3, ls=":", label="Hansen–Verlet reference: 2.85"),
        ],
        loc="lower center", bbox_to_anchor=(.5, .015), ncol=3, frameon=False,
    )
    png = OUT / "pure_water_structure_factor_and_state_i6_i40.png"
    pdf = OUT / "pure_water_structure_factor_and_state_i6_i40.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_structures(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(4, 4, figsize=(15.8, 13.0))
    for position, (row, color) in enumerate(zip(rows, COLORS)):
        group = position // 4
        col = position % 4
        cross_ax = axes[2 * group, col]
        shell_ax = axes[2 * group + 1, col]
        index = int(row["index"])
        data = np.load(OUT / f"cnt_{index}_{index}" / "representative_structure.npz")
        all_xy, selected_xy = data["all_dxy_nm"], data["selected_dxy_nm"]
        cnt_xy, selected_z = data["cnt_dxy_nm"], data["selected_z_rel_nm"]

        cross_ax.scatter(all_xy[:, 0], all_xy[:, 1], s=4, color="#b8c2cc", alpha=.30, linewidths=0)
        cross_ax.scatter(selected_xy[:, 0], selected_xy[:, 1], s=7, color=color, alpha=.72, linewidths=0)
        cross_ax.scatter(cnt_xy[:, 0], cnt_xy[:, 1], s=2, color="#222222", alpha=.48, linewidths=0)
        radius = float(np.max(np.linalg.norm(cnt_xy, axis=1)))
        cross_ax.set_xlim(-1.08*radius, 1.08*radius)
        cross_ax.set_ylim(-1.08*radius, 1.08*radius)
        cross_ax.set_aspect("equal")
        cross_ax.set_title(f"CNT{row['cnt']} — {row['state']}", color=color, fontweight="bold", fontsize=10)
        cross_ax.set_xlabel(r"$x-x_c$ (nm)")
        cross_ax.set_ylabel(r"$y-y_c$ (nm)")
        cross_ax.grid(alpha=.14)

        if index == 6:
            shell_ax.scatter(selected_z, selected_xy[:, 0], s=11, color=color, alpha=.70, linewidths=0)
            shell_ax.set_ylabel(r"$x-x_c$ (nm)")
            shell_ax.set_title("single-file axial view", fontsize=9)
        else:
            shell_ax.scatter(selected_z, data["selected_u_nm"], s=6, color=color, alpha=.55, linewidths=0)
            shell_ax.set_ylabel(r"unfolded shell, $R_{peak}\theta$ (nm)")
            shell_ax.set_title("selected shell unfolded", fontsize=9)
        shell_ax.set_xlim(-2.0, 2.0)
        shell_ax.set_xlabel(r"$z-z_c$ (nm)")
        shell_ax.grid(alpha=.17)

    fig.subplots_adjust(left=.06, right=.99, bottom=.05, top=.92, wspace=.30, hspace=.42)
    fig.suptitle(
        "Representative confined-water structures at 500 ns\n"
        "gray: all central-CNT water; color: radial layer used for structure factor",
        y=.985, fontsize=14, fontweight="bold",
    )
    png = OUT / "pure_water_structure_views_i6_i40.png"
    pdf = OUT / "pure_water_structure_views_i6_i40.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    rows = load_rows()
    paths = [write_csv(rows), *plot_structure_factor(rows), *plot_structures(rows)]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
