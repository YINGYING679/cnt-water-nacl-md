#!/usr/bin/env python3
"""Plot the surface-normalized pure-water deficit length versus CNT radius."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT")
OUT = ROOT / "pure_water_deficit_length_i6_i30"
SOURCE = OUT / "pure_water_delta_ex_R_K_table_i6_i40_300_500ns.csv"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    labels = [row["CNT"] for row in source_rows]
    radius = np.array([float(row["R_nm"]) for row in source_rows])
    ratio = np.array(
        [float(row["K_C_Cw_CNT_over_Cw_bulk"]) for row in source_rows]
    )
    ratio_sem = np.array([float(row["K_C_SEM"]) for row in source_rows])

    # Surface-normalized deficit length.  For a cylinder S/V=2/R:
    # K_C = 1 - (S/V) delta_theta = 1 - 2 delta_theta/R.
    delta = 0.5 * radius * (1.0 - ratio)
    delta_sem = 0.5 * radius * ratio_sem
    delta_over_r = 0.5 * (1.0 - ratio)
    delta_over_r_sem = 0.5 * ratio_sem

    table_csv = OUT / "pure_water_delta_theta_R_K_table_i6_i40_300_500ns.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "CNT",
                "R_nm",
                "K_C_Cw_CNT_over_Cw_bulk",
                "K_C_SEM",
                "delta_theta_nm",
                "delta_theta_SEM_nm",
                "delta_theta_over_R",
                "delta_theta_over_R_SEM",
                "definition",
            ]
        )
        for values in zip(
            labels,
            radius,
            ratio,
            ratio_sem,
            delta,
            delta_sem,
            delta_over_r,
            delta_over_r_sem,
        ):
            writer.writerow(
                [*values, "surface-normalized deficit length: R/2*(1-K_C)"]
            )

    table_md = OUT / "pure_water_delta_theta_R_K_table_i6_i40_300_500ns.md"
    with table_md.open("w", encoding="utf-8") as handle:
        handle.write(
            "| CNT | $R$ (nm) | $K_C=C_w^{CNT}/C_w^{bulk}$ | "
            "$\\delta_\\Theta$ (nm) | $\\delta_\\Theta/R$ |\n"
        )
        handle.write("|---|---:|---:|---:|---:|\n")
        for label, r_nm, k_c, delta_nm, delta_r in zip(
            labels, radius, ratio, delta, delta_over_r
        ):
            handle.write(
                f"| {label} | {r_nm:.4f} | {k_c:.6f} | "
                f"{delta_nm:.6f} | {delta_r:.6f} |\n"
            )
        handle.write(
            "\n$\\delta_\\Theta(R)=R[1-K_C]/2$; values are based on "
            "300–500 ns concentration ratios. SEM values are retained in the CSV.\n"
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    ax.plot(
        radius,
        delta,
        color="#2166ac",
        marker="o",
        markersize=7.5,
        linewidth=2.0,
        label=r"Surface-normalized deficit length, $\delta_\Theta$",
    )

    offsets = [
        (8, 10),
        (8, -25),
        (-68, 15),
        (10, -30),
        (8, 11),
        (8, -24),
        (-60, 16),
        (-66, -27),
    ]
    for x, y, label, offset in zip(radius, delta, labels, offsets):
        ax.annotate(
            f"{label}\n{y:.3f} nm",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color="#17365d",
        )

    ax.text(
        0.02,
        0.96,
        r"$K_C=C_w^{CNT}/C_w^{bulk}$" "\n" r"$\delta_\Theta=\frac{R}{2}(1-K_C)$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.set_xlabel(r"CNT radius, $R$ (nm)")
    ax.set_ylabel(r"Surface-normalized deficit length, $\delta_\Theta$ (nm)")
    ax.set_title(
        "Pure water: surface-normalized deficit length vs CNT radius\n"
        "(300–500 ns)"
    )
    ax.set_xlim(0.30, 2.82)
    padding = 0.006
    ax.set_ylim(float(delta.min() - padding), float(delta.max() + padding))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    figure_base = OUT / "pure_water_delta_theta_vs_R_i6_i40_300_500ns"
    fig.savefig(figure_base.with_suffix(".png"), dpi=300)
    fig.savefig(figure_base.with_suffix(".pdf"))
    plt.close(fig)

    method = OUT / "PURE_WATER_DELTA_THETA_METHOD.md"
    method.write_text(
        """# Surface-normalized pure-water deficit length

The concentration partition coefficient is

$$K_C=\\frac{C_w^{CNT}}{C_w^{bulk}}.$$

Defining the integrated deficit per carbon-wall area gives

$$K_C=1-\\frac{S}{V}\\delta_\\Theta
     =1-\\frac{2\\delta_\\Theta}{R},$$

because $S/V=2/R$ for a cylindrical pore. Therefore

$$\\delta_\\Theta(R)=\\frac{R}{2}(1-K_C).$$

Treating $R$ as exact, the propagated block SEM is

$$\\mathrm{SEM}_{\\delta_\\Theta}=\\frac{R}{2}\\mathrm{SEM}_{K_C}.$$

This is a surface-normalized integrated deficit length, not the thickness of a
sharp empty annulus. The older equivalent exclusion-layer results are retained
as separate files.
""",
        encoding="utf-8",
    )

    for path in (table_csv, table_md, figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf"), method):
        print(path)


if __name__ == "__main__":
    main()
