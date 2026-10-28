#!/usr/bin/env python3
"""Merge original and newly added CNT radii and plot two deficit definitions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT_NaCl_rigid_only")
OLD = ROOT / "saltwater_ion_exclusion_lengths_300_500ns" / "Na_Cl_equivalent_exclusion_lengths_i8_i30_300_500ns.csv"
NEW = ROOT / "saltwater_ion_partition_new_radii_300_500ns" / "systems"
OUT = ROOT / "saltwater_ion_exclusion_lengths_300_500ns"
INDICES = (8, 9, 10, 12, 22, 30, 40)
NEW_INDICES = (9, 10, 40)
CONCENTRATIONS = ("0.2", "1", "2", "3", "4", "5")
COLORS = ("#355f9d", "#2a9d8f", "#73ad43", "#d4a017", "#e8782d", "#c33d4a")
MARKERS = ("o", "s", "^", "D", "v", "P")


def sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def sampling_status(blocks: list[dict[str, str]], species: str) -> tuple[int, str]:
    nonzero = sum(float(row[f"C{species}_CNT_M"]) > 0.0 for row in blocks)
    if nonzero == 0:
        status = "no ions observed in any 50 ns block"
    elif nonzero < 4:
        status = f"sparse/intermittent: nonzero in {nonzero}/4 blocks"
    else:
        status = "nonzero in 4/4 blocks"
    return nonzero, status


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with OLD.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            index = int(source["CNT_index"])
            if index not in INDICES:
                continue
            rows.append(
                {
                    "CNT": source["CNT"],
                    "CNT_index": index,
                    "R_nm": float(source["R_nm"]),
                    "nominal_NaCl_M": float(source["nominal_NaCl_M"]),
                    "actual_Csalt_bulk_M": float(source["actual_Csalt_bulk_M"]),
                    "actual_Csalt_bulk_SEM_M": float(source["actual_Csalt_bulk_SEM_M"]),
                    "K_Na": float(source["K_Na"]),
                    "K_Na_SEM": float(source["K_Na_SEM"]),
                    "K_Cl": float(source["K_Cl"]),
                    "K_Cl_SEM": float(source["K_Cl_SEM"]),
                    "Na_nonzero_blocks_of_4": int(source["Na_nonzero_blocks_of_4"]),
                    "Na_sampling_status": source["Na_sampling_status"],
                    "Cl_nonzero_blocks_of_4": int(source["Cl_nonzero_blocks_of_4"]),
                    "Cl_sampling_status": source["Cl_sampling_status"],
                    "source": "original ion concentration analysis",
                }
            )

    for index in NEW_INDICES:
        for concentration in CONCENTRATIONS:
            token = concentration.replace(".", "p") + "M"
            directory = NEW / f"cnt{index}_{token}"
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            with (directory / "concentration_blocks.csv").open(newline="", encoding="utf-8") as handle:
                blocks = list(csv.DictReader(handle))
            k_na = np.array([float(row["K_Na"]) for row in blocks])
            k_cl = np.array([float(row["K_Cl"]) for row in blocks])
            c_bulk = np.array([float(row["Csalt_bulk_M"]) for row in blocks])
            na_nonzero, na_status = sampling_status(blocks, "Na")
            cl_nonzero, cl_status = sampling_status(blocks, "Cl")
            rows.append(
                {
                    "CNT": f"({index},{index})",
                    "CNT_index": index,
                    "R_nm": float(summary["geometry_at_300ns"]["CNT_radius_nm"]),
                    "nominal_NaCl_M": float(concentration),
                    "actual_Csalt_bulk_M": float(np.mean(c_bulk)),
                    "actual_Csalt_bulk_SEM_M": sem(c_bulk),
                    "K_Na": float(np.mean(k_na)),
                    "K_Na_SEM": sem(k_na),
                    "K_Cl": float(np.mean(k_cl)),
                    "K_Cl_SEM": sem(k_cl),
                    "Na_nonzero_blocks_of_4": na_nonzero,
                    "Na_sampling_status": na_status,
                    "Cl_nonzero_blocks_of_4": cl_nonzero,
                    "Cl_sampling_status": cl_status,
                    "source": "new-radius concentration-only analysis",
                }
            )

    for row in rows:
        radius = float(row["R_nm"])
        for species in ("Na", "Cl"):
            k_value = float(row[f"K_{species}"])
            k_sem = float(row[f"K_{species}_SEM"])
            if k_value < 0.0:
                raise ValueError(f"negative K for {row['CNT']} {species}")
            row[f"delta_ex_{species}_nm"] = radius * (1.0 - math.sqrt(k_value))
            row[f"delta_ex_{species}_SEM_nm"] = (
                math.nan if k_value == 0.0 else radius * k_sem / (2.0 * math.sqrt(k_value))
            )
            row[f"delta_theta_{species}_nm"] = 0.5 * radius * (1.0 - k_value)
            row[f"delta_theta_{species}_SEM_nm"] = 0.5 * radius * k_sem

    rows.sort(key=lambda row: (int(row["CNT_index"]), float(row["nominal_NaCl_M"])))
    return rows


def write_long_table(rows: list[dict[str, object]]) -> Path:
    path = OUT / "Na_Cl_deficit_lengths_i8_i40_300_500ns.csv"
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_compact_table(rows: list[dict[str, object]], kind: str) -> tuple[Path, Path]:
    symbol = "delta_ex" if kind == "ex" else "delta_theta"
    path = OUT / f"Na_Cl_{symbol}_R_K_table_i8_i40_300_500ns.csv"
    fields = [
        "CNT", "R_nm", "nominal_NaCl_M", "actual_Csalt_bulk_M",
        "K_Na", f"{symbol}_Na_nm", "K_Cl", f"{symbol}_Cl_nm",
        "Na_sampling_status", "Cl_sampling_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md = path.with_suffix(".md")
    delta_label = r"\delta_{\rm ex}" if kind == "ex" else r"\delta_\Theta"
    with md.open("w", encoding="utf-8") as handle:
        handle.write(
            f"| CNT | $R$ (nm) | nominal (M) | $K_{{Na}}$ | "
            f"${delta_label},Na$ (nm) | $K_{{Cl}}$ | ${delta_label},Cl$ (nm) |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['CNT']} | {float(row['R_nm']):.4f} | "
                f"{float(row['nominal_NaCl_M']):.1f} | "
                f"{float(row['K_Na']):.6f} | {float(row[f'{symbol}_Na_nm']):.6f} | "
                f"{float(row['K_Cl']):.6f} | {float(row[f'{symbol}_Cl_nm']):.6f} |\n"
            )
    return path, md


def wide_rows(rows: list[dict[str, object]], symbol: str) -> tuple[list[str], list[list[object]]]:
    headers = ["CNT", "R_nm"]
    for concentration in CONCENTRATIONS:
        headers.extend([f"delta_Na_{concentration}M_nm", f"delta_Cl_{concentration}M_nm"])
    table: list[list[object]] = []
    for index in INDICES:
        subset = [row for row in rows if int(row["CNT_index"]) == index]
        subset.sort(key=lambda row: float(row["nominal_NaCl_M"]))
        line: list[object] = [f"({index},{index})", float(subset[0]["R_nm"])]
        for row in subset:
            line.extend([float(row[f"{symbol}_Na_nm"]), float(row[f"{symbol}_Cl_nm"])])
        table.append(line)
    return headers, table


def write_wide_table(rows: list[dict[str, object]], kind: str) -> tuple[Path, Path]:
    symbol = "delta_ex" if kind == "ex" else "delta_theta"
    headers, table = wide_rows(rows, symbol)
    path = OUT / f"Na_Cl_{symbol}_wide_table_i8_i40_300_500ns.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table)
    md = path.with_suffix(".md")
    with md.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "---|" * len(headers) + "\n")
        for line in table:
            handle.write(
                "| " + " | ".join(
                    [str(line[0]), f"{float(line[1]):.4f}"]
                    + [f"{float(value):.5f}" for value in line[2:]]
                ) + " |\n"
            )
    return path, md


def plot_two_panel(rows: list[dict[str, object]], kind: str) -> tuple[Path, Path]:
    symbol = "delta_ex" if kind == "ex" else "delta_theta"
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9), sharex=True, sharey=True)
    all_y: list[float] = []
    for ax, species in zip(axes, ("Na", "Cl")):
        for concentration, color, marker in zip(CONCENTRATIONS, COLORS, MARKERS):
            subset = [row for row in rows if float(row["nominal_NaCl_M"]) == float(concentration)]
            subset.sort(key=lambda row: float(row["R_nm"]))
            x = np.array([float(row["R_nm"]) for row in subset])
            y = np.array([float(row[f"{symbol}_{species}_nm"]) for row in subset])
            all_y.extend(y.tolist())
            ax.plot(x, y, color=color, marker=marker, markersize=5.8, linewidth=1.7, label=f"{concentration} M")
            zero = np.array([float(row[f"K_{species}"]) == 0.0 for row in subset])
            if np.any(zero):
                ax.scatter(x[zero], y[zero], s=72, marker="x", linewidth=1.6, color=color, zorder=4)
        species_label = r"Na$^+$" if species == "Na" else r"Cl$^-$"
        descriptor = "equivalent exclusion-layer thickness" if kind == "ex" else "surface-normalized deficit length"
        ax.set_title(f"{species_label} {descriptor}", fontweight="bold")
        ax.grid(alpha=0.23)
        ax.set_xlabel(r"CNT radius, $R$ (nm)")
    radii = [float(next(row["R_nm"] for row in rows if int(row["CNT_index"]) == index)) for index in INDICES]
    tick_labels = [f"{r:.3f}\n({i},{i})" for i, r in zip(INDICES, radii)]
    for ax in axes:
        ax.set_xticks(radii, tick_labels, fontsize=8)
        ax.set_xlim(min(radii) - 0.07, max(radii) + 0.10)
    y_min, y_max = min(all_y), max(all_y)
    pad = max(0.02, 0.07 * (y_max - y_min))
    axes[0].set_ylim(y_min - pad, y_max + pad)
    ylabel = r"Equivalent exclusion-layer thickness, $\delta_{\rm ex,i}$ (nm)" if kind == "ex" else r"Surface-normalized deficit length, $\delta_{\Theta,i}$ (nm)"
    axes[0].set_ylabel(ylabel)
    axes[0].legend(title="Nominal NaCl", frameon=False, ncol=2, loc="best", fontsize=8.5)
    title = "Equivalent ion exclusion-layer thickness" if kind == "ex" else "Surface-normalized ion deficit length"
    fig.suptitle(f"{title} vs CNT radius (300–500 ns)", fontsize=15, fontweight="bold")
    formula = r"$K_i=C_i^{CNT}/C_i^{bulk}$; $\delta_{\rm ex,i}=R(1-\sqrt{K_i})$." if kind == "ex" else r"$K_i=C_i^{CNT}/C_i^{bulk}$; $\delta_{\Theta,i}=R(1-K_i)/2$."
    fig.text(0.5, 0.015, formula + "  ×: no ions observed in the four 50 ns blocks; lines are guides to the eye.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.94))
    base = OUT / f"Na_Cl_{symbol}_vs_R_six_concentrations_i8_i40_300_500ns"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def plot_composite_ex(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    _, table = wide_rows(rows, "delta_ex")
    fig = plt.figure(figsize=(19.2, 10.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.42, 1.0), hspace=0.18, wspace=0.08)
    table_ax = fig.add_subplot(grid[0, :])
    axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    table_ax.axis("off")
    headers = ["CNT"]
    for concentration in CONCENTRATIONS:
        headers.extend([rf"$\delta_{{Na}}$ ({concentration} M)", rf"$\delta_{{Cl}}$ ({concentration} M)"])
    cell_text = [[line[0]] + [f"{float(value):.5f}" for value in line[2:]] for line in table]
    mpl_table = table_ax.table(cellText=cell_text, colLabels=headers, cellLoc="center", loc="center")
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8.0)
    mpl_table.scale(1.0, 1.38)
    for (row, _), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4f79c7")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#e2e8f5" if row % 2 else "#cbd5eb")
        cell.set_edgecolor("white")

    all_y: list[float] = []
    for ax, species in zip(axes, ("Na", "Cl")):
        for concentration, color, marker in zip(CONCENTRATIONS, COLORS, MARKERS):
            subset = [row for row in rows if float(row["nominal_NaCl_M"]) == float(concentration)]
            subset.sort(key=lambda row: float(row["R_nm"]))
            x = np.array([float(row["R_nm"]) for row in subset])
            y = np.array([float(row[f"delta_ex_{species}_nm"]) for row in subset])
            all_y.extend(y.tolist())
            ax.plot(x, y, color=color, marker=marker, markersize=5.8, linewidth=1.7, label=f"{concentration} M")
            zero = np.array([float(row[f"K_{species}"]) == 0.0 for row in subset])
            if np.any(zero):
                ax.scatter(x[zero], y[zero], s=72, marker="x", linewidth=1.6, color=color, zorder=4)
        label = r"Na$^+$" if species == "Na" else r"Cl$^-$"
        ax.set_title(f"{label} equivalent exclusion-layer thickness", fontweight="bold")
        ax.set_xlabel(r"CNT radius, $R$ (nm)")
        ax.grid(alpha=0.23)
    radii = [float(next(row["R_nm"] for row in rows if int(row["CNT_index"]) == index)) for index in INDICES]
    for ax in axes:
        ax.set_xticks(radii, [f"{r:.3f}\n({i},{i})" for i, r in zip(INDICES, radii)], fontsize=8)
        ax.set_xlim(min(radii) - 0.07, max(radii) + 0.10)
    y_min, y_max = min(all_y), max(all_y)
    pad = max(0.02, 0.07 * (y_max - y_min))
    axes[0].set_ylim(y_min - pad, y_max + pad)
    axes[0].set_ylabel(r"Equivalent exclusion-layer thickness, $\delta_{\rm ex,i}$ (nm)")
    axes[0].legend(title="Nominal NaCl", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Equivalent ion exclusion-layer thickness vs CNT radius (300–500 ns)", fontsize=16, fontweight="bold", y=.985)
    fig.text(.5, .018, r"$K_i=C_i^{CNT}/C_i^{bulk}$; $\delta_{\rm ex,i}=R(1-\sqrt{K_i})$. ×: no ions observed in the four 50 ns blocks; lines are guides to the eye.", ha="center", fontsize=9.5)
    base = OUT / "Na_Cl_delta_ex_table_and_plot_i8_i40_300_500ns"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    expected = len(INDICES) * len(CONCENTRATIONS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    paths: list[Path] = [write_long_table(rows)]
    for kind in ("ex", "theta"):
        paths.extend(write_compact_table(rows, kind))
        paths.extend(write_wide_table(rows, kind))
        paths.extend(plot_two_panel(rows, kind))
    paths.extend(plot_composite_ex(rows))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
