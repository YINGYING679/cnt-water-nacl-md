#!/usr/bin/env python3
"""Bulk-reservoir water diffusion for the five pure-water CNT systems.

Water oxygens are selected at each time origin from a 2 nm slab centred in the
periodic reservoir gap opposite the CNT.  This is the same outer-reservoir
plateau used for the bulk concentration, with a small per-frame recentering
that keeps equal clearance from both CNT/graphene faces as Lz changes.
Molecules are followed without a continuous-window condition,
because such a condition preferentially removes fast molecules.  The fit is
limited to 20--200 ps, while the closest carbon surface is about 2 nm beyond
the inner boundary of the selection window.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT")
OUT_ROOT = ROOT / "pure_water_bulk_diffusion_i6_i30"
INDICES = (6, 8, 12, 22, 30)
N_CNT_WALL = {index: 100 * index for index in INDICES}

START_PS = 300_000.0
END_PS = 500_000.0
FRAME_DT_PS = 2.0
BULK_HALF_WIDTH_NM = 1.0
ORIGIN_STRIDE_PS = 200.0
ORIGIN_STRIDE_FRAMES = int(round(ORIGIN_STRIDE_PS / FRAME_DT_PS))
MAX_LAG_PS = 500.0
MAX_LAG_FRAMES = int(round(MAX_LAG_PS / FRAME_DT_PS))
FIT_BEGIN_PS = 20.0
FIT_END_PS = 200.0
N_BLOCKS = 10


@dataclass
class Origin:
    start_frame: int
    block: int
    ow_ids: np.ndarray
    frac0: np.ndarray
    box0_nm: np.ndarray


def block_index(time_ps: float) -> int:
    fraction = (time_ps - START_PS) / (END_PS - START_PS)
    return min(N_BLOCKS - 1, max(0, int(fraction * N_BLOCKS)))


def reservoir_geometry(cnt_positions_nm: np.ndarray, box_nm: np.ndarray) -> tuple[float, float, float, float]:
    """Return wrapped CNT limits, midpoint and length of the periodic reservoir gap."""
    cnt_z = np.mod(cnt_positions_nm[:, 2], box_nm[2])
    z_min = float(np.min(cnt_z))
    z_max = float(np.max(cnt_z))
    gap = float(box_nm[2] - (z_max - z_min))
    midpoint = float(np.mod(0.5 * (z_max + z_min + box_nm[2]), box_nm[2]))
    return z_min, z_max, midpoint, gap


def bulk_mask(positions_nm: np.ndarray, box_nm: np.ndarray, midpoint_z_nm: float) -> np.ndarray:
    z = np.mod(positions_nm[:, 2], box_nm[2])
    dz = z - midpoint_z_nm
    dz -= box_nm[2] * np.rint(dz / box_nm[2])
    return np.abs(dz) < BULK_HALF_WIDTH_NM


def fit_curve(lag_ps: np.ndarray, values: np.ndarray, divisor: float) -> dict[str, float]:
    use = (
        (lag_ps >= FIT_BEGIN_PS)
        & (lag_ps <= FIT_END_PS)
        & np.isfinite(values)
    )
    x = lag_ps[use]
    y = values[use]
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    diffusion = float(slope / divisor * 1000.0)  # nm2/ps -> 1e-9 m2/s
    return {
        "fit_begin_ps": FIT_BEGIN_PS,
        "fit_end_ps": FIT_END_PS,
        "slope_nm2_per_ps": float(slope),
        "intercept_nm2": float(intercept),
        "r_squared": r_squared,
        "D_1e-9_m2_per_s": diffusion,
    }


def sem_by_lag(block_values: np.ndarray) -> np.ndarray:
    result = np.full(block_values.shape[1], np.nan)
    for lag in range(block_values.shape[1]):
        values = block_values[:, lag]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            result[lag] = np.std(values, ddof=1) / math.sqrt(values.size)
    return result


def mean_from_sums(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_sums = np.sum(sums, axis=0)
    total_counts = np.sum(counts, axis=0)
    pooled = np.divide(
        total_sums,
        total_counts,
        out=np.full(sums.shape[1], np.nan),
        where=total_counts > 0,
    )
    block_mean = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return pooled, block_mean, sem_by_lag(block_mean)


def probe(index: int) -> None:
    base = ROOT / f"cnt_{index}_{index}" / "NPT" / "continue_20ns_to_500ns"
    top = base / "npt_20ns_to_500ns.part0002.gro"
    traj = base / "npt_20ns_to_500ns.part0002.xtc"
    universe = mda.Universe(str(top), str(traj))
    cnt = universe.atoms[: N_CNT_WALL[index]]
    ow = universe.select_atoms("name OW")
    first_time_ps = float(universe.trajectory[0].time)
    start_frame = int(round((START_PS - first_time_ps) / float(universe.trajectory.dt)))
    ts = universe.trajectory[start_frame]
    box_nm = ts.dimensions[:3].astype(float) / 10.0
    cnt_nm = cnt.positions.astype(float) / 10.0
    ow_nm = ow.positions.astype(float) / 10.0
    z_min, z_max, midpoint, reservoir_gap = reservoir_geometry(cnt_nm, box_nm)
    mask = bulk_mask(ow_nm, box_nm, midpoint)
    clearance = 0.5 * reservoir_gap - BULK_HALF_WIDTH_NM
    print(f"CNT({index},{index}) probe at {float(ts.time)/1000:.3f} ns")
    print(f"box_nm={box_nm.tolist()}")
    print(f"CNT z range={z_min:.4f}..{z_max:.4f} nm")
    print(f"periodic reservoir gap={reservoir_gap:.4f} nm; midpoint={midpoint:.4f} nm")
    print(f"bulk slab=midpoint +/- {BULK_HALF_WIDTH_NM:.1f} nm (periodically wrapped)")
    print(f"clearance from each slab edge to CNT carbon={clearance:.4f} nm")
    print(f"selected OW={int(np.count_nonzero(mask))} / {len(ow)}")


def analyze(index: int) -> None:
    base = ROOT / f"cnt_{index}_{index}" / "NPT" / "continue_20ns_to_500ns"
    top = base / "npt_20ns_to_500ns.part0002.gro"
    traj = base / "npt_20ns_to_500ns.part0002.xtc"
    for path in (top, traj):
        if not path.is_file():
            raise FileNotFoundError(path)

    out = OUT_ROOT / f"cnt_{index}_{index}"
    out.mkdir(parents=True, exist_ok=True)
    universe = mda.Universe(str(top), str(traj))
    cnt = universe.atoms[: N_CNT_WALL[index]]
    ow = universe.select_atoms("name OW")
    if len(ow) == 0:
        raise RuntimeError("No atoms named OW were found")

    first_time_ps = float(universe.trajectory[0].time)
    trajectory_dt_ps = float(universe.trajectory.dt)
    start_frame_index = int(round((START_PS - first_time_ps) / trajectory_dt_ps))
    end_frame_index = int(round((END_PS - first_time_ps) / trajectory_dt_ps))
    if start_frame_index < 0 or end_frame_index >= len(universe.trajectory):
        raise RuntimeError("Requested 300--500 ns window lies outside the trajectory")

    n_lags = MAX_LAG_FRAMES + 1
    count = np.zeros((N_BLOCKS, n_lags), dtype=np.int64)
    retained = np.zeros_like(count)
    components = ("x", "y", "z", "xy", "xyz")
    raw_sums = {key: np.zeros((N_BLOCKS, n_lags), dtype=float) for key in components}
    corrected_sums = {key: np.zeros((N_BLOCKS, n_lags), dtype=float) for key in components}
    origin_counts = np.zeros(N_BLOCKS, dtype=np.int64)
    selected_counts = np.zeros(N_BLOCKS, dtype=np.int64)
    active: list[Origin] = []

    analysed_frames = 0
    first_geometry: dict[str, float | list[float]] = {}
    for ts in universe.trajectory[start_frame_index : end_frame_index + 1]:
        time_ps = float(ts.time)
        analysis_frame = int(round((time_ps - START_PS) / FRAME_DT_PS))
        box_nm = ts.dimensions[:3].astype(np.float64) / 10.0
        cnt_nm = cnt.positions.astype(np.float64) / 10.0
        ow_nm = ow.positions.astype(np.float64) / 10.0
        ow_frac = np.mod(ow_nm / box_nm, 1.0)
        z_min, z_max, reservoir_midpoint, reservoir_gap = reservoir_geometry(cnt_nm, box_nm)
        current_bulk = bulk_mask(ow_nm, box_nm, reservoir_midpoint)

        if not first_geometry:
            first_geometry = {
                "box_nm": box_nm.tolist(),
                "CNT_z_min_nm": z_min,
                "CNT_z_max_nm": z_max,
                "periodic_reservoir_gap_nm": reservoir_gap,
                "reservoir_gap_midpoint_z_nm": reservoir_midpoint,
                "bulk_slab_half_width_nm": BULK_HALF_WIDTH_NM,
                "clearance_each_side_nm": 0.5 * reservoir_gap - BULK_HALF_WIDTH_NM,
            }

        next_active: list[Origin] = []
        for origin in active:
            lag = analysis_frame - origin.start_frame
            if lag > MAX_LAG_FRAMES:
                continue
            delta_frac = ow_frac[origin.ow_ids] - origin.frac0
            delta_frac -= np.rint(delta_frac)
            mean_box_nm = 0.5 * (box_nm + origin.box0_nm)
            delta_nm = delta_frac * mean_box_nm
            n_selected = delta_nm.shape[0]
            drift_nm = np.mean(delta_nm, axis=0)
            internal_nm = delta_nm - drift_nm

            raw_xyz = delta_nm * delta_nm
            corrected_xyz = internal_nm * internal_nm
            block = origin.block
            count[block, lag] += n_selected
            retained[block, lag] += int(np.count_nonzero(current_bulk[origin.ow_ids]))
            raw_sums["x"][block, lag] += float(np.sum(raw_xyz[:, 0]))
            raw_sums["y"][block, lag] += float(np.sum(raw_xyz[:, 1]))
            raw_sums["z"][block, lag] += float(np.sum(raw_xyz[:, 2]))
            raw_sums["xy"][block, lag] += float(np.sum(raw_xyz[:, 0] + raw_xyz[:, 1]))
            raw_sums["xyz"][block, lag] += float(np.sum(raw_xyz))
            corrected_sums["x"][block, lag] += float(np.sum(corrected_xyz[:, 0]))
            corrected_sums["y"][block, lag] += float(np.sum(corrected_xyz[:, 1]))
            corrected_sums["z"][block, lag] += float(np.sum(corrected_xyz[:, 2]))
            corrected_sums["xy"][block, lag] += float(np.sum(corrected_xyz[:, 0] + corrected_xyz[:, 1]))
            corrected_sums["xyz"][block, lag] += float(np.sum(corrected_xyz))
            if lag < MAX_LAG_FRAMES:
                next_active.append(origin)
        active = next_active

        if analysis_frame % ORIGIN_STRIDE_FRAMES == 0 and time_ps <= END_PS - MAX_LAG_PS:
            ids = np.flatnonzero(current_bulk)
            block = block_index(time_ps)
            active.append(
                Origin(
                    start_frame=analysis_frame,
                    block=block,
                    ow_ids=ids,
                    frac0=ow_frac[ids].copy(),
                    box0_nm=box_nm.copy(),
                )
            )
            origin_counts[block] += 1
            selected_counts[block] += ids.size
            count[block, 0] += ids.size
            retained[block, 0] += ids.size

        analysed_frames += 1
        if analysed_frames % 10_000 == 0:
            print(f"i={index}: processed {analysed_frames} frames; time={time_ps/1000:.1f} ns", flush=True)

    lag_ps = np.arange(n_lags, dtype=float) * FRAME_DT_PS
    divisors = {"x": 2.0, "y": 2.0, "z": 2.0, "xy": 4.0, "xyz": 6.0}
    output_arrays: dict[str, np.ndarray] = {"lag_ps": lag_ps}
    fit_results: dict[str, dict[str, dict[str, float]]] = {"raw": {}, "collective_corrected": {}}

    for mode, sums_by_component in (("raw", raw_sums), ("collective_corrected", corrected_sums)):
        for component in components:
            mean, block_mean, sem = mean_from_sums(sums_by_component[component], count)
            output_arrays[f"MSD_{component}_{mode}_nm2"] = mean
            output_arrays[f"MSD_{component}_{mode}_SEM_nm2"] = sem
            output_arrays[f"MSD_{component}_{mode}_blocks_nm2"] = block_mean
            pooled_fit = fit_curve(lag_ps, mean, divisors[component])
            block_d = np.array(
                [fit_curve(lag_ps, row, divisors[component])["D_1e-9_m2_per_s"] for row in block_mean]
            )
            pooled_fit["D_block_mean_1e-9_m2_per_s"] = float(np.mean(block_d))
            pooled_fit["D_block_SEM_1e-9_m2_per_s"] = float(np.std(block_d, ddof=1) / math.sqrt(block_d.size))
            fit_results[mode][component] = pooled_fit

    total_count = np.sum(count, axis=0)
    total_retained = np.sum(retained, axis=0)
    retained_fraction = np.divide(
        total_retained,
        total_count,
        out=np.full(n_lags, np.nan),
        where=total_count > 0,
    )
    output_arrays["sample_count"] = total_count
    output_arrays["fraction_currently_in_bulk_window"] = retained_fraction
    np.savez_compressed(out / "bulk_reservoir_msd_blocks.npz", **output_arrays)

    csv_columns = ["lag_ps", "sample_count", "fraction_currently_in_bulk_window"]
    csv_data = [lag_ps, total_count, retained_fraction]
    for mode in ("raw", "collective_corrected"):
        for component in components:
            csv_columns.extend([f"MSD_{component}_{mode}_nm2", f"MSD_{component}_{mode}_SEM_nm2"])
            csv_data.extend(
                [
                    output_arrays[f"MSD_{component}_{mode}_nm2"],
                    output_arrays[f"MSD_{component}_{mode}_SEM_nm2"],
                ]
            )
    np.savetxt(
        out / "bulk_reservoir_msd.csv",
        np.column_stack(csv_data),
        delimiter=",",
        header=",".join(csv_columns),
        comments="",
    )

    fit_end_index = int(round(FIT_END_PS / FRAME_DT_PS))
    max_lag_index = int(round(MAX_LAG_PS / FRAME_DT_PS))
    summary = {
        "system": f"pure-water CNT({index},{index}) reservoirs",
        "analysis_window_ns": [START_PS / 1000.0, END_PS / 1000.0],
        "trajectory_frame_spacing_ps": FRAME_DT_PS,
        "time_origin_spacing_ps": ORIGIN_STRIDE_PS,
        "maximum_lag_ps": MAX_LAG_PS,
        "blocks": N_BLOCKS,
        "selection": {
            "definition": "OW in a 2 nm slab centered in the periodic reservoir gap opposite the CNT at each time origin",
            "periodic_interpretation": "the slab may wrap across z=0; it is recentered each frame to retain equal clearance from the two CNT/graphene faces",
            "continuous_residence_required": False,
            "reason": "requiring window residence preferentially removes fast molecules",
            "mean_selected_OW_per_origin": float(np.sum(selected_counts) / np.sum(origin_counts)),
            "fraction_still_in_selection_at_200ps": float(retained_fraction[fit_end_index]),
            "fraction_still_in_selection_at_500ps": float(retained_fraction[max_lag_index]),
        },
        "geometry_at_300ns": first_geometry,
        "displacement": {
            "PBC": "minimum-image fractional displacement, converted with mean origin/current box lengths",
            "collective_correction": "subtract mean xyz displacement of selected waters separately for each origin and lag",
        },
        "fit_models": {
            "x_y_z": "MSD_alpha = 2 D_alpha t + intercept",
            "xy": "MSD_xy = 4 D_xy t + intercept",
            "xyz": "MSD_3D = 6 D_3D t + intercept",
        },
        "fit_range_ps": [FIT_BEGIN_PS, FIT_END_PS],
        "SEM": "SEM of diffusion coefficients across ten 20 ns time-origin blocks",
        "origin_counts_by_block": origin_counts.tolist(),
        "results": fit_results,
        "recommended_D_bulk": "collective_corrected xyz D_3D; compare x, y and z for isotropy",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = fit_results["collective_corrected"]
    print(
        f"i={index}: D3D={result['xyz']['D_block_mean_1e-9_m2_per_s']:.6f} +/- "
        f"{result['xyz']['D_block_SEM_1e-9_m2_per_s']:.6f}; "
        f"Dx/Dy/Dz={result['x']['D_block_mean_1e-9_m2_per_s']:.6f}/"
        f"{result['y']['D_block_mean_1e-9_m2_per_s']:.6f}/"
        f"{result['z']['D_block_mean_1e-9_m2_per_s']:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, choices=INDICES, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        probe(args.index)
    else:
        analyze(args.index)


if __name__ == "__main__":
    main()
