#!/usr/bin/env python3
"""Paper-inspired liquid/crystal screening for pure water in CNTs.

The analysis combines:
1. A static structure factor for the confined-water radial layer, adapted from
   Eq. 2 of ACS Nano 2026, DOI 10.1021/acsnano.6c07508.
2. The Hansen-Verlet reference threshold S_max = 2.85.
3. Residence-conditioned axial MSD before and after subtracting collective
   water displacement.

The structure factor is evaluated in the central 4 nm of the CNT.  A
cylindrical water shell is unfolded to (R_peak * theta, z); single-file water
uses its one-dimensional axial structure factor.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT_NaCl_rigid_only")
OUTPUT_ROOT = ROOT / "saltwater_phase_state_30systems_300_500ns"
CONCENTRATIONS = ("0.2M", "1M", "2M", "3M", "4M", "5M")
from analyze_saltwater_radial_layers_300_500ns import system_paths
INDICES = (6, 8, 12, 22, 30)
N_CNT_WALL = {index: 100 * index for index in INDICES}

# Boundaries follow minima in the saved 300--500 ns radial density profiles.
SHELLS = {
    6: {"kind": "single_file", "r_min_nm": 0.000, "r_max_nm": 0.130, "r_peak_nm": 0.050},
    8: {"kind": "cylindrical_shell", "r_min_nm": 0.090, "r_max_nm": 0.270, "r_peak_nm": 0.210},
    12: {"kind": "outer_shell", "r_min_nm": 0.330, "r_max_nm": 0.540, "r_peak_nm": 0.470},
    22: {"kind": "outer_shell", "r_min_nm": 1.000, "r_max_nm": 1.220, "r_peak_nm": 1.150},
    30: {"kind": "outer_shell", "r_min_nm": 1.550, "r_max_nm": 1.770, "r_peak_nm": 1.690},
}

START_PS = 300_000.0
END_PS = 500_000.0
FRAME_DT_PS = 2.0
STRUCTURE_STRIDE_FRAMES = 10  # 20 ps
CENTRAL_LENGTH_NM = 4.0
N_BLOCKS = 4
GRID = 256
Q_MIN_NM_INV = 10.0
Q_MAX_NM_INV = 50.0
HANSEN_VERLET_THRESHOLD = 2.85

ORIGIN_STRIDE_FRAMES = 10  # 20 ps
MAX_LAG_PS = 500.0
MAX_LAG_FRAMES = int(round(MAX_LAG_PS / FRAME_DT_PS))
FIT_BEGIN_PS = 20.0
FIT_END_PS = 200.0


@dataclass
class Origin:
    start_frame: int
    block: int
    ow_ids: np.ndarray
    z0_rel_nm: np.ndarray
    alive: np.ndarray


def sem_rows(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])


def fit_msd(lag_ps: np.ndarray, msd_nm2: np.ndarray) -> dict[str, float]:
    use = (
        (lag_ps >= FIT_BEGIN_PS)
        & (lag_ps <= FIT_END_PS)
        & np.isfinite(msd_nm2)
    )
    x = lag_ps[use]
    y = msd_nm2[use]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    diffusion = float(slope / 2.0 * 1000.0)  # nm2/ps -> 1e-9 m2/s
    return {
        "fit_begin_ps": FIT_BEGIN_PS,
        "fit_end_ps": FIT_END_PS,
        "slope_nm2_per_ps": float(slope),
        "intercept_nm2": float(intercept),
        "r_squared": r_squared,
        "D_z_1e-9_m2_per_s": diffusion,
    }


def block_index(time_ps: float) -> int:
    fraction = (time_ps - START_PS) / (END_PS - START_PS)
    return min(N_BLOCKS - 1, max(0, int(fraction * N_BLOCKS)))


def prepare_structure_axes(index: int) -> dict[str, np.ndarray | float | str]:
    shell = SHELLS[index]
    length_z = CENTRAL_LENGTH_NM
    dz = length_z / GRID
    q_z = 2.0 * np.pi * np.fft.rfftfreq(GRID, d=dz)
    result: dict[str, np.ndarray | float | str] = {
        "kind": shell["kind"],
        "q_z": q_z,
    }
    if index != 6:
        circumference = 2.0 * np.pi * shell["r_peak_nm"]
        du = circumference / GRID
        result["circumference_nm"] = circumference
        result["q_u"] = 2.0 * np.pi * np.fft.fftfreq(GRID, d=du)
    return result


def analyze(index: int, concentration: str) -> None:
    if index not in INDICES:
        raise ValueError(f"index must be one of {INDICES}")

    top, traj, seed = system_paths(index, concentration)
    out = OUTPUT_ROOT / "systems" / f"cnt{index}_{concentration.replace('.', 'p')}"
    out.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(str(top), str(traj))
    cnt = universe.atoms[: N_CNT_WALL[index]]
    ow = universe.select_atoms("resname SOL and name OW")
    shell = SHELLS[index]
    axes = prepare_structure_axes(index)

    n_lags = MAX_LAG_FRAMES + 1
    raw_sums = np.zeros((N_BLOCKS, n_lags), dtype=np.float64)
    internal_sums = np.zeros_like(raw_sums)
    msd_counts = np.zeros((N_BLOCKS, n_lags), dtype=np.int64)
    origin_counts = np.zeros(N_BLOCKS, dtype=np.int64)
    active: list[Origin] = []

    q_z = np.asarray(axes["q_z"])
    sz_sums = np.zeros((N_BLOCKS, q_z.size), dtype=np.float64)
    structure_counts = np.zeros(N_BLOCKS, dtype=np.int64)
    if index != 6:
        q_u_all = np.asarray(axes["q_u"])
        q_u_positive = q_u_all[: GRID // 2 + 1]
        su_sums = np.zeros((N_BLOCKS, q_u_positive.size), dtype=np.float64)
    else:
        q_u_positive = np.empty(0, dtype=float)
        su_sums = np.empty((N_BLOCKS, 0), dtype=float)

    selected_counts = np.zeros(N_BLOCKS, dtype=np.float64)
    snapshot: dict[str, np.ndarray] = {}
    analysed_frames = 0
    structure_frames = 0

    first_time_ps = float(universe.trajectory[0].time)
    trajectory_dt_ps = float(universe.trajectory.dt)
    start_frame_index = int(round((START_PS - first_time_ps) / trajectory_dt_ps))
    end_frame_index = int(round((END_PS - first_time_ps) / trajectory_dt_ps))
    if start_frame_index < 0 or end_frame_index >= len(universe.trajectory):
        raise RuntimeError(
            f"Requested window is outside trajectory: first={first_time_ps}, "
            f"last={float(universe.trajectory[-1].time)} ps"
        )

    # XTC contains an offset index, so this slice seeks directly to 300 ns.
    for ts in universe.trajectory[start_frame_index : end_frame_index + 1]:
        time_ps = float(ts.time)
        if time_ps < START_PS - 1.0e-6:
            continue
        if time_ps > END_PS + 1.0e-6:
            break
        analysis_frame = int(round((time_ps - START_PS) / FRAME_DT_PS))
        block = block_index(time_ps)

        box_nm = ts.dimensions[:3].astype(np.float64) / 10.0
        cnt_nm = cnt.positions.astype(np.float64) / 10.0
        ow_nm = ow.positions.astype(np.float64) / 10.0
        center_nm = np.mean(cnt_nm, axis=0)

        cnt_dxy = cnt_nm[:, :2] - center_nm[:2]
        cnt_dxy -= box_nm[:2] * np.rint(cnt_dxy / box_nm[:2])
        radius_nm = float(np.mean(np.linalg.norm(cnt_dxy, axis=1)))
        z_min_nm = float(np.min(cnt_nm[:, 2]))
        z_max_nm = float(np.max(cnt_nm[:, 2]))

        dxy = ow_nm[:, :2] - center_nm[:2]
        dxy -= box_nm[:2] * np.rint(dxy / box_nm[:2])
        radial_nm = np.linalg.norm(dxy, axis=1)
        z_rel_nm = ow_nm[:, 2] - center_nm[2]
        inside_full = (
            (radial_nm < radius_nm)
            & (ow_nm[:, 2] >= z_min_nm)
            & (ow_nm[:, 2] <= z_max_nm)
        )

        # Residence-conditioned raw and collective-drift-corrected axial MSD.
        next_active: list[Origin] = []
        for origin in active:
            lag = analysis_frame - origin.start_frame
            if lag > MAX_LAG_FRAMES:
                continue
            origin.alive &= inside_full[origin.ow_ids]
            if np.any(origin.alive):
                dz_values = (
                    z_rel_nm[origin.ow_ids[origin.alive]]
                    - origin.z0_rel_nm[origin.alive]
                )
                n_alive = int(dz_values.size)
                raw_sums[origin.block, lag] += float(np.dot(dz_values, dz_values))
                relative = dz_values - float(np.mean(dz_values))
                internal_sums[origin.block, lag] += float(np.dot(relative, relative))
                msd_counts[origin.block, lag] += n_alive
            if lag < MAX_LAG_FRAMES and np.any(origin.alive):
                next_active.append(origin)
        active = next_active

        if analysis_frame % ORIGIN_STRIDE_FRAMES == 0 and time_ps <= END_PS - MAX_LAG_PS:
            ids = np.flatnonzero(inside_full)
            origin = Origin(
                start_frame=analysis_frame,
                block=block,
                ow_ids=ids,
                z0_rel_nm=z_rel_nm[ids].copy(),
                alive=np.ones(ids.size, dtype=bool),
            )
            origin_counts[block] += 1
            msd_counts[block, 0] += ids.size
            active.append(origin)

        # Paper-inspired structure factor every 20 ps in the central 4 nm.
        if analysis_frame % STRUCTURE_STRIDE_FRAMES == 0:
            central = np.abs(z_rel_nm) < CENTRAL_LENGTH_NM / 2.0
            layer = (
                central
                & (radial_nm >= shell["r_min_nm"])
                & (radial_nm < shell["r_max_nm"])
            )
            ids = np.flatnonzero(layer)
            if ids.size > 0:
                z_layer = z_rel_nm[ids]
                if index == 6:
                    hist, _ = np.histogram(
                        z_layer,
                        bins=GRID,
                        range=(-CENTRAL_LENGTH_NM / 2.0, CENTRAL_LENGTH_NM / 2.0),
                    )
                    fft = np.fft.rfft(hist.astype(float))
                    sz = np.abs(fft) ** 2 / ids.size
                    sz_sums[block] += sz
                else:
                    theta = np.arctan2(dxy[ids, 1], dxy[ids, 0])
                    u_layer = shell["r_peak_nm"] * theta
                    half_u = np.pi * shell["r_peak_nm"]
                    hist, _, _ = np.histogram2d(
                        u_layer,
                        z_layer,
                        bins=(GRID, GRID),
                        range=((-half_u, half_u), (-CENTRAL_LENGTH_NM / 2.0, CENTRAL_LENGTH_NM / 2.0)),
                    )
                    fft = np.fft.rfft2(hist)
                    power = np.abs(fft) ** 2 / ids.size
                    sz_sums[block] += power[0, :]
                    su_sums[block] += power[: GRID // 2 + 1, 0]

                structure_counts[block] += 1
                selected_counts[block] += ids.size
                structure_frames += 1
                snapshot = {
                    "all_dxy_nm": dxy[central & (radial_nm < radius_nm)].copy(),
                    "all_z_rel_nm": z_rel_nm[central & (radial_nm < radius_nm)].copy(),
                    "selected_dxy_nm": dxy[ids].copy(),
                    "selected_z_rel_nm": z_layer.copy(),
                    "cnt_dxy_nm": cnt_dxy.copy(),
                }
                if index != 6:
                    snapshot["selected_u_nm"] = u_layer.copy()

        analysed_frames += 1
        if analysed_frames % 10_000 == 0:
            print(
                f"i={index} {concentration}: processed {analysed_frames} frames; "
                f"time={time_ps / 1000.0:.1f} ns",
                flush=True,
            )

    if np.any(structure_counts == 0):
        raise RuntimeError(f"Empty structure block(s): {structure_counts.tolist()}")

    # MSD aggregation.
    lag_ps = np.arange(n_lags, dtype=float) * FRAME_DT_PS
    raw_blocks = np.divide(
        raw_sums,
        msd_counts,
        out=np.full_like(raw_sums, np.nan),
        where=msd_counts > 0,
    )
    internal_blocks = np.divide(
        internal_sums,
        msd_counts,
        out=np.full_like(internal_sums, np.nan),
        where=msd_counts > 0,
    )
    raw_mean = np.nansum(raw_sums, axis=0) / np.sum(msd_counts, axis=0)
    internal_mean = np.nansum(internal_sums, axis=0) / np.sum(msd_counts, axis=0)
    raw_sem = sem_rows(raw_blocks)
    internal_sem = sem_rows(internal_blocks)
    raw_fit = fit_msd(lag_ps, raw_mean)
    internal_fit = fit_msd(lag_ps, internal_mean)
    raw_block_d = np.array([fit_msd(lag_ps, x)["D_z_1e-9_m2_per_s"] for x in raw_blocks])
    internal_block_d = np.array(
        [fit_msd(lag_ps, x)["D_z_1e-9_m2_per_s"] for x in internal_blocks]
    )
    raw_fit["block_mean_D_z_1e-9_m2_per_s"] = float(np.mean(raw_block_d))
    raw_fit["block_sem_D_z_1e-9_m2_per_s"] = float(np.std(raw_block_d, ddof=1) / 2.0)
    internal_fit["block_mean_D_z_1e-9_m2_per_s"] = float(np.mean(internal_block_d))
    internal_fit["block_sem_D_z_1e-9_m2_per_s"] = float(
        np.std(internal_block_d, ddof=1) / 2.0
    )
    np.savetxt(
        out / "axial_msd_raw_and_collective_corrected.csv",
        np.column_stack([lag_ps, raw_mean, raw_sem, internal_mean, internal_sem, np.sum(msd_counts, axis=0)]),
        delimiter=",",
        header=(
            "lag_ps,msd_z_raw_nm2,raw_sem_nm2,msd_z_collective_corrected_nm2,"
            "corrected_sem_nm2,surviving_pairs"
        ),
        comments="",
    )

    # Structure-factor aggregation and Hansen-Verlet screening.
    sz_blocks = sz_sums / structure_counts[:, None]
    sz_mean = np.mean(sz_blocks, axis=0)
    sz_sem = sem_rows(sz_blocks)
    if index != 6:
        su_blocks = su_sums / structure_counts[:, None]
        su_mean = np.mean(su_blocks, axis=0)
        su_sem = sem_rows(su_blocks)
    else:
        su_blocks = np.empty((N_BLOCKS, 0))
        su_mean = np.empty(0)
        su_sem = np.empty(0)

    mask_z = (q_z >= Q_MIN_NM_INV) & (q_z <= Q_MAX_NM_INV)
    z_local = int(np.argmax(sz_mean[mask_z]))
    z_indices = np.flatnonzero(mask_z)
    z_index = int(z_indices[z_local])
    candidates = [(float(sz_mean[z_index]), float(q_z[z_index]), "axial")]
    if index != 6:
        mask_u = (q_u_positive >= Q_MIN_NM_INV) & (q_u_positive <= Q_MAX_NM_INV)
        u_local = int(np.argmax(su_mean[mask_u]))
        u_indices = np.flatnonzero(mask_u)
        u_index = int(u_indices[u_local])
        candidates.append((float(su_mean[u_index]), float(q_u_positive[u_index]), "circumferential"))
    s_max, q_peak, peak_direction = max(candidates, key=lambda value: value[0])
    phase_state = "Crystal-like" if s_max >= HANSEN_VERLET_THRESHOLD else "Liquid"

    np.savez_compressed(
        out / "structure_factor_blocks.npz",
        q_z_nm_inv=q_z,
        S_z_blocks=sz_blocks,
        S_z_mean=sz_mean,
        S_z_sem=sz_sem,
        q_u_nm_inv=q_u_positive,
        S_u_blocks=su_blocks,
        S_u_mean=su_mean,
        S_u_sem=su_sem,
    )
    np.savez_compressed(out / "representative_structure.npz", **snapshot)

    summary = {
        "CNT": f"({index},{index})",
        "CNT_index": index,
        "nominal_concentration": concentration,
        "seed": seed,
        "source_topology": str(top),
        "source_trajectory": str(traj),
        "analysed_frames": analysed_frames,
        "structure_counts_by_block": structure_counts.tolist(),
        "reference_method": "/scratch/yingyiz94/CNT_NPT/analyze_pure_water_phase_state.py",
        "analysis_window_ns": [300.0, 500.0],
        "central_structure_window_nm": CENTRAL_LENGTH_NM,
        "structure_stride_ps": STRUCTURE_STRIDE_FRAMES * FRAME_DT_PS,
        "structure_frames": int(structure_frames),
        "structure_blocks": N_BLOCKS,
        "layer_selection": shell,
        "mean_selected_OW_per_frame": float(np.sum(selected_counts) / np.sum(structure_counts)),
        "structure_factor_method": (
            "1D axial S(q_z)" if index == 6 else "2D cylindrical shell unfolded to (R_peak*theta,z); axial and circumferential cuts"
        ),
        "q_screening_range_nm_inv": [Q_MIN_NM_INV, Q_MAX_NM_INV],
        "Hansen_Verlet_reference_threshold": HANSEN_VERLET_THRESHOLD,
        "S_max": s_max,
        "q_peak_nm_inv": q_peak,
        "peak_direction": peak_direction,
        "binary_state_from_structure_factor": phase_state,
        "threshold_caveat": (
            "The 2.85 threshold is empirical for simple fluids and is used here as a paper-inspired reference; "
            "curvature and one-dimensional confinement require cautious interpretation."
        ),
        "MSD": {
            "inside_definition": "r < CNT carbon-wall-centre radius and CNT z_min <= z <= z_max",
            "continuous_residence_required": True,
            "time_origin_spacing_ps": ORIGIN_STRIDE_FRAMES * FRAME_DT_PS,
            "maximum_lag_ps": MAX_LAG_PS,
            "blocks": N_BLOCKS,
            "raw": raw_fit,
            "collective_corrected": internal_fit,
            "correction": "subtract the mean axial displacement of surviving water molecules for every origin and lag",
            "time_origins": int(np.sum(origin_counts)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"i={index} {concentration}: Smax={s_max:.5f} at q={q_peak:.3f} nm^-1 ({peak_direction}); "
        f"state={phase_state}; D_raw={raw_fit['D_z_1e-9_m2_per_s']:.5f}; "
        f"D_corrected={internal_fit['D_z_1e-9_m2_per_s']:.5f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=INDICES)
    parser.add_argument("--concentration", required=True, choices=CONCENTRATIONS)
    args = parser.parse_args()
    analyze(args.index, args.concentration)


if __name__ == "__main__":
    main()
