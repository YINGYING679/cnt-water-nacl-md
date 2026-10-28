#!/usr/bin/env python3
"""Concentration-only Na/Cl partition analysis for newly added CNT radii.

The definitions exactly match the concentration part of
analyze_saltwater_ion_C_D_300_500ns.py, while omitting the costly MSD work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
import numpy as np


ROOT = Path("/scratch/yingyiz94/CNT_NPT_NaCl_rigid_only")
OUT_ROOT = ROOT / "saltwater_ion_partition_new_radii_300_500ns"
INDICES = (9, 10, 40)
CONCENTRATIONS = ("0.2M", "1M", "2M", "3M", "4M", "5M")
SPECIES = ("Na", "Cl")
START_PS = 300_000.0
END_PS = 500_000.0
FRAME_DT_PS = 2.0
STRIDE = 10
BLOCK_PS = 50_000.0
N_BLOCKS = 4
CENTRAL_LENGTH_NM = 4.0
BULK_HALF_WIDTH_NM = 1.0
PARTICLES_PER_NM3_PER_M = 0.602214076


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def system_paths(index: int, concentration: str) -> tuple[Path, Path, int]:
    candidates = sorted(
        (ROOT / f"cnt_{index}_{index}").glob(
            f"NaCl_{concentration}_no_distance_seed*/NPT/continue_20ns_to_500ns"
        )
    )
    candidates = [
        path
        for path in candidates
        if (path / "npt_20ns_to_500ns.part0002.gro").is_file()
        and (path / "npt_20ns_to_500ns.part0002.xtc").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"CNT({index},{index}) {concentration}: expected one trajectory, "
            f"found {len(candidates)}"
        )
    base = candidates[0]
    seed = int(base.parts[-3].rsplit("seed", 1)[1])
    return (
        base / "npt_20ns_to_500ns.part0002.gro",
        base / "npt_20ns_to_500ns.part0002.xtc",
        seed,
    )


def reservoir_geometry(
    cnt_positions_nm: np.ndarray, box_nm: np.ndarray
) -> tuple[float, float]:
    cnt_z = np.mod(cnt_positions_nm[:, 2], box_nm[2])
    z_min = float(np.min(cnt_z))
    z_max = float(np.max(cnt_z))
    gap = float(box_nm[2] - (z_max - z_min))
    midpoint = float(np.mod(0.5 * (z_max + z_min + box_nm[2]), box_nm[2]))
    return midpoint, gap


def bulk_mask(
    positions_nm: np.ndarray, box_nm: np.ndarray, midpoint_z_nm: float
) -> np.ndarray:
    z = np.mod(positions_nm[:, 2], box_nm[2])
    dz = z - midpoint_z_nm
    dz -= box_nm[2] * np.rint(dz / box_nm[2])
    return np.abs(dz) < BULK_HALF_WIDTH_NM


def analyse(index: int, concentration: str) -> None:
    topology, trajectory, seed = system_paths(index, concentration)
    token = concentration.replace(".", "p")
    out = OUT_ROOT / "systems" / f"cnt{index}_{token}"
    out.mkdir(parents=True, exist_ok=True)
    label = f"CNT({index},{index}) {concentration}"

    universe = mda.Universe(str(topology))
    n_cnt = 100 * index
    cnt_indices = universe.atoms[:n_cnt].indices
    ion_indices = {
        species: universe.select_atoms(f"name {species}").indices
        for species in SPECIES
    }
    if cnt_indices.size != n_cnt or any(ids.size == 0 for ids in ion_indices.values()):
        raise RuntimeError(f"{label}: unexpected CNT/ion selection")

    frame_counts = np.zeros(N_BLOCKS, dtype=np.int64)
    cnt_count_sums = {s: np.zeros(N_BLOCKS, dtype=float) for s in SPECIES}
    bulk_count_sums = {s: np.zeros(N_BLOCKS, dtype=float) for s in SPECIES}
    bulk_volume_sums = np.zeros(N_BLOCKS, dtype=float)
    first_radius_nm: float | None = None
    first_geometry: dict[str, object] | None = None
    sampled_frames = 0
    last_sampled_time_ps = float("nan")

    print(f"{label}: reading 300--500 ns", flush=True)
    with XTCFile(str(trajectory), mode="r") as xtc:
        if xtc.n_atoms != universe.atoms.n_atoms:
            raise RuntimeError(f"{label}: topology/XTC atom-count mismatch")
        for frame in xtc:
            time_ps = float(frame.time)
            if time_ps < START_PS - 1.0e-4:
                continue
            if time_ps > END_PS + 1.0e-4:
                break
            frame_number = int(round(time_ps / FRAME_DT_PS))
            if frame_number % STRIDE != 0:
                continue

            xyz_nm = np.asarray(frame.x, dtype=np.float64)
            box_nm = np.diag(np.asarray(frame.box, dtype=np.float64))
            cnt_nm = xyz_nm[cnt_indices]
            center_nm = np.mean(cnt_nm, axis=0)
            cnt_dxy_nm = minimum_image(cnt_nm[:, :2] - center_nm[:2], box_nm[:2])
            radius_nm = float(np.mean(np.linalg.norm(cnt_dxy_nm, axis=1)))
            reservoir_midpoint_nm, reservoir_gap_nm = reservoir_geometry(cnt_nm, box_nm)

            if first_radius_nm is None:
                first_radius_nm = radius_nm
                first_geometry = {
                    "box_nm": box_nm.tolist(),
                    "CNT_radius_nm": radius_nm,
                    "periodic_reservoir_gap_nm": reservoir_gap_nm,
                    "reservoir_midpoint_z_nm": reservoir_midpoint_nm,
                    "bulk_slab_half_width_nm": BULK_HALF_WIDTH_NM,
                }

            block = min(N_BLOCKS - 1, int((time_ps - START_PS) // BLOCK_PS))
            for species in SPECIES:
                ion_nm = xyz_nm[ion_indices[species]]
                dxy_nm = minimum_image(ion_nm[:, :2] - center_nm[:2], box_nm[:2])
                radial_nm = np.linalg.norm(dxy_nm, axis=1)
                dz_nm = minimum_image(ion_nm[:, 2] - center_nm[2], box_nm[2])
                central = (
                    (np.abs(dz_nm) < CENTRAL_LENGTH_NM / 2.0)
                    & (radial_nm < radius_nm)
                )
                cnt_count_sums[species][block] += np.count_nonzero(central)
                bulk_count_sums[species][block] += np.count_nonzero(
                    bulk_mask(ion_nm, box_nm, reservoir_midpoint_nm)
                )
            bulk_volume_sums[block] += (
                2.0 * BULK_HALF_WIDTH_NM * box_nm[0] * box_nm[1]
            )
            frame_counts[block] += 1
            sampled_frames += 1
            last_sampled_time_ps = time_ps
            if sampled_frames % 2500 == 0:
                print(
                    f"{label}: sampled {sampled_frames} frames, "
                    f"time={time_ps / 1000.0:.1f} ns",
                    flush=True,
                )

    if first_radius_nm is None or first_geometry is None:
        raise RuntimeError(f"{label}: no frames analysed")
    if np.any(frame_counts == 0):
        raise RuntimeError(f"{label}: empty concentration block")
    if not math.isclose(last_sampled_time_ps, END_PS, abs_tol=1.0e-3):
        raise RuntimeError(f"{label}: final sampled time={last_sampled_time_ps} ps")

    cnt_volume_nm3 = math.pi * first_radius_nm**2 * CENTRAL_LENGTH_NM
    cnt_c = {
        s: cnt_count_sums[s] / frame_counts / cnt_volume_nm3 / PARTICLES_PER_NM3_PER_M
        for s in SPECIES
    }
    bulk_c = {
        s: bulk_count_sums[s] / bulk_volume_sums / PARTICLES_PER_NM3_PER_M
        for s in SPECIES
    }
    salt_bulk = 0.5 * (bulk_c["Na"] + bulk_c["Cl"])
    k = {s: cnt_c[s] / bulk_c[s] for s in SPECIES}

    block_path = out / "concentration_blocks.csv"
    with block_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "block", "start_ns", "end_ns", "sampled_frames",
                "CNa_CNT_M", "CCl_CNT_M", "Csalt_CNT_M",
                "CNa_bulk_M", "CCl_bulk_M", "Csalt_bulk_M", "K_Na", "K_Cl",
            ]
        )
        for block in range(N_BLOCKS):
            writer.writerow(
                [
                    block + 1,
                    300.0 + 50.0 * block,
                    350.0 + 50.0 * block,
                    int(frame_counts[block]),
                    cnt_c["Na"][block],
                    cnt_c["Cl"][block],
                    0.5 * (cnt_c["Na"][block] + cnt_c["Cl"][block]),
                    bulk_c["Na"][block],
                    bulk_c["Cl"][block],
                    salt_bulk[block],
                    k["Na"][block],
                    k["Cl"][block],
                ]
            )

    summary = {
        "CNT": f"({index},{index})",
        "CNT_index": index,
        "nominal_concentration": concentration,
        "seed": seed,
        "analysis_window_ns": [300.0, 500.0],
        "sample_spacing_ps": FRAME_DT_PS * STRIDE,
        "sampled_frames": sampled_frames,
        "geometry_at_300ns": first_geometry,
        "definitions": {
            "CNT_region": "central 4.0 nm cylinder, r < carbon-wall-centre radius",
            "bulk_region": "2.0 nm full-cross-section slab centred in periodic reservoir gap",
            "blocks": "four contiguous 50 ns blocks",
        },
        "CNa_CNT_M": float(np.mean(cnt_c["Na"])),
        "CNa_CNT_SEM_M": sem(cnt_c["Na"]),
        "CCl_CNT_M": float(np.mean(cnt_c["Cl"])),
        "CCl_CNT_SEM_M": sem(cnt_c["Cl"]),
        "CNa_bulk_M": float(np.mean(bulk_c["Na"])),
        "CNa_bulk_SEM_M": sem(bulk_c["Na"]),
        "CCl_bulk_M": float(np.mean(bulk_c["Cl"])),
        "CCl_bulk_SEM_M": sem(bulk_c["Cl"]),
        "Csalt_bulk_M": float(np.mean(salt_bulk)),
        "Csalt_bulk_SEM_M": sem(salt_bulk),
        "K_Na": float(np.mean(k["Na"])),
        "K_Na_SEM": sem(k["Na"]),
        "K_Cl": float(np.mean(k["Cl"])),
        "K_Cl_SEM": sem(k["Cl"]),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{label}: complete; R={first_radius_nm:.6f} nm; "
        f"K_Na={summary['K_Na']:.6f}; K_Cl={summary['K_Cl']:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=INDICES)
    parser.add_argument("--concentration", required=True, choices=CONCENTRATIONS)
    args = parser.parse_args()
    analyse(args.index, args.concentration)


if __name__ == "__main__":
    main()
