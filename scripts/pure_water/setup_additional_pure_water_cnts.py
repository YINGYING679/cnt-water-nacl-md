#!/usr/bin/env python3
"""Build fixed-box pure-water CNT reservoir systems matching the existing series.

The established CNT(8,8), CNT(12,12), CNT(22,22), and CNT(30,30)
systems share one 8.094 x 8.36234 x 12.12298 nm box and the same 12,374
reservoir-water coordinates.  This script extends that construction to new
armchair indices without changing the reservoir or force-field settings.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
BUILDER_SOURCE = WORKSPACE / "gmx_builder-0.2.0.dev0"
sys.path.insert(0, str(BUILDER_SOURCE))

from gmx_builder.modules.honeycomb_structure import (  # noqa: E402
    armcnt,
    ccbond,
    graphite,
    write_gro,
    write_itp,
)


TEMPLATE = ROOT / "cnt_30_30"
WATER_TEMPLATE = ROOT / "cnt_8_8" / "out.gro"
INDICES = (9, 10, 40)
REQUESTED_LENGTH_A = 60.0
RESERVOIR_A = 30.0
SHEET_SIZE_A = 80.0
N_WATER = 12_374


def read_water_records(path: Path) -> list[list[object]]:
    lines = path.read_text().splitlines()
    n_atoms = int(lines[1])
    records: list[list[object]] = []
    for line in lines[2 : 2 + n_atoms]:
        residue_number = int(line[0:5])
        residue_name = line[5:10].strip()
        if residue_name != "SOL":
            continue
        atom_name = line[10:15].strip()
        x_a = float(line[20:28]) * 10.0
        y_a = float(line[28:36]) * 10.0
        z_a = float(line[36:44]) * 10.0
        records.append(
            [atom_name, x_a, y_a, z_a, residue_number, residue_name]
        )
    if len(records) != 3 * N_WATER:
        raise RuntimeError(
            f"Expected {3 * N_WATER} template water atoms, found {len(records)}"
        )
    return records


def write_index(path: Path, n_carbon: int, n_atoms: int) -> None:
    def group(name: str, indices: range) -> str:
        values = list(indices)
        rows = [" ".join(str(v) for v in values[i : i + 15])
                for i in range(0, len(values), 15)]
        return f"[ {name} ]\n" + "\n".join(rows) + "\n"

    path.write_text(
        group("System", range(1, n_atoms + 1))
        + group("HCS", range(1, n_carbon + 1))
        + group("Water", range(n_carbon + 1, n_atoms + 1))
    )


def write_parameters(
    path: Path,
    index: int,
    radius_a: float,
    length_a: float,
    x_a: float,
    y_a: float,
    z_a: float,
    n_carbon: int,
) -> None:
    x_nm, y_nm, z_nm = x_a / 10.0, y_a / 10.0, z_a / 10.0
    radius_nm, length_nm = radius_a / 10.0, length_a / 10.0
    reservoir_nm = RESERVOIR_A / 10.0
    box_volume = x_nm * y_nm * z_nm
    channel_volume = math.pi * radius_nm**2 * length_nm
    cap_volume = 2.0 * x_nm * y_nm * reservoir_nm
    accessible_volume = channel_volume + cap_volume
    path.write_text(
        f"{index}\t\t# n unit cells\n"
        f"{radius_a:.3f}\t# nanotube radius       (Angstrom)\n"
        f"{length_a:.3f}\t# nanotube length       (Angstrom)\n"
        f"{RESERVOIR_A:.3f}\t# cap thickness         (Angstrom)\n"
        f"{math.pi * radius_a**2:.3f}\t# tube cross-section   (Angstrom^2)\n"
        f"{x_a:.3f}\t# box x-width           (Angstrom)\n"
        f"{y_a:.3f}\t# box y-width           (Angstrom)\n"
        f"{z_a:.3f}\t# box z-width           (Angstrom)\n\n"
        f"{n_carbon}\t# number of C atoms\n"
        f"{N_WATER}\t# number of water molecules\n\n"
        f"{box_volume:.3f}\t# box volume           (nm^3)\n"
        f"{channel_volume:.3f}\t# channel volume       (nm^3)\n"
        f"{cap_volume:.3f}\t# caps/slabs volume    (nm^3)\n"
        f"{accessible_volume:.3f}\t# accessible volume    (nm^3)\n\n"
        f"{N_WATER / box_volume:.3f}\t# global water number density     (molecules/nm^3)\n"
        f"{N_WATER / accessible_volume:.3f}\t# accessible water number density (molecules/nm^3)\n"
    )


def copy_run_tree(destination: Path, index: int) -> None:
    (destination / "EM").mkdir()
    (destination / "NPT" / "test_0_20ns").mkdir(parents=True)
    (destination / "NPT" / "continue_20ns_to_500ns").mkdir(parents=True)

    sources = {
        TEMPLATE / "EM" / "em.mdp": destination / "EM" / "em.mdp",
        TEMPLATE / "EM" / "run.sh": destination / "EM" / "run.sh",
        TEMPLATE / "NPT" / "test_0_20ns" / "npt.mdp": destination / "NPT" / "test_0_20ns" / "npt.mdp",
        TEMPLATE / "NPT" / "test_0_20ns" / "run.sh": destination / "NPT" / "test_0_20ns" / "run.sh",
        TEMPLATE / "NPT" / "continue_20ns_to_500ns" / "npt.mdp": destination / "NPT" / "continue_20ns_to_500ns" / "npt.mdp",
        TEMPLATE / "NPT" / "continue_20ns_to_500ns" / "run.sh": destination / "NPT" / "continue_20ns_to_500ns" / "run.sh",
    }
    for source, target in sources.items():
        text = source.read_text()
        text = text.replace("cnt3030", f"cnt{index}{index}")
        text = text.replace("(30,30)", f"({index},{index})")
        target.write_text(text)
        if target.suffix == ".sh":
            target.chmod(0o755)


def build_one(index: int, force: bool, water_records: list[list[object]]) -> None:
    destination = ROOT / f"cnt_{index}_{index}"
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists; use --force to replace it")
        shutil.rmtree(destination)
    destination.mkdir()

    cnt, radius_a, length_a = armcnt(
        index, REQUESTED_LENGTH_A, elements=("C", "C")
    )
    cnt = [atom + [1, "CNT"] for atom in cnt]

    cap_bottom, x_a, y_a = graphite(
        SHEET_SIZE_A, SHEET_SIZE_A, length_a, elements=("C", "C")
    )
    cap_top = graphite(SHEET_SIZE_A, SHEET_SIZE_A, elements=("C", "C"))[0]
    caps = [atom + [2, "HCS"] for atom in cap_bottom + cap_top]

    for atom in cnt:
        atom[1] += x_a / 2.0
        atom[2] += y_a / 2.0

    opening_radius_a = radius_a + ccbond
    caps = [
        atom
        for atom in caps
        if (atom[1] - x_a / 2.0) ** 2 + (atom[2] - y_a / 2.0) ** 2
        > opening_radius_a**2
    ]

    carbon_records = cnt + caps
    for atom in carbon_records:
        atom[3] += length_a + RESERVOIR_A

    z_a = length_a + 2.0 * RESERVOIR_A
    all_records = carbon_records + water_records
    gro_path = destination / "out.gro"
    write_gro(str(gro_path), all_records, x_a, y_a, z_a)
    gro_lines = gro_path.read_text().splitlines()
    gro_lines[0] = (
        f"CNT({index},{index}) caps-water; common XY box; "
        f"12374-water template from CNT(8,8)"
    )
    gro_path.write_text("\n".join(gro_lines) + "\n")

    write_itp(str(destination / "honeycombstructure.itp"), carbon_records)
    shutil.copy2(TEMPLATE / "spce.itp", destination / "spce.itp")

    topology = (TEMPLATE / "topol.top").read_text()
    topology = topology.replace("(30,30)", f"({index},{index})")
    (destination / "topol.top").write_text(topology)

    n_carbon = len(carbon_records)
    n_atoms = n_carbon + len(water_records)
    write_index(destination / "index.ndx", n_carbon, n_atoms)
    write_parameters(
        destination / "parameters.txt",
        index,
        radius_a,
        length_a,
        x_a,
        y_a,
        z_a,
        n_carbon,
    )
    copy_run_tree(destination, index)

    (destination / "README.md").write_text(
        f"# Pure-water CNT({index},{index})\n\n"
        "This system extends the fixed-box pure-water armchair CNT series. "
        "It uses the same 8.094 x 8.36234 x 12.12298 nm box, 6.123 nm "
        "CNT length, two 3.0 nm reservoirs, 12,374 initial reservoir waters, "
        "SPC/E model, carbon-water parameters, and frozen carbon framework as "
        "CNT(8,8), CNT(12,12), CNT(22,22), and CNT(30,30).\n\n"
        "Workflow: EM -> 0-20 ns NPT -> dependent 20-500 ns NPT.\n"
    )
    print(
        f"CNT({index},{index}): R={radius_a / 10:.4f} nm, "
        f"C={n_carbon}, water={N_WATER}, atoms={n_atoms}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, default=list(INDICES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    water_records = read_water_records(WATER_TEMPLATE)
    for index in args.indices:
        build_one(index, args.force, water_records)


if __name__ == "__main__":
    main()
