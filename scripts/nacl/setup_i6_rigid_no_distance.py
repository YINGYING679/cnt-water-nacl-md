#!/usr/bin/env python3
"""Build an all-rigid CNT(6,6) NaCl EM system from the 500 ns endpoint.

The target concentration follows SI Eq. C_salt = 55.5*n_pairs/n_water.
Each NaCl pair replaces two reservoir water molecules.  Output molecules are
reordered as HCS, SOL, Na, Cl so that the GRO order matches topol.top.  Ion
sites are selected randomly without a minimum-distance rejection criterion.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import MDAnalysis as mda
import numpy as np


WORKSPACE = Path("/scratch/yingyiz94")
BUILDER = WORKSPACE / "gmx_builder-0.2.0.dev0"
sys.path.insert(0, str(BUILDER))

from gmx_builder import CNTGeometry  # noqa: E402


SOURCE_ROOT = WORKSPACE / "CNT_NPT/cnt_6_6"
SOURCE_GRO = (
    SOURCE_ROOT
    / "NPT/continue_20ns_to_500ns/npt_20ns_to_500ns.part0002.gro"
)
DEST_ROOT = WORKSPACE / "CNT_NPT_NaCl_rigid_only/cnt_6_6"

N_HCS_ATOMS = 5694
N_WATER_INITIAL = 12372

# Concentration label: (target concentration in M, number of NaCl pairs).
SYSTEMS = {
    "0.2M": (0.2, 44),
    "1M": (1.0, 215),
    "2M": (2.0, 416),
    "3M": (3.0, 604),
    "4M": (4.0, 779),
    "5M": (5.0, 944),
}

SPCE_TEMPLATE = """; Rigid SPC/E water with SETTLE.

[ moleculetype ]
; molname  nrexcl
SOL        2

[ atoms ]
; nr  type  resnr  residue  atom  cgnr  charge   mass
  1   OW      1      SOL     OW      1  -0.8476  15.999
  2   HW      1      SOL     HW1     1   0.4238   1.008
  3   HW      1      SOL     HW2     1   0.4238   1.008

[ settles ]
; OW  funct  dOH (nm)  dHH (nm)
  1     1      0.1000    0.16330

[ exclusions ]
1  2  3
2  1  3
3  1  2
"""


TOP_TEMPLATE = """; CNT(6,6) reservoir-water NaCl topology.
; Ion parameters follow SI Table S3 (Duenas-Herrera et al.).

[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
  1       2          no         1.0      1.0

[ atomtypes ]
; name  at.num  mass     charge  ptype  sigma       epsilon
  C      6      12.011   0.000   A      0.3568      0.2966
  HW     1       1.008   0.000   A      0.0000      0.0000
  OW     8      15.999   0.000   A      0.3166      0.6500
  Na    11      22.990   0.000   A      0.2310      0.4500
  Cl    17      35.450   0.000   A      0.4300      0.4200

[ nonbond_params ]
; i   j   func  sigma   epsilon
  C   OW   1    0.3367  0.4391
  C   C    1    0.0     0.0

[ moleculetype ]
; molname  nrexcl
Na          1

[ atoms ]
; nr  type  resnr  residue  atom  cgnr  charge  mass
  1   Na      1      Na      Na     1     1.0   22.990

[ moleculetype ]
; molname  nrexcl
Cl          1

[ atoms ]
; nr  type  resnr  residue  atom  cgnr  charge  mass
  1   Cl      1      Cl      Cl     1    -1.0   35.450

#include "honeycombstructure.itp"
#include "spce.itp"

[ system ]
CNT(6,6) reservoir water with NaCl at {target:g} M nominal concentration

[ molecules ]
; Molecule  count
HCS         1
SOL         {nwater}
Na          {npairs}
Cl          {npairs}
"""


def write_index(path: Path, nwater: int, npairs: int) -> None:
    first_fluid = N_HCS_ATOMS + 1
    last_water = N_HCS_ATOMS + 3 * nwater
    last_na = last_water + npairs
    last_cl = last_na + npairs
    groups = {
        "System": range(1, last_cl + 1),
        "HCS": range(1, N_HCS_ATOMS + 1),
        "Water": range(first_fluid, last_water + 1),
        "Na": range(last_water + 1, last_na + 1),
        "Cl": range(last_na + 1, last_cl + 1),
        "Water_Na_Cl": range(first_fluid, last_cl + 1),
    }
    with path.open("w") as fh:
        for name, indices in groups.items():
            fh.write(f"[ {name} ]\n")
            row = []
            for atom_index in indices:
                row.append(f"{atom_index:6d}")
                if len(row) == 15:
                    fh.write("".join(row) + "\n")
                    row = []
            if row:
                fh.write("".join(row) + "\n")


def gro_line(line: str, atom_number: int, resname: str | None = None,
             atomname: str | None = None) -> str:
    old_resname = line[5:10].strip()
    old_atomname = line[10:15].strip()
    return (
        line[:5]
        + f"{(resname or old_resname):>5}"
        + f"{(atomname or old_atomname):>5}"
        + f"{atom_number:5d}"
        + line[20:]
    )


def build_salted_gro(out_path: Path, npairs: int, seed: int) -> tuple[int, int]:
    lines = SOURCE_GRO.read_text().splitlines()
    natoms = int(lines[1])
    atom_lines = lines[2:2 + natoms]
    box_line = lines[2 + natoms]

    universe = mda.Universe(str(SOURCE_GRO))
    geometry = CNTGeometry(str(SOURCE_GRO))
    reservoir_resindices = np.unique(
        geometry.get_solvents_in_reservoir().resindices
    )
    if 2 * npairs > len(reservoir_resindices):
        raise ValueError("Not enough reservoir waters for requested ions")

    chosen = np.random.default_rng(seed).choice(
        reservoir_resindices, size=2 * npairs, replace=False
    )
    na_resindices = set(map(int, chosen[:npairs]))
    cl_resindices = set(map(int, chosen[npairs:]))
    selected = na_resindices | cl_resindices

    carbon_records: list[tuple[str, str | None, str | None]] = []
    water_records: list[tuple[str, str | None, str | None]] = []
    na_records: list[tuple[str, str | None, str | None]] = []
    cl_records: list[tuple[str, str | None, str | None]] = []

    for atom, line in zip(universe.atoms, atom_lines):
        resindex = int(atom.resindex)
        if resindex in selected:
            if atom.name != "OW":
                continue
            if resindex in na_resindices:
                na_records.append((line, "Na", "Na"))
            else:
                cl_records.append((line, "Cl", "Cl"))
        elif atom.index < N_HCS_ATOMS:
            carbon_records.append((line, None, None))
        else:
            water_records.append((line, None, None))

    nwater = len(water_records) // 3
    records = carbon_records + water_records + na_records + cl_records
    expected_atoms = N_HCS_ATOMS + 3 * nwater + 2 * npairs
    if len(records) != expected_atoms:
        raise RuntimeError("Unexpected atom count while constructing salted GRO")

    with out_path.open("w") as fh:
        fh.write(f"CNT(6,6) NaCl: {npairs} pairs inserted in reservoirs\n")
        fh.write(f"{len(records):5d}\n")
        for atom_number, (line, resname, atomname) in enumerate(records, 1):
            fh.write(gro_line(line, atom_number, resname, atomname) + "\n")
        fh.write(box_line + "\n")

    check = mda.Universe(str(out_path))
    names = Counter(check.atoms.names)
    if names["Na"] != npairs or names["Cl"] != npairs:
        raise RuntimeError("Ion count verification failed")
    if names["OW"] != nwater or names["HW1"] != nwater or names["HW2"] != nwater:
        raise RuntimeError("Water count verification failed")
    if not np.all(check.atoms[:N_HCS_ATOMS].names == "C"):
        raise RuntimeError("HCS atom ordering verification failed")
    return nwater, len(records)


def em_run_script(label: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=cnt66_{label}_em
#SBATCH --output=em_%j.out
#SBATCH --error=em_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --partition=main

source /etc/profile
set -euo pipefail
module purge
module load gromacs/single/cpu/2025.1

EM_DIR="${{SLURM_SUBMIT_DIR}}"
ROOT_DIR="$(dirname "${{EM_DIR}}")"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-6}}"
cd "${{ROOT_DIR}}"

gmx grompp -f "${{EM_DIR}}/em.mdp" -c "${{ROOT_DIR}}/solvated_ions.gro" \\
    -p "${{ROOT_DIR}}/topol.top" -n "${{ROOT_DIR}}/index.ndx" \\
    -o "${{EM_DIR}}/em.tpr" -po "${{EM_DIR}}/em_mdout.mdp"
gmx mdrun -s "${{EM_DIR}}/em.tpr" -deffnm "${{EM_DIR}}/em" \\
    -ntmpi 1 -ntomp "${{OMP_NUM_THREADS}}"
"""


def npt_run_script(label: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=cnt66_{label}_20ns
#SBATCH --output=npt_%j.out
#SBATCH --error=npt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

source /etc/profile
set -euo pipefail
module purge
module load gromacs/single/cuda/2025.1

NPT_DIR="${{SLURM_SUBMIT_DIR}}"
ROOT_DIR="$(cd "${{NPT_DIR}}/../.." && pwd)"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_GPU:-6}}"
cd "${{ROOT_DIR}}"

for required in "${{ROOT_DIR}}/EM/em.gro" "${{ROOT_DIR}}/topol.top" \\
    "${{ROOT_DIR}}/index.ndx" "${{NPT_DIR}}/npt.mdp"; do
    [[ -f "${{required}}" ]] || {{ echo "ERROR: missing ${{required}}" >&2; exit 1; }}
done

if [[ -f "${{NPT_DIR}}/npt.cpt" && -f "${{NPT_DIR}}/npt.tpr" ]]; then
    echo "Restarting from ${{NPT_DIR}}/npt.cpt"
    gmx mdrun -deffnm "${{NPT_DIR}}/npt" -cpi "${{NPT_DIR}}/npt.cpt" \\
        -append -nb gpu -pme gpu -ntmpi 1 -ntomp "${{OMP_NUM_THREADS}}" \\
        -cpt 15 -maxh 3.5
else
    gmx grompp -f "${{NPT_DIR}}/npt.mdp" -c "${{ROOT_DIR}}/EM/em.gro" \\
        -p "${{ROOT_DIR}}/topol.top" -n "${{ROOT_DIR}}/index.ndx" \\
        -o "${{NPT_DIR}}/npt.tpr" -po "${{NPT_DIR}}/npt_mdout.mdp"
    gmx mdrun -deffnm "${{NPT_DIR}}/npt" -nb gpu -pme gpu \\
        -ntmpi 1 -ntomp "${{OMP_NUM_THREADS}}" -cpt 15 -maxh 3.5
fi
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one rigid-SPC/E CNT(6,6) NaCl EM system."
    )
    parser.add_argument("concentration", choices=SYSTEMS)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_universe = mda.Universe(str(SOURCE_GRO))
    if len(source_universe.select_atoms("name OW")) != N_WATER_INITIAL:
        raise RuntimeError("Unexpected source water count")

    target, npairs = SYSTEMS[args.concentration]
    root = DEST_ROOT / f"NaCl_{args.concentration}_no_distance_seed{args.seed}"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing system: {root}")
    em_dir = root / "EM"
    em_dir.mkdir(parents=True)

    shutil.copy2(SOURCE_GRO, root / "pure_water_500ns.gro")
    shutil.copy2(SOURCE_ROOT / "honeycombstructure.itp", root)
    (root / "spce.itp").write_text(SPCE_TEMPLATE)

    nwater, natoms = build_salted_gro(
        root / "solvated_ions.gro", npairs, args.seed
    )
    realized = 55.5 * npairs / nwater
    (root / "topol.top").write_text(
        TOP_TEMPLATE.format(target=target, nwater=nwater, npairs=npairs)
    )
    write_index(root / "index.ndx", nwater, npairs)

    # Keep the original pure-water rigid EM parameters.  Remove the inactive
    # preprocessor example so the generated input has no alternative water model.
    em_mdp = "\n".join(
        line
        for line in (SOURCE_ROOT / "EM/em.mdp").read_text().splitlines()
        if not line.lstrip().startswith(";define")
    ) + "\n"
    (em_dir / "em.mdp").write_text(em_mdp)
    run_file = em_dir / "run.sh"
    label = f"nacl_{args.concentration.lower().replace('.', 'p')}_nodist_s{args.seed}"
    run_file.write_text(
        em_run_script(label).replace(
            "#SBATCH --partition=main", "#SBATCH --partition=test"
        )
    )
    run_file.chmod(0o755)

    (root / "parameters.txt").write_text(
        "CNT = (6,6)\n"
        f"source_gro = {SOURCE_GRO}\n"
        f"initial_water = {N_WATER_INITIAL}\n"
        f"target_concentration_M = {target:g}\n"
        f"NaCl_pairs = {npairs}\n"
        f"Na_ions = {npairs}\n"
        f"Cl_ions = {npairs}\n"
        f"remaining_water = {nwater}\n"
        f"realized_concentration_M = {realized:.6f}\n"
        "placement = random reservoir-water replacement without geometric rejection\n"
        "distance_screening = none\n"
        f"random_seed = {args.seed}\n"
        f"total_atoms = {natoms}\n"
        "water_model = rigid SPC/E with SETTLE\n"
        "water_geometry_during_EM = rigid\n"
        "EM_emtol_kJ_mol_nm = 1000\n"
        "EM_emstep_nm = 0.001\n"
        "EM_nsteps = 100000\n"
        "Na_sigma_nm = 0.2310\nNa_epsilon_kJ_mol = 0.450\n"
        "Cl_sigma_nm = 0.4300\nCl_epsilon_kJ_mol = 0.420\n"
    )
    print(root)
    print(
        f"water={nwater} Na={npairs} Cl={npairs} atoms={natoms} "
        f"realized={realized:.6f} M"
    )


if __name__ == "__main__":
    main()
