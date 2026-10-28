#!/usr/bin/env python3
"""Prepare 20-to-500 ns NPT continuations from completed 20 ns checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/scratch/yingyiz94/CNT_NPT_NaCl_rigid_only")
MDP_TEMPLATE = (
    ROOT
    / "cnt_6_6/NaCl_0.2M_no_distance_seed17/NPT/continue_20ns_to_500ns/npt.mdp"
)
CONCENTRATIONS = ("0.2M", "1M", "2M", "3M", "4M", "5M")


def system_dir(cnt: int, concentration: str) -> Path:
    seed = 18 if cnt == 22 and concentration == "3M" else 17
    return ROOT / f"cnt_{cnt}_{cnt}/NaCl_{concentration}_no_distance_seed{seed}"


def run_script(cnt: int, concentration: str) -> str:
    label = concentration.replace(".", "p").lower()
    return f"""#!/bin/bash
#SBATCH --job-name=cnt{cnt}{cnt}_nacl_{label}_500ns
#SBATCH --output=npt_500ns_%j.out
#SBATCH --error=npt_500ns_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu

source /etc/profile
set -euo pipefail

module purge
module load gromacs/single/cuda/2025.1

SEGMENT_DIR="${{SLURM_SUBMIT_DIR}}"
NPT_DIR="$(dirname "${{SEGMENT_DIR}}")"
ROOT_DIR="$(dirname "${{NPT_DIR}}")"
PARENT_DIR="${{NPT_DIR}}/test_0_20ns"
OUTPUT_BASE="${{SEGMENT_DIR}}/npt_20ns_to_500ns"
TPR="${{OUTPUT_BASE}}.tpr"

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_GPU:-6}}"

run_npt() {{
    gmx mdrun \
        -s "${{TPR}}" \
        -deffnm "${{OUTPUT_BASE}}" \
        -nb gpu \
        -pme gpu \
        -ntmpi 1 \
        -ntomp "${{OMP_NUM_THREADS}}" \
        -cpt 15 \
        -maxh 95.5 \
        "$@"
}}

for required in \
    "${{SEGMENT_DIR}}/npt.mdp" \
    "${{PARENT_DIR}}/npt.gro" \
    "${{PARENT_DIR}}/npt.cpt" \
    "${{ROOT_DIR}}/topol.top" \
    "${{ROOT_DIR}}/index.ndx" \
    "${{ROOT_DIR}}/honeycombstructure.itp" \
    "${{ROOT_DIR}}/spce.itp"; do
    if [[ ! -f "${{required}}" ]]; then
        echo "ERROR: missing required file: ${{required}}" >&2
        exit 1
    fi
done

if [[ -f "${{OUTPUT_BASE}}.cpt" ]]; then
    if [[ ! -f "${{TPR}}" ]]; then
        echo "ERROR: continuation checkpoint exists but ${{TPR}} is missing." >&2
        exit 1
    fi
    echo "Continuing the 20-to-500 ns segment from its own checkpoint."
    run_npt -cpi "${{OUTPUT_BASE}}.cpt" -append
else
    echo "Preparing the 20-to-500 ns segment from the parent 20 ns checkpoint."
    gmx grompp \
        -f "${{SEGMENT_DIR}}/npt.mdp" \
        -c "${{PARENT_DIR}}/npt.gro" \
        -t "${{PARENT_DIR}}/npt.cpt" \
        -p "${{ROOT_DIR}}/topol.top" \
        -n "${{ROOT_DIR}}/index.ndx" \
        -o "${{TPR}}" \
        -po "${{SEGMENT_DIR}}/npt_mdout.mdp"

    run_npt -cpi "${{PARENT_DIR}}/npt.cpt" -noappend
fi
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cnt", type=int, choices=(8, 9, 10, 12, 22, 30, 40), required=True
    )
    args = parser.parse_args()

    if not MDP_TEMPLATE.is_file():
        raise FileNotFoundError(MDP_TEMPLATE)
    template = MDP_TEMPLATE.read_text().replace("(6,6)", f"({args.cnt},{args.cnt})")

    for concentration in CONCENTRATIONS:
        root = system_dir(args.cnt, concentration)
        parent = root / "NPT/test_0_20ns"
        required = (
            parent / "npt.gro",
            parent / "npt.cpt",
            root / "topol.top",
            root / "index.ndx",
            root / "honeycombstructure.itp",
            root / "spce.itp",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required files: " + ", ".join(missing))

        segment = root / "NPT/continue_20ns_to_500ns"
        if any(segment.glob("npt_20ns_to_500ns*")):
            raise FileExistsError(f"Continuation output already exists: {segment}")
        segment.mkdir(parents=True, exist_ok=True)
        (segment / "npt.mdp").write_text(template)
        script = segment / "run.sh"
        script.write_text(run_script(args.cnt, concentration))
        script.chmod(0o755)
        print(segment)


if __name__ == "__main__":
    main()
