#!/bin/bash
# Train a detector on a SLURM cluster, surviving pre-emption and the wall clock.
#
#   sbatch scripts/slurm/train.sbatch experiment=nuscenes
#   sbatch scripts/slurm/train.sbatch experiment=nuscenes_av2_wod taxonomy=mixed2_3cls trainer=ddp
#
# Everything after the script name is passed to py123det-train, so a submitted job takes the same
# overrides as a local one.
#
# Three settings below do the work that makes a multi-day run finish on a three-day queue:
#
#   --signal=SIGUSR1@90   SLURM signals the job 90 seconds before it is killed. Lightning handles
#                         SIGUSR1 by saving hpc_ckpt_<n>.ckpt into the run directory and asking
#                         scontrol to requeue the job, so the save happens mid-epoch rather than
#                         at the next epoch boundary.
#   --requeue             Lets the job go back into the queue instead of ending. Without it the
#                         requeue request is refused and the save is never used.
#   run.resume=last       The requeued attempt runs this same script again, finds that save and
#                         continues from it. The first attempt finds nothing and starts fresh,
#                         which is why the same command works either way.
#
# The pre-emptable partitions are where this matters most: they are cheaper to get onto and the
# job can be stopped at any moment, which with the three settings above costs a few minutes of
# recomputation rather than the run.
#
# --gres has to match trainer.devices, or Lightning waits for ranks that were never allocated.

#SBATCH --job-name=py123det-train
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail

# The environment holding the torch extra. Point PY123DET_ENV at its bin/ directory, or leave it
# unset and have the right python already on PATH.
if [[ -n "${PY123DET_ENV:-}" ]]; then
    export PATH="${PY123DET_ENV%/}/bin:$PATH"
fi

# 123D logs. Hydra reads this through paths.data_root.
export PY123D_DATA_ROOT="${PY123D_DATA_ROOT:?set PY123D_DATA_ROOT to the 123D data root}"

mkdir -p slurm_logs

echo "attempt $(( ${SLURM_RESTART_COUNT:-0} + 1 )) of job ${SLURM_JOB_ID} on $(hostname)"

# srun, not a bare call: Lightning reads SLURM's own rank variables to set up DDP, and they are
# only set for job steps. One task per GPU.
srun --unbuffered py123det-train run.resume=last "$@"
