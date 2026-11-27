#!/bin/bash
# Score a checkpoint on a SLURM cluster.
#
#   sbatch scripts/slurm/eval.sbatch experiment=nuscenes \
#       evaluation.checkpoint=work_dirs/petr_nuscenes_10cls/checkpoints/last.ckpt
#
# Everything after the script name is passed to py123det-eval.
#
# No requeue handling here: evaluation writes each frame's predictions as it goes, so a job that
# dies is restarted with evaluation.rescore_from pointed at what it already wrote, and inference
# is not repeated for the frames already in the file.
#
# evaluation.devices shards inference across GPUs and has to match --gres. It defaults to 1.

#SBATCH --job-name=py123det-eval
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

if [[ -n "${PY123DET_ENV:-}" ]]; then
    export PATH="${PY123DET_ENV%/}/bin:$PATH"
fi

export PY123D_DATA_ROOT="${PY123D_DATA_ROOT:?set PY123D_DATA_ROOT to the 123D data root}"

mkdir -p slurm_logs

srun --unbuffered py123det-eval "$@"
