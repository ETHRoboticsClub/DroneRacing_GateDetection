#!/bin/bash
#SBATCH --job-name=gate_train
#SBATCH --account=ls_polle
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=4090:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "=========================================="

module load eth_proxy
export HF_HOME=/cluster/project/cvg/students/xanyap/.cache/huggingface

source /cluster/project/cvg/students/xanyap/miniconda3/bin/activate base
conda activate ethrc

which python
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

REPO=/cluster/scratch/xanyap/ETHRC-Drone-Racing/src/perception/DroneRacing_GateDetection
cd "$REPO"
mkdir -p logs checkpoints

python scripts/train.py

echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
