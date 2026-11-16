#!/bin/bash
#SBATCH -A dlp
#SBATCH -p class_nodes
#SBATCH -q dlp_qos
#SBATCH --exclude=cmpt[047-050,053-055,066-069,082-085,096-099]   # GTX 1060 nodes: driver too old for our torch
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=10G                 # RTX 5050 nodes only have 11.6 GB of RAM
#SBATCH -t 01:00:00               # default = 5 min! The pilot should take about 15 minutes.
#SBATCH -J tai_pilot
#SBATCH -o logs/%x_%j.out

# Usage (from the folder holding pilot.py):
#   once, on the login node : python pilot.py --prepare     (downloads only, no GPU needed)
#   then                    : mkdir -p logs && sbatch run_pilot.sbatch
#   other settings          : sbatch run_pilot.sbatch --teacher-epochs 2 --seed 1
# Any argument after the script name is passed to pilot.py.

export LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1   # requested by the NYCU guide
export PATH=$HOME/venv_lab2/bin:$PATH
export TORCH_HOME=$HOME/tai_pilot/torch_home              # keep DINO weights inside ~/tai_pilot
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
nvidia-smi
python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -u pilot.py --data ./data --out ./runs "$@"
