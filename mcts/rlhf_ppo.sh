#!/bin/bash
#SBATCH --job-name=rlhf_ppo
#SBATCH --partition=instruct-opt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1M
#SBATCH --time=72:00:00
#SBATCH --output="~/l-mcts/logs/%J.%x.out"

wrapper="rlhf_ppo.sh.wrapper"
cat $0
echo "--------------------"
cat $wrapper
echo "--------------------"

time=$(date +"%Y%m%d-%H%M%S")
srun --label ${wrapper} \
    "runs/rlhf_ppo" \
    ${time}.${SLURM_JOB_ID}.${SLURM_JOB_NAME} \
    "pretrained_models/reward-model-sim" "pretrained_models/sft10k"
