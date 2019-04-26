#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080

#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=train-cadgan-mimic-umn
#SBATCH --output=slurm_output/train-cadgan-mimic-umn.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=192GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Run the job
python train_cadgan.py --data_path data/umn/ data/mimic/ --save save/cadgan/ --batch_size 256 --char_ckpt save/sf_lf_pretrain/ckpt_epoch30.pt --word_ckpt save/context_pretrain_small/ckpt_epoch20.pt --finetune_ae