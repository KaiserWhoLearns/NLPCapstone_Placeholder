#!/bin/bash

#SBATCH --job-name=EvalUnsupSimCSE
#SBATCH --mail-user=huikas@cs.washington.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ark
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=0-23:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/gscratch/ark/huikas/nlp-capstone-simcse/
#SBATCH --export=all
#SBATCH --output=/gscratch/ark/huikas/nlp-capstone-simcse/logs/eval_unsup_out.log
#SBATCH --error=/gscratch/ark/huikas/nlp-capstone-simcse/logs/eval_unsup_err.log

# Modules to use (optional).
# <e.g., module load singularity>

# Your programs to run.
# <my_programs>
# conda init bash

conda init bash
conda activate /gscratch/ark/huikas/miniconda3/envs/simcse

python evaluation.py --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased --task_set=sts --mode=test

python evaluation.py --model_name_or_path princeton-nlp/unsup-simcse-bert-large-uncased --task_set sts --mode test

python evaluation.py --model_name_or_path princeton-nlp/unsup-simcse-roberta-base --task_set sts --mode test

python evaluation.py --model_name_or_path princeton-nlp/unsup-simcse-roberta-large --task_set sts --mode test
