#!/bin/bash

#SBATCH --job-name=BertLarge
#SBATCH --mail-user=huikas@cs.washington.edu
#SBATCH --mail-type=ALL
#SBATCH --account=ark
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=0-23:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=/gscratch/ark/huikas/nlp-capstone-simcse/
#SBATCH --export=all
#SBATCH --output=/gscratch/ark/huikas/nlp-capstone-simcse/logs/training/unsup_bert_large_out.log
#SBATCH --error=/gscratch/ark/huikas/nlp-capstone-simcse/logs/training/unsup_bert_large_err.log

# Modules to use (optional).
# <e.g., module load singularity>

# Your programs to run.
# <my_programs>
# conda init bash

conda init bash
conda activate /gscratch/ark/huikas/miniconda3/envs/simcse

python train.py \
    --model_name_or_path bert-large-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir train_models/unsup-simcse-bert-large-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
