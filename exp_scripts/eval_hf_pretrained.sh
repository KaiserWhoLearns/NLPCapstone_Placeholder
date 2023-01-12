#!/bin/bash
# Evaluate the pre-trained models (on Huggingface)
read -p "Enter your base directory for SimCSE: " base_dir

cd $base_dir

python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-bert-large-uncased --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-roberta-base --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-roberta-large --task_set sts --mode test