#!/bin/bash

python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-bert-large-uncased --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-roberta-base --task_set sts --mode test
python3 evaluation.py --model_name_or_path princeton-nlp/sup-simcse-roberta-large --task_set sts --mode test