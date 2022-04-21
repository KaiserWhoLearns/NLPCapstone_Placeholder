## Reproducing SimCSE: Simple Contrastive Learning of Sentence Embeddings

## Running our experimental scripts

The extra experiment scripts created for capstone class are included in directory `local_exp_scripts`, which contains `bash` scripts. For `slurm` scripts, refer to directory `cluster_exp_scripts`. 

Below is a list of scripts in `local_exp_scripts`:

* `run_exp.sh` is the general run experiment file. It will ask for the experiment to run, the output location, and your output file name. It is okay to run experiments without calling this file, while it just provides a general automated way to run experiments.

* `run_sup_evaluation.sh` runs evaluation on all the available supervised models (`sup-simcse-bert-base-uncased`, `sup-simcse-bert-large-uncased`, `sup-simcse-roberta-base`, `sup-simcse-roberta-large`) and output the results in the directory specificed by user under `logs/`.