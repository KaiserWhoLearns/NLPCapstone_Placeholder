#!/bin/bash
read -p "Enter the file name to run; This file has to be under local_exp_scripts: " expfile
read -p "Enter the output file prefix (will be concat with _out or _err) " output_prefix
read -p "Enter your base directory for SimCSE: " base_dir

cd $base_dir

bash "local_exp_scripts/$expfile" 2> "logs/${output_prefix}_err.log" 1> "logs/${output_prefix}_out.log"