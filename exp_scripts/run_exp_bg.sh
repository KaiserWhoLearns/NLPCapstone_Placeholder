#!/bin/bash

# This script run experiments in background and output logs to the specified log files
# Remember to specify the corresponding environ variables before running the script
read -p "Enter the file name to run" expfile
read -p "Enter the output file prefix (will be concat with _out or _err) " output_prefix
read -p "Enter your base directory for SimCSE: " base_dir

cd $base_dir
mkdir -p logs

bash "$expfile" 2> "logs/${output_prefix}_err.log" 1> "logs/${output_prefix}_out.log"