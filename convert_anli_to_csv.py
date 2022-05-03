import re
import csv
from datasets import load_dataset

def convert_anli(split):
    """
    Convert anli to CSV file for simcse; Save to data/anli.csvs
    split = ['all', 'train_r1', 'train_r2', 'train_r3'], where 'all' means 
    include all available split
    """
    avail_datasets = []
    if split != 'all':
        dataset = load_dataset('anli', split=split)
        avail_datasets.append(dataset)
    else:
        avail_datasets.append(load_dataset('anli', split='train_r1'))
        avail_datasets.append(load_dataset('anli', split='train_r2'))
        avail_datasets.append(load_dataset('anli', split='train_r3'))
    
    f = open('data/anli_for_simcse.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["sent0","sent1"])
    for dataset in avail_datasets:
        for instance in dataset:
            writer.writerow([instance['hypothesis'].replace(",", ""), instance['premise'].replace(",", "")])
    f.close()

def main():
    convert_anli('train_r1')

if __name__ == "__main__":
    main()