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

    # Key: Premises, values=[entailment (label 0), contradiction (label 1)]
    premises = {}
    for dataset in avail_datasets:
        # for instance in dataset:
        #     writer.writerow([instance['hypothesis'].replace(",", ""), instance['premise'].replace(",", "")])
        for instance in dataset:
            premise = instance['premise'].replace(",", "")
            if premise in premises:
                if instance['label'] == 1:
                    premises[premise][1] = instance['hypothesis'].replace(",", "")
                elif instance['label'] == 0:
                    premises[premise][0] = instance['hypothesis'].replace(",", "")
            else:
                if instance['label'] == 1:
                    premises[premise] = ["", instance['hypothesis'].replace(",", "")]
                elif instance['label'] == 0:
                    premises[premise] = [instance['hypothesis'].replace(",", ""), ""]

    f = open('data/anli_for_simcse.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["sent0","sent1", "hard_neg"])
    for premise in premises.keys():
        if premises[premise][0] != "" and premises[premise][1] != "":
            # log into dataset
            writer.writerow([premise, premises[premise][0], premises[premise][1]])
    f.close()

def main():
    convert_anli('all')

if __name__ == "__main__":
    main()