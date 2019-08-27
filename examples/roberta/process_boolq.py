import argparse
import os
import random
from glob import glob

random.seed(0)

def main():
    with open("/Users/danielk/ideaProjects/fairseq/examples/roberta/glue_data/SNLI/train.tsv") as f:
        lines = f.readlines()
        lines_all = []
        for line in lines:
            split_line = line.split("\t")
            print(len(split_line))
            print(split_line)
            sentence1 = split_line[7]
            sentence2 = split_line[8]
            label = split_line[-1]
            lines_all.append(f"{sentence1}\t{sentence2}\t{label}")
    with open("/Users/danielk/ideaProjects/fairseq/examples/roberta/glue_data/SNLI/train-simple.tsv", 'w+') as f:
        f.write("".join(lines_all))

if __name__ == '__main__':
    main()