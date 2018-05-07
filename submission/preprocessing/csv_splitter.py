# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys

def split_csv(infile, outfile):
    with open(infile, "rt") as inf:
        with open(outfile, "w+") as output:
            reader = csv.reader(inf)
            for row in reader:
                output.write("{}{}".format(row, "\n"))

def main():
    split_csv(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
