# !/usr/bin/env python3
# -*- coding: utf8 -*-

import nltk
import sys
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()


def tokenize_sents(infile, outfile):
    with open(infile, "rt") as inf:
        with open(outfile, "w+") as outf:
            for line in inf:
                sents = nltk.tokenize.sent_tokenize(line)
                for i in sents:
                    outf.write("{}{}".format(i, "\n"))

def twitter_tokenize(infile, outfile):
    with open(infile, "rt") as inf:
        with open(outfile, "w+") as outf:
            for line in inf:
                tokens = tknzr.tokenize(line)
                for i in tokens:
                    outf.write("{}{}".format(i, " "))
                outf.write("\n")

def main():
    # tokenize_sents(sys.argv[1], sys.argv[2])
    twitter_tokenize(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
