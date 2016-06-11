#!/bin/bash

# python preprocess.py --srcfiles /dfs/scratch0/wleif/Reddit/clean_comments/FantasyLCS.tsv --n 4 --vocab_size 5000 --outfile /dfs/scratch0/wangalex/data/fantasylcs --w2v /dfs/scratch0/gigawordvecs/GoogleNews-vectors-negative300.bin
python ptb_preprocess.py --srcpath /dfs/scratch0/wangalex/data/simple-examples/data/ptb --n 5 --vocab_size 10000 --outfile /dfs/scratch0/wangalex/data/ptb/ptb10k5
