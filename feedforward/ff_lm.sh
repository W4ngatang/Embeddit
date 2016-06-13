#!/bin/bash

python ff_lm.py --datafile /dfs/scratch0/wangalex/data/ptb/ptb10k5.hdf5 --batch_size 1024 --nepochs 30 --vocabfile /dfs/scratch0/wangalex/data/ptb/ptb10k5.vocab.pkl --outfile /dfs/scratch0/wangalex/data/ptb/ptb10k5.ckpt --w2v 0 --d_emb 30 --learning_rate 4.0 --normalize 1
# sendmail alexwang@college.harvard.edu
