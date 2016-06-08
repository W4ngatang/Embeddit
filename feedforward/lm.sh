#!/bin/bash

python build_lm.py --datafile /dfs/scratch0/wangalex/data/fantasylcs.0.hdf5 --batch_size 1024 --nepochs 20 --vocabfile /dfs/scrat0/wangalex/data/fantasylcs.vocab.pkl --outfile /dfs/scratch0/wangalex/models/fantasylcs_noW2V.ckpt > fantasy.out
echo "Finished $1" > sendmail alexwang@college.harvard.edu
