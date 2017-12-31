#!/bin/bash

python ff_lm.py --datafile /n/home09/wangalexc/Embeddit/feedforward/ptb.hdf5 --batch_size 1024 --nepochs 30 --vocabfile /n/home09/wangalexc/Embeddit/feedforward/ptb.vocab.pkl --outfile /n/home09/wangalexc/Embeddit/feedforward/ptb10k5.ckpt --d_emb 30 --learning_rate 4.0 --normalize 1
# sendmail alexwang@college.harvard.edu
