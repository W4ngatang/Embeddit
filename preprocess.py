import numpy as np
import h5py
import pickle
import argparse
import sys
import pdb
from collections import defaultdict
import re

def is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        return False

# TODO nice to have: write vocabulary out
def build_vocab(args, txts, special=['<EOS>', '<UNK>', '<NUM>', '<SPECIAL>', '<URL>']):
    freqs = defaultdict(int)
    for txt in txts:
        clean_sent = re.sub(r"<(.+?)>", "", txt).strip().split()
        clean_words = [float(word) if is_number(word) else word for word in clean_sent]
        for word in clean_words:
            freqs[word] += 1

    # pruning vocab borrowed from Yoon Kim
    vocab = [(word, count) for word, count in freqs.iteritems()]
    vocab.sort(key = lambda x: x[1], reverse = True)
    if args.vocab_size >= 1:
        vocab_size = int(min(args.vocab_size, len(vocab)))
        pruned = {pair[0]:pair[1] for pair in vocab[:vocab_size]}
    else:
        pruned = {pair[0]:pair[1] for pair in vocab[:int(vocab_size*len(vocab))]}

    word2ind = {}
    ind2word = {}
    ind = 0
    for word in special:
        word2ind[word] = ind
        ind2word[ind] = word
        ind += 1

    for word in pruned:
        if word not in word2ind:
            word2ind[word] = ind
            ind2word[ind] = word
            ind += 1

    print 'Full vocab size: %d, pruned vocab size: %d' % (len(vocab), len(pruned))
    return word2ind, ind2word

def build_data(txts, word2ind, gram_size):
    inputs = []
    targs = []

    for i, txt in enumerate(txts):
        words = [word2ind['<NUM>'] if is_number(word) else word for word in txt.split()]
        # TODO may want to pad start of comment / end of comment differently
        conv_txt = [word2ind[word] if word in word2ind else word2ind['<UNK>'] for word in words]
        for j in xrange(len(words)-gram_size):
            inputs.append(conv_txt[j:j+gram_size])
            targs.append(conv_txt[j+gram_size])

    return np.array(inputs, dtype=int), np.array(targs, dtype=int)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--srcfile', help='source data file', type=str)
    parser.add_argument('--n', help='gram size', type=int, default=5)
    parser.add_argument('--split', help='train size as fraction of data', type=float, default=.8)
    parser.add_argument('--outfile', help='file prefix to write to (hdf5)', type=str)
    parser.add_argument('--vocab_size', help='max vocab size, either an absolute number or \
                                                a percentage', type=float, default=1.)
    # TODO want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)
    if not args.srcfile or not args.outfile:
        raise ValueError("srcfile or outfile not provided")

    print "Reading in data..."
    complete_data = np.genfromtxt(args.srcfile, dtype=str, delimiter='\t')
    txt_data = complete_data[:,-1] # due to file formatting

    print "Generating vocab..."
    word2ind, ind2word = build_vocab(args, txt_data)

    print "Generating data..."
    inputs, targs = build_data(txt_data, word2ind, args.n-1) # do n-1 b/c lazy indexing
    split_pt = int(inputs.shape[0]*args.split)

    with h5py.File(args.outfile+'.hdf5', 'w') as f:
        f['train_inputs'] = inputs[:split_pt]
        f['train_targets'] = targs[:split_pt]
        f['valid_inputs'] = inputs[split_pt:]
        f['valid_targets'] = targs[split_pt:]
        f['gram_size'] = np.array([args.n], dtype=np.int32)
        f['vocab_size'] = np.array([len(word2ind)], dtype=np.int32)

    with file(args.outfile+'.vocab.hdf5', 'w') as f:
        pickle.dump((word2ind, ind2word), f) 
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
