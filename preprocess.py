import numpy as np
import h5py
import pickle
import argparse
import sys
import pdb

# TODO UNK filter, need to keep track of freqs
def build_vocab(txts):
    word2ind = {}
    ind2word = {}
    ind = 0

    for txt in txts:
        words = txt.split()
        for word in words:
            if word not in word2ind:
                word2ind[word] = ind
                ind2word[ind] = word
                ind += 1

    return word2ind, ind2word

def build_data(txts, word2ind, gram_size):
    inputs = []
    targs = []

    for i, txt in enumerate(txts):
        words = txt.split()
        conv_txt = [word2ind[word] for word in words] # note: data is already padded
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
    # want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)
    if not args.srcefile or not args.outfile:
        raise ValueError("srcfile or outfile not provided")

    print "Reading in data..."
    complete_data = np.genfromtxt(args.srcfile, dtype=str, delimiter='\t')
    txt_data = complete_data[:,-1] # due to file formatting

    print "Generating vocab..."
    word2ind, ind2word = build_vocab(txt_data)

    print "Generating data..."
    inputs, targs = build_data(txt_data, word2ind, args.n-1) # do n-1 b/c lazy indexing
    split_pt = int(inputs.shape[0]*args.split)

    with h5py.File(args.outfile+'.hdf5', 'w') as f:
        f['train_inputs'] = inputs[:split_pt]
        f['train_targets'] = targs[:split_pt]
        f['valid_inputs'] = inputs[split_pt:]
        f['valid_targets'] = targs[split_pt:]
        f['gram_size'] = np.array([args.n], dtype=np.int32)

    with file(args.outfile+'.vocab.hdf5', 'w') as f:
        pickle.dump((word2ind, ind2word), f) 
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
