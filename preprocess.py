import numpy as np
import h5py
import pickle
import argparse
import sys
import pdb
from collections import defaultdict
import re

def build_vocab(args, txts, special=['<EOS>', '<UNK>', '<NUM>', '<SPECIAL>', '<URL>']):
    freqs = defaultdict(int)
    for txt in txts:
        clean_words = re.sub(r"[0-9]{1,}", "<NUM>", re.sub(r"<(.+?)>", "", txt)).strip().split()
        #clean_words = [float(word) if is_number(word) else word for word in clean_sent]
        for word in clean_words:
            freqs[word] += 1

    # pruning vocab borrowed from Yoon Kim
    vocab = [(word, count) for word, count in freqs.iteritems()]
    vocab.sort(key = lambda x: x[1], reverse = True)
    if args.vocab_size >= 1:
        vocab_size = int(min(args.vocab_size, len(vocab)))
        pruned = [(pair[0],pair[1]) for pair in vocab[:vocab_size]] # originally was a dict, but don't think it's important...
    else:
        pruned = [(pair[0],pair[1]) for pair in vocab[:int(vocab_size*len(vocab))]]

    word2ind = {}
    ind2word = {}
    ind = 0
    for word in special:
        word2ind[word] = ind
        ind2word[ind] = word
        ind += 1

    for word, _ in pruned:
        if word not in word2ind:
            word2ind[word] = ind
            ind2word[ind] = word
            ind += 1

    # Write out vocab
    with open(args.outfile+'.vocab.txt', 'w') as f:
        f.write("Word Index Count\n")
        words = [(word, idx) for word, idx in word2ind.iteritems()]
        words.sort(key = lambda x: x[1])
        for word, idx in words:
            if word in freqs:
                f.write("%s %d %d\n" % (word, idx, freqs[word]))
            else:
                f.write("%s %d %d\n" % (word, idx, -1))

    print '\tFull vocab size: %d, pruned vocab size: %d' % (len(vocab), len(pruned))
    return word2ind, ind2word

# Build pretrained embeddings, credit to Jeffrey (Harvard NLP)
def build_embeds(vec_file, vocab):
    word_vecs = {}
    with open(vec_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    embed = np.random.uniform(-0.25, 0.25, (len(vocab), len(word_vecs.values()[0])))
    embed[0] = 0
    for word, vec in word_vecs.items():
        embed[vocab[word]] = vec
    print "\tLoaded %d vectors" % len(word_vecs)
    return embed

def build_data(txts, word2ind, gram_size):
    inputs = []
    targs = []

    for i, txt in enumerate(txts):
        clean_words = re.sub(r"[0-9]{1,}", "<NUM>", txt).strip().split()
        #words = [word2ind['<NUM>'] if is_number(word) else word for word in txt.split()]
        # TODO may want to pad start of comment / end of comment differently
        conv_txt = [word2ind[word] if word in word2ind else word2ind['<UNK>'] for word in clean_words]
        for j in xrange(len(clean_words)-gram_size):
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
    parser.add_argument('--w2v', help='path to w2v binary', type=str, default='')
    # TODO want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)
    if not args.srcfile or not args.outfile:
        raise ValueError("srcfile or outfile not provided")

    print "Reading in data..."
    complete_data = np.genfromtxt(args.srcfile, dtype=str, delimiter='\t')
    txt_data = complete_data[:,-1] # due to file formatting

    print "Generating vocab..."
    word2ind, ind2word = build_vocab(args, txt_data)

    if args.w2v:
        print "Loading vectors from word2vec..."
        embeds = build_embeds(args.w2v, word2ind)

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
        if args.w2v:
            f['embeds'] = embeds

    '''
    with file(args.outfile+'.vocab.hdf5', 'w') as f:
        pickle.dump((word2ind, ind2word), f) 
    '''
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
