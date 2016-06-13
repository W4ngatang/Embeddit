import numpy as np
import h5py
import pickle
import argparse
import sys
import pdb
from collections import defaultdict
import re

def build_vocab(args, special=['<eos>', '<unk>', 'N']):
    print "Generating vocab..."
    datafiles = [args.srcpath + d for d in ['.train.txt', '.valid.txt', '.test.txt']]
    freqs = defaultdict(int) 
    for i, datafile in enumerate(datafiles):
        print "\tProcessing datafile %d of %d..." % (i+1, len(datafiles))
        with open(datafile, 'r') as df:
            for post in df:
                for word in post.strip().split():
                    freqs[word] += 1

    print '\tPruning vocab...'
    # pruning vocab borrowed from Yoon Kim
    vocab = [(word, count) for word, count in freqs.iteritems()]
    vocab.sort(key = lambda x: x[1], reverse = True)
    if args.vocab_size >= 1:
        vocab_size = int(min(args.vocab_size, len(vocab)))
        pruned = [(pair[0],pair[1]) for pair in vocab[:vocab_size]] 
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
    with open(args.outfile+'.vocab.pkl', 'w') as f:
        pickle.dump((word2ind, ind2word), f)

    print '\tFull vocab size: %d, pruned vocab size: %d' % (len(vocab), len(word2ind))
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

def build_data(datafile, word2ind, batch_size, seq_len):
    big_txt = [word2ind['<eos>']]
    with open(datafile, 'r') as df:
        for post in df:
            padded_words = post.strip().split() + ['<eos>']
            big_txt += [word2ind[word] if word in word2ind else word2ind['<unk>'] for word in padded_words]
    input_txt = big_txt[:-1] # after processing into batches, may not evenly divide -> won't end in eos
    target_txt = big_txt[1:]

    inputs = []
    targets = []
    nbatches = int(len(big_txt) / (seq_len * batch_size))
    block_size = seq_len * nbatches
    for i in xrange(nbatches):
        input_batch = []
        target_batch = []
        for j in xrange(batch_size):
            offset = j*block_size+i*seq_len
            input_batch.append(input_txt[offset:offset+seq_len])
            target_batch.append(target_txt[offset:offset+seq_len])
        inputs.append(input_batch)
        targets.append(target_batch)

    return np.array(inputs,dtype=int), np.array(targets,dtype=int)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--srcpath', help='path prefix to source files (train, val, test)', type=str)
    parser.add_argument('--outfile', help='file to write to (hdf5)', type=str)
    parser.add_argument('--vocab_size', help='max vocab size, either an absolute number or \
                                                a percentage', type=float, default=1.)
    parser.add_argument('--w2v', help='path to w2v binary', type=str, default='')
    parser.add_argument('--batch_size', help='path to w2v binary', type=int, default=32)
    parser.add_argument('--seq_len', help='path to w2v binary', type=int, default=20)
    args = parser.parse_args(arguments)
    if not args.srcpath or not args.outfile:
        raise ValueError("srcpath or outfile not provided")

    word2ind, ind2word = build_vocab(args)

    if args.w2v:
        print "Loading vectors from word2vec..."
        embeds = build_embeds(args.w2v, word2ind)

    print "Generating train data..."
    tr_inputs, tr_targs = build_data(args.srcpath+'.train.txt', word2ind, \
                            args.batch_size, args.seq_len) 
    print "Generating valid data..."
    va_inputs, va_targs = build_data(args.srcpath+'.valid.txt', word2ind, \
                            args.batch_size, args.seq_len)
    print "Generating test data..."
    te_inputs, te_targs = build_data(args.srcpath+'.test.txt', word2ind, \
                            args.batch_size, args.seq_len)

    with h5py.File(args.outfile + '.hdf5') as f:
        f['train_inputs'] = tr_inputs
        f['train_targets'] = tr_targs
        f['valid_inputs'] = va_inputs
        f['valid_targets'] = va_targs
        f['test_inputs'] = te_targs
        f['test_targets'] = te_targs
        f['batch_size'] = np.array([args.batch_size], dtype=np.int32)
        f['seq_len'] = np.array([args.seq_len], dtype=np.int32)
        f['vocab_size'] = np.array([len(word2ind)], dtype=np.int32)
        if args.w2v:
            f['embeds'] = embeds
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
