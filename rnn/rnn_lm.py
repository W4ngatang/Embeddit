import numpy as np
import tensorflow as tf
import sys
import argparse
import h5py
import pickle
import pdb
import time
import ops
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data import Dataset

NUM_THREADS = 32
BIG_NUM = 100000

def build_data(args):
    datafile = h5py.File(args.datafile, 'r')
    train_inputs = datafile['train_inputs']
    train_targs = datafile['train_targets']
    valid_inputs = datafile['valid_inputs']
    valid_targs = datafile['valid_targets']
    test_inputs = datafile['test_inputs']
    train = Dataset(args.batch_size, train_inputs, train_targs)
    valid = Dataset(args.batch_size, valid_inputs, valid_targs)
    test = Dataset(args.batch_size, test_inputs)
    return {'train':train, 'valid':valid, 'test':test}, \
            {'gram_size':datafile['gram_size'][0], 'batch_size':args.batch_size, \
                'vocab_size':datafile['vocab_size'][0], 'd_hid':args.d_hid, \
                'seq_len':args.seq_len}

def get_feed_dict(data, i, model):
    input_batch, targ_batch = data.batch(i)
    feed_dict = {
        model.input: input_batch,
        model.targ: targ_batch,
        model.state: model.initial_state.eval()
    }
    return feed_dict

def run_epoch(data, model, train=1):
    loss = 0
    for i in xrange(data.nbatches):
        batch_d = get_feed_dict(data, i, model)
        if train:
            _ , batch_loss = sess.run([model.train, model.loss], feed_dict=batch_d)
        else:
            batch_loss = sess.run([model.loss], feed_dict=d)[0]
        loss += batch_loss
    return loss

def train(args, data, params, model):
    train = data['train']
    valid = data['valid']
    learning_rate = args.learning_rate

    with tf.Graph().as_default():
        last_valid = BIG_NUM

        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
				intra_op_parallelism_threads=NUM_THREADS))
        init = tf.initialize_all_variables() # initialize variables before they can be used
        saver = tf.train.Saver()
        sess.run(init)

        if args.modelfile:
            saver.restore(sess, args.modelfile)
            print "Model restored from %s" % args.modelfile

        for epoch in xrange(args.nepochs):
            print "Training epoch %d w/ learning rate %.3f..." % (epoch, learning_rate)
            start_time = time.time()
            train_loss = run_epoch(train_data, model, 1)
            valid_loss = run_epoch(valid_data, model)
            # TODO regularization?
            duration = time.time() - start_time
            print "\tloss = %.3f, valid ppl = %.3f, %.3f s" % \
                (math.exp(train_loss/train.nbatches), \
                    math.exp(valid_loss/valid.nbatches), duration)
            if last_valid < valid_loss:
                learning_rate /= 2.
            elif args.outfile:
                saver.save(sess, args.outfile)
            last_valid = valid_loss
        
        return #sess.run([normalize_op])[0] # return final normalized embeddings

def visualize(args, embeddings):
    with open(args.vocabfile, 'r') as f:
        word2ind, ind2word = pickle.load(f)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    pdb.set_trace()
    plot_only = 1000
    print '\tRunning t-SNE...'
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only,:])
    labels = [ind2word[i] for i in xrange(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(25, 25))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x,y), xytext=(5,2), \
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datafile', help='source data file', type=str)
    parser.add_argument('--outfile', help='file to save best model to', type=str, default='')
    parser.add_argument('--modelfile', help='file to load model variable values from', type=str, default='')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--seq_len', help='length of each sequence per batch', type=int, default=20)
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=1.)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int, default=10)
    parser.add_argument('--d_hid', help='hidden layer size', type=int, default=100)
    parser.add_argument('--grad_reg', help='type of gradient regularization (either norm or clip)', type=str, default='')
    parser.add_argument('--max_grad', help='maximum gradient value', type=float, default=5.)
    parser.add_argument('--normalize', help='1 if normalize, 0 otherwise', type=int, default=0)
    parser.add_argument('--w2v', help='1 if load w2v vectors, 0 if no', type=int, default=0)
    parser.add_argument('--visualize', help='1 if visualize, 0 otherwise', type=int, default=0)
    parser.add_argument('--vocabfile', help='path to pickle file containing vocabulary', type=str)
    args = parser.parse_args(arguments)
    if not args.datafile:
        raise ValueError("Must include path to data")

    # load data and model parameters
    print "Loading data..."
    data, params = build_data(args)

    # get model
    print "Building model..." # TODO move loading model to here?
    model = ops.Model(params)

    # train and validate
    print "Beginning training using %d threads..." % NUM_THREADS
    embeddings = train(args, data, params, model)

    # test
    print "Test perplexity: %.3f" % (run_epoch(data['test'], model))

    if args.visualize:
        print "Visualizing embeddings"
        visualize(args, embeddings)
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
