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

def build_data(args):
    datafile = h5py.File(args.datafile, 'r')
    train_inputs = datafile['train_inputs']
    train_targs = datafile['train_targets']
    valid_inputs = datafile['valid_inputs']
    valid_targs = datafile['valid_targets']
    train = Dataset(train_inputs, train_targs, args.batch_size)
    valid = Dataset(valid_inputs, valid_targs, args.batch_size)
    return {'train':train, 'valid':valid}, \
            {'gram_size':datafile['gram_size'][0], 'vocab_size':datafile['vocab_size'][0], \
                'hid_size':args.d_hid, 'emb_size':args.d_emb}

def get_feed_dict(data, i, input_ph, targ_ph, learning_rate_ph, learning_rate=0.):
    input_batch, targ_batch = data.batch(i)

    feed_dict = {
        input_ph: input_batch,
        targ_ph: targ_batch,
        learning_rate_ph: learning_rate
    }
    return feed_dict

def train(args, data, params):
    train = data['train']
    valid = data['valid']
    learning_rate = args.learning_rate

    with tf.Graph().as_default():
        input_ph = tf.placeholder(tf.int32, shape=[args.batch_size,params['gram_size']-1])
        targ_ph = tf.placeholder(tf.int32, shape=[args.batch_size])
        learning_rate_ph = tf.placeholder(tf.float32, shape=[])

        if args.w2v:
            with h5py.File(args.datafile, 'r') as datafile:
                embeds = datafile['embeds'][:]   
            scores, normalize_op = ops.model(input_ph, params, embeds)
        else:
            scores, normalize_op = ops.model(input_ph, params)
        
        loss = ops.loss(scores, targ_ph)
        train_op = ops.train(loss, learning_rate_ph, args)
        valid_op = ops.validate(loss)

        last_valid = 1000000 # big number

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
            train_loss = 0.
            valid_loss = 0.
            for i in xrange(train.nbatches):
                train_feed_dict = get_feed_dict(train, i, input_ph, targ_ph, learning_rate_ph, learning_rate)
                _, batch_loss = sess.run([train_op, loss], feed_dict=train_feed_dict)
                train_loss += batch_loss

            if args.normalize:
                _ = sess.run([normalize_op])

            for i in xrange(valid.nbatches):
                valid_feed_dict = get_feed_dict(valid, i, input_ph, targ_ph, learning_rate_ph)
                batch_loss = sess.run([loss], feed_dict=valid_feed_dict)[0]
                valid_loss += batch_loss

            duration = time.time() - start_time
            print "\tloss = %.3f, valid ppl = %.3f, %.3f s" % \
                (math.exp(train_loss/train.nbatches), \
                    math.exp(valid_loss/valid.nbatches), duration)
            if last_valid < valid_loss:
                learning_rate /= 2.
            elif args.outfile:
                saver.save(sess, args.outfile)#, global_step=epoch)
            last_valid = valid_loss
        
        return sess.run([normalize_op])[0] # return final normalized embeddings

def visualize(args, embeddings):
    with open(args.vocabfile, 'r') as f:
        _, ind2word = pickle.load(f)
        print "Loaded vocabulary"
    pdb.set_trace()
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 50
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only,:])
    labels = [ind2word[i] for i in xrange(plot_only)]
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x,y), xytext=(5,2), textcoord='offset points', ha='right', va='bottom')

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datafile', help='source data file', type=str)
    parser.add_argument('--outfile', help='file to save best model to', type=str, default='')
    parser.add_argument('--modelfile', help='file to load model variable values from', type=str, default='')
    parser.add_argument('--batch_size', help='batch_size', type=int)
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=1.)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int)
    parser.add_argument('--d_hid', help='hidden layer size', type=int, default=100)
    parser.add_argument('--d_emb', help='embedding size', type=int, default=300)
    parser.add_argument('--grad_reg', help='type of gradient regularization (either norm or clip)', type=str, default='norm')
    parser.add_argument('--max_grad', help='maximum gradient value', type=float, default=5.)
    parser.add_argument('--normalize', help='1 if normalize, 0 otherwise', type=int, default=0)
    parser.add_argument('--w2v', help='1 if load w2v vectors, 0 if no', type=int, default=0)
    parser.add_argument('--visualize', help='1 if visualize, 0 otherwise', type=int, default=0)
    parser.add_argument('--vocabfile', help='path to pickle file containing vocabulary', type=str)
    # want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)

    # load data and model parameters
    print "Loading data..."
    d, p = build_data(args)

    # train
    print "Beginning training using %d threads..." % NUM_THREADS
    embeddings = train(args, d, p)

    if args.visualize:
        print "Visualizing embeddings"
        visualize(args, embeddings)
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
