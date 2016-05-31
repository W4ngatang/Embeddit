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
from data import Dataset

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

def get_feed_dict(data, i, input_ph, targ_ph):
    input_batch, targ_batch = data.batch(i)

    feed_dict = {
        input_ph: input_batch,
        targ_ph: targ_batch,
    }
    return feed_dict

def train(args, data, params):
    train = data['train']
    valid = data['valid']

    with tf.Graph().as_default():
        input_ph = tf.placeholder(tf.int32, shape=[args.batch_size,params['gram_size']-1])
        targ_ph = tf.placeholder(tf.int32, shape=[args.batch_size])

        scores = ops.model(input_ph, params)
        loss = ops.loss(scores, targ_ph)
        train_op = ops.train(loss, args.learn_rate)
        valid_op = ops.validate(loss)

        '''
        summary_op = tf.merge_all_summaries() # TODO read about summaries + savers
        saver = tf.train.Saver()
        '''

        sess = tf.Session()
        init = tf.initialize_all_variables() # initialize variables before they can be used
        sess.run(init)

        for epoch in xrange(args.nepochs):
            print "Training epoch %d..." % epoch
            start_time = time.time()
            train_loss = 0.
            valid_loss = 0.
            for i in xrange(train.nbatches):
                train_feed_dict = get_feed_dict(train, i, input_ph, targ_ph)
                _, batch_loss = sess.run([train_op, loss], feed_dict=train_feed_dict)
                train_loss += batch_loss

            for i in xrange(valid.nbatches):
                valid_feed_dict = get_feed_dict(valid, i, input_ph, targ_ph)
                batch_loss = sess.run([loss], feed_dict=valid_feed_dict)[0]
                valid_loss += batch_loss

            duration = time.time() - start_time
            print "\tloss = %.3f, valid ppl = %.3f, %.3f s" % \
                (math.exp(train_loss/train.nbatches), \
                    math.exp(valid_loss/valid.nbatches), duration)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datafile', help='source data file', type=str)
    parser.add_argument('--batch_size', help='batch_size', type=int)
    parser.add_argument('--learn_rate', help='initial learning rate', type=float)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int)
    parser.add_argument('--d_hid', help='hidden layer size', type=int, default=100)
    parser.add_argument('--d_emb', help='embedding size', type=int, default=300)
    # want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)

    # load data and model parameters
    print "Loading data..."
    d, p = build_data(args)
    print "Vocab size: %d" % p['vocab_size']

    # train
    print "Beginning training..."
    train(args, d, p)
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
