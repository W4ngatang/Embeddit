import numpy as np
import tensorflow as tf
import h5py
import pickle
import pdb
import time
import ops
import data

def build_data(args):
    datafile = h5py.File(args.datafile, 'r')
    data = Dataset(datafile, args.batch_size)
    return data

def get_feed_dict(data, i, input_p, targ_p):
    input_b, targ_b = data.batches[i] # implement dataset class

    feed_dict = {
        input_p: input_b,
        tag_p: targ_p,
    }
    return feed_dict

def train(args):

    with tf.Graph().as_default():
        input_p = tf.placeholder(tf.int32, shape=[args.batch_size, args.n-1])
        targ_p = tf.placeholder(tf.int32, shape=[args.batch_size])

        scores = ops.model(inputs, args.d_hid, args.d_emb)
        loss = ops.loss(scores, targs)
        train_op = ops.train(loss, args.learn_rate)
        valid_op = ops.valid(loss)

        '''
        summary_op = tf.merge_all_summaries() # TODO read about summaries + savers
        saver = tf.train.Saver()
        '''

        sess = tf.Session()
        init = tf.initialize_all_variables() # initialize variables before they can be used
        sess.run(init)

        summary_writer = tf.train.SummaryWriter()

        for epoch in args.nepochs: # NOTE this is steps, not epochs
            start_time = time.time()
            for i in xrange(nbatches):
                feed_dict = get_feed_dict(dataset, i, input_p, targ_p)
                # build feed dict

                # if you run loss node, does it not run train as well?
                _, val_score = sess.run([train_op, loss, valid_op], feed_dict=feed_dict)

            duration = time.time() - start_time
            print "Epoch %d: loss = %.3f, valid ppl = %.3f; .3f s" % 
                (epoch, loss_score, val_score, duration)

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datafile', help='source data file', type=str)
    parser.add_argument('--outfile', help='file prefix to write to (hdf5)', type=str)
    parser.add_argument('--batch_size', help='batch_size', type=int)
    parser.add_argument('--learn_rate', help='initial learning rate', type=float)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int)
    # want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)
    if not args.srcefile or not args.outfile:
        raise ValueError("srcfile or outfile not provided")

    # load data
    data = build_data(args)

    # train
    train(args, data)

         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
