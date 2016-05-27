import numpy as np
import tensorflow as tf
import pdb
import time
import ops

def train(args):

    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.int32, shape=[args.batch_size])
        targs = tf.placeholder(tf.int32, shape=[args.batch_size, 1])

        scores = ops.model(inputs, args.d_hid, args.d_emb)
        loss = ops.loss(scores, targs)
        train_op = ops.train(loss, args.learn_rate)
        valid_op = ops.valid(loss)

        summary_op = tf.merge_all_summaries() # TODO read about summaries + savers
        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.initialize_all_variables() # initialize variables before they can be used
        sess.run(init)

        summary_writer = tf.train.SummaryWriter()

        for epoch in args.nepochs: # NOTE this is steps, not epochs
            start_time = time.time()

            # build feed dict

            sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if not epoch % 100:
                print "Epoch %d: loss = %.3f, time = %.3f s" % (epoch, loss_val, duration)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summar_writer.add_summary(summary_str, epoch)
                summary_writer.flush()


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--srcfile', help='source data file', type=str)
    parser.add_argument('--outfile', help='file prefix to write to (hdf5)', type=str)
    parser.add_argument('--batch_size', help='batch_size', type=int)
    parser.add_argument('--learn_rate', help='initial learning rate', type=float)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int)
    # want to potentially save input data as pickle, write vocab
    args = parser.parse_args(arguments)
    if not args.srcefile or not args.outfile:
        raise ValueError("srcfile or outfile not provided")

    # load data

    # train

         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
