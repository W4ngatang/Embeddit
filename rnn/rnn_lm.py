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
from tensorflow.models.rnn.ptb import reader

NUM_THREADS = 32
BIG_NUM = 100000

def build_data(args):
    df = h5py.File(args.datafile, 'r')
    train_inputs = df['train_inputs']
    train_targs = df['train_targets']
    valid_inputs = df['valid_inputs']
    valid_targs = df['valid_targets']
    test_inputs = df['test_inputs']
    train = Dataset(train_inputs, train_targs)
    valid = Dataset(valid_inputs, valid_targs)
    test = Dataset(test_inputs)

    model_params = {'batch_size':df['batch_size'][0], 'seq_len':df['seq_len'][0],\
                    'vocab_size':df['vocab_size'][0], 'd_hid':args.d_hid, \
                    'nlayers':args.nlayers, 'drop_prob':args.drop_prob, \
                    'model':args.model, 'init_scale':args.init_scale,
                    'max_grad_norm':args.max_grad_norm}
    train_params = {'nepochs':args.nepochs, 'learning_rate':args.learning_rate,\
                    'decay_after':args.decay_after, 'decay_rate':args.decay_rate}

    def typed(val, o_val):
        if isinstance(o_val, float):
            return float(val)
        if isinstance(o_val, int):
            return int(val)
        if isinstance(o_val, str):
            return str(val)

    if args.configfile:
        with open(args.configfile, 'r') as cf:
            for row in cf:
                arg, val = row.split()
                if arg in model_params:
                    model_params[arg] = typed(val, model_params[arg])
                elif arg in train_params:
                    train_params[arg] = typed(val, train_params[arg])
                else:
                    raise ValueError("Unknown parameter %s" % arg)

    return {'train':train, 'valid':valid, 'test':test}, model_params, train_params

def run_epoch(sess, data, model, eval_op):
    loss = 0.0
    state = model.initial_state.eval()
    for i in xrange(data.nbatches):
        x, y = data.batch(i)
        batch_loss, state, _ = sess.run([model.loss, model.last_state, eval_op],
                                    {model.input:x, model.targ:y, model.initial_state:state})
        loss += batch_loss
        #if steps % (epoch_size // 10) == 10:
        #    print ".%.3f, perplexity: %.3f" % (steps * 1.0 / epoch_size, np.exp(loss/ steps))

    return math.exp(loss/(data.nbatches*model.seq_len))

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

def log(lf, message):
    print message
    lf.write(message+'\n')
    return

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datafile', help='source data file', type=str)
    parser.add_argument('--datapath', help='source data path', type=str)
    parser.add_argument('--outfile', help='file to save best model to', type=str, default='') # unused
    parser.add_argument('--modelfile', help='file to load model variable values from', type=str, default='') # unused
    parser.add_argument('--logfile', help='file to write progress to', type=str)
    parser.add_argument('--configfile', help='optional file to load model parameters from', type=str, default='')

    # Model parameters
    parser.add_argument('--model', help='type of RNN', type=str, default='lstm')
    parser.add_argument('--d_hid', help='hidden layer size', type=int, default=650)
    parser.add_argument('--nlayers', help='number of layers of rnn', type=int, default=2)
    parser.add_argument('--drop_prob', help='probability of dropout', type=float, default=0.5)
    parser.add_argument('--init_scale', help='range to initialize over', type=float, default=.05)

    # Training parameters
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=1.)
    parser.add_argument('--decay_after', help='number of epochs after which \
                                            to begin decaying learning rate. \
                                            If none, then halve learning rate if \
                                            validation score increases between epochs. ', type=int, default=-1)
    parser.add_argument('--decay_rate', help='rate to decay learning rate', type=float, default=1.2)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int, default=40)
    parser.add_argument('--grad_reg', help='type of gradient regularization', type=str, default='norm') # unused
    parser.add_argument('--max_grad_norm', help='maximum gradient value', type=float, default=5.)
    parser.add_argument('--normalize', help='1 if normalize, 0 otherwise', type=int, default=0) # unused

    # Visualization
    parser.add_argument('--visualize', help='1 if visualize, 0 otherwise', type=int, default=0) # unused
    parser.add_argument('--vocabfile', help='path to pickle file containing vocabulary', type=str) # unused

    args = parser.parse_args(arguments)
    if not args.datafile and args.logfile:
        raise ValueError("Must include path to data and logfile")
    lf = open(args.logfile, 'w')
    config = tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
            intra_op_parallelism_threads=NUM_THREADS)
#    with tf.Graph().as_default(), tf.Session(config=config) as sess:
    with tf.Graph().as_default(), tf.Session() as sess:
        log(lf, "Loading data...")
        data, mp, tp = build_data(args)

        # get model, use train and valid models because of dropout
        log(lf, "Building model...") # TODO add saver
        initializer = tf.random_uniform_initializer(-mp['init_scale'], mp['init_scale'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = ops.Model(mp, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = ops.Model(mp)
        tf.initialize_all_variables().run()

        if args.modelfile:
            saver.restore(sess, args.modelfile)
            log(lf, "Model restored from %s" % args.modelfile)

        # train and validate
        log(lf, "Beginning training using %d threads..." % NUM_THREADS)
        last_valid = BIG_NUM
        learning_rate = tp['learning_rate']
        train_model.assign_lr(sess, learning_rate)
        valid_loss = run_epoch(sess, data['valid'], valid_model, tf.no_op())
        print 'Initial valid loss: %.3f' % valid_loss
        for epoch in xrange(tp['nepochs']):
            log(lf, "Training epoch %d w/ learning rate %.3f..." % (epoch+1, learning_rate))
            start_time = time.time()
            train_loss = run_epoch(sess, data['train'], train_model, train_model.train)
            valid_loss = run_epoch(sess, data['valid'], valid_model, tf.no_op())
            duration = time.time() - start_time
            log(lf, "\tloss = %.3f, valid ppl = %.3f, %.3f s" % \
                (train_loss, valid_loss, duration))

            if tp['decay_after'] > 0 and epoch >= tp['decay_after'] - 1: # cuz 0-indexing
                learning_rate /= tp['decay_rate']
                train_model.assign_lr(sess, learning_rate)
            elif last_valid < valid_loss:
                learning_rate /= 2.
                train_model.assign_lr(sess, learning_rate)
            elif args.outfile:
                saver.save(sess, args.outfile)
            last_valid = valid_loss

        # test
        log(lf, "Test perplexity: %.3f" % (run_epoch(sess, data['test'], valid_model, tf.no_op())))

        if args.visualize:
            log(lf, "Visualizing embeddings")
            visualize(args, embeddings)
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
