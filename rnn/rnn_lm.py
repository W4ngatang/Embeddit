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
    df = h5py.File(args.datafile, 'r')
    train_inputs = df['train_inputs']
    train_targs = df['train_targets']
    valid_inputs = df['valid_inputs']
    valid_targs = df['valid_targets']
    test_inputs = df['test_inputs']
    train = Dataset(train_inputs, train_targs)
    valid = Dataset(valid_inputs, valid_targs)
    test = Dataset(test_inputs)
    # there has to be a better way to return parameters...
    return {'train':train, 'valid':valid, 'test':test}, \
            {'batch_size':df['batch_size'][0], 'seq_len':df['seq_len'][0], \
                'vocab_size':df['vocab_size'][0], 'd_hid':args.d_hid, \
                'model':args.model, 'max_grad_norm':args.max_grad, \
                'drop_prob':args.drop_prob, 'nlayers':args.nlayers}

def run_epoch(sess, data, model, eval_op):
    loss = 0
    state = [] 
    for c,m in model.initial_state: # initial_state: (c_i, m_i) * nlayers
        state.append((c.eval(), m.eval()))
    for i in xrange(data.nbatches):
        fetches = [model.loss, eval_op]
        for c,m in model.last_state:
            fetches.append(c)
            fetches.append(m)

        input_batch, targ_batch = data.batch(i)
        batch_d = {model.input:input_batch, model.targ:targ_batch}
        for i, (c,m) in enumerate(model.initial_state):
            batch_d[c], batch_d[m] = state[i] 

        res = sess.run(fetches, batch_d)
        loss += res[0]
        state_flat = res[2:] # states returned flattened
        state = [state_flat[i:i+model.nlayers] for i in \
                    xrange(0, len(state_flat), model.nlayers)]

    return math.exp(loss/data.nbatches)

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

    # Model parameters
    parser.add_argument('--model', help='type of RNN', type=str, default='lstm')
    parser.add_argument('--d_hid', help='hidden layer size', type=int, default=200)
    parser.add_argument('--nlayers', help='number of layers of rnn', type=int, default=2)
    parser.add_argument('--drop_prob', help='probability of dropout', type=float, default=0.0)
    parser.add_argument('--init_scale', help='range to initialize over', type=float, default=.1)
    parser.add_argument('--w2v', help='1 if load w2v vectors, 0 if no', type=int, default=0)

    # Training parameters
    parser.add_argument('--learning_rate', help='initial learning rate', type=float, default=1.)
    parser.add_argument('--nepochs', help='number of epochs to train for', type=int, default=13)
    parser.add_argument('--grad_reg', help='type of gradient regularization (either norm or clip)', type=str, default='norm')
    parser.add_argument('--max_grad', help='maximum gradient value', type=float, default=5.)
    parser.add_argument('--normalize', help='1 if normalize, 0 otherwise', type=int, default=0)

    # Visualization
    parser.add_argument('--visualize', help='1 if visualize, 0 otherwise', type=int, default=0)
    parser.add_argument('--vocabfile', help='path to pickle file containing vocabulary', type=str)
    args = parser.parse_args(arguments)
    if not args.datafile:
        raise ValueError("Must include path to data")

    config = tf.ConfigProto(inter_op_parallelism_threads=NUM_THREADS,\
            intra_op_parallelism_threads=NUM_THREADS)
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        # load data and model parameters
        print "Loading data..."
        data, params = build_data(args)

        # get model, use train and valid models because of dropout
        print "Building model..." # TODO add saver
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = ops.Model(params, is_training=True)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = ops.Model(params)
        tf.initialize_all_variables().run()

        if args.modelfile:
            saver.restore(sess, args.modelfile)
            print "Model restored from %s" % args.modelfile

        # train and validate
        print "Beginning training using %d threads..." % NUM_THREADS
        last_valid = BIG_NUM
        learning_rate = args.learning_rate
        train_model.assign_lr(sess, learning_rate)
        for epoch in xrange(args.nepochs):
            print "Training epoch %d w/ learning rate %.3f..." % (epoch, learning_rate)
            start_time = time.time()
            train_loss = run_epoch(sess, data['train'], train_model, train_model.train)
            valid_loss = run_epoch(sess, data['valid'], valid_model, tf.no_op())
            duration = time.time() - start_time
            print "\tloss = %.3f, valid ppl = %.3f, %.3f s" % \
                (train_loss, valid_loss, duration)
            if epoch >= 4:
                learning_rate /= 2.
                train_model.assign_lr(sess, learning_rate)

            '''
            if last_valid < valid_loss:
                "\tHalving learning rate"
                learning_rate /= 2.
                train_model.assign_lr(sess, learning_rate)
            elif args.outfile:
                saver.save(sess, args.outfile)
            last_valid = valid_loss
            '''

        # test
        print "Test perplexity: %.3f" % (run_epoch(sess, data['test'], valid_model, tf.no_op()))

        if args.visualize:
            print "Visualizing embeddings"
            visualize(args, embeddings)
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
