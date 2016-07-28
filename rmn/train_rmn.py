from stanza.cluster import pick_gpu
pick_gpu.bind_theano()
#from theano.compile.debugmode import DebugMode
#import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0')
import cPickle, h5py, lasagne, random, csv, gzip, time
import argparse, sys
import numpy as np
import theano.tensor as T 
from layers import *
from util import *        
import pdb

# assemble the network
def build_rmn(d_word, d_char, d_book, len_voc, 
    num_descs, num_chars, num_books, span_size, We, 
    freeze_words=True, eps=1e-5, lr=0.01, negs=10):

    # input theano vars
    in_spans = T.imatrix(name='spans')
    in_neg = T.imatrix(name='neg_spans')
    in_chars = T.ivector(name='chars')
    in_book = T.ivector(name='books')
    in_negbooks = T.imatrix(name='neg_books')
    in_currmasks = T.matrix(name='curr_masks')
    in_dropmasks = T.matrix(name='drop_masks')
    in_negmasks = T.matrix(name='neg_masks')

    # define network
    l_inspans = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_spans)
    l_inneg = lasagne.layers.InputLayer(shape=(negs, span_size), 
        input_var=in_neg)
    l_inchars = lasagne.layers.InputLayer(shape=(1, ), # used to be shape=(1, )
        input_var=in_chars)
    l_inbook = lasagne.layers.InputLayer(shape=(1, ), 
        input_var=in_book)
    l_currmask = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_currmasks)
    l_dropmask = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_dropmasks)
    l_negmask = lasagne.layers.InputLayer(shape=(negs, span_size), 
        input_var=in_negmasks)

    # negative examples should use same embedding matrix
    l_emb = MyEmbeddingLayer(l_inspans, len_voc, 
        d_word, W=We, name='word_emb')
    l_negemb = MyEmbeddingLayer(l_inneg, len_voc, 
            d_word, W=l_emb.W, name='word_emb_copy1')

    # freeze embeddings
    if freeze_words:
        l_emb.params[l_emb.W].remove('trainable')
        l_negemb.params[l_negemb.W].remove('trainable')

    l_chars = lasagne.layers.EmbeddingLayer(\
        l_inchars, num_chars, d_char, name='char_emb')
    l_books = lasagne.layers.EmbeddingLayer(\
        l_inbook, num_books, d_book, name='book_emb')

    # average each span's embeddings
    l_currsum = AverageLayer([l_emb, l_currmask], d_word, no_average=False) # used for reconstruction
    l_dropsum = AverageLayer([l_emb, l_dropmask], d_word, no_average=False) # actual inputs
    l_negsum = AverageLayer([l_negemb, l_negmask], d_word, no_average=False)

    # pass all embeddings thru feed-forward layer
    l_mix = MixingLayer([l_dropsum, l_chars, l_books],
        d_word, d_char, d_book)

    #l_currsum = AverageLayerWithOffset([l_emb, l_currmask, l_books], d_word)
    #l_mix = MixingLayerNoBook([l_dropsum, l_chars, l_books], d_word, d_char)

    # compute recurrent weights over dictionary
    l_rels = RecurrentRelationshipLayer(\
        l_mix, d_word, num_descs)

    # multiply weights with dictionary matrix
    l_recon = ReconLayer(l_rels, d_word, num_descs)

    # compute loss
    currsums = lasagne.layers.get_output(l_currsum)
    negsums = lasagne.layers.get_output(l_negsum)
    recon = lasagne.layers.get_output(l_recon)

    currsums /= currsums.norm(2, axis=1)[:, None]
    recon /= recon.norm(2, axis=1)[:, None]
    negsums /= negsums.norm(2, axis=1)[:, None]
    correct = T.sum(recon * currsums, axis=1)
    negs = T.dot(recon, negsums.T)
    loss = T.sum(T.maximum(0., 
        T.sum(1. - correct[:, None] + negs, axis=1)))

    # enforce orthogonality constraint
    norm_R = l_recon.R / l_recon.R.norm(2, axis=1)[:, None]
    ortho_penalty = eps * T.sum((T.dot(norm_R, norm_R.T) - \
        T.eye(norm_R.shape[0])) ** 2)
    loss += ortho_penalty
    pdb.set_trace()

    all_params = lasagne.layers.get_all_params(l_recon, trainable=True)
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    traj_fn = theano.function([in_chars, in_book, 
        in_spans, in_dropmasks], 
        lasagne.layers.get_output(l_rels))
    train_fn = theano.function([in_chars, in_book, 
        in_spans, in_currmasks, in_dropmasks,
        in_neg, in_negmasks], 
        [loss, ortho_penalty], updates=updates)#, mode='DebugMode')
    return train_fn, traj_fn, l_recon

def log_output(lf, message):
    print message
    lf.write(message+'\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prefix', help='prefix to things', type=str, \
                            default='')
    parser.add_argument('--spanfile', help='span data', type=str, default='spans.hdf5')
    parser.add_argument('--metafile', help='meta data', type=str, default='meta.pkl')
    parser.add_argument('--w2v', help='path to w2v', type=str, default='ccrawl-vecs.hdf5')
    parser.add_argument('--subreddit_init', help='path to subreddit init file', type=str, \
            default='ccrawl-vecs.hdf5')
    parser.add_argument('--logfile', help='file to write progress to', type=str, default='baseline')
    parser.add_argument('--configfile', help='optional file to load model parameters from', type=str, default='')

    # Model parameters
    parser.add_argument('--d_char', help='hidden layer size', type=int, default=300)
    parser.add_argument('--d_book', help='hidden layer size', type=int, default=300)
    parser.add_argument('--p_drop', help='probability of word dropout', type=float, default=0.75)
    parser.add_argument('--num_descs', help='number of descriptors', type=int, default=20)
    parser.add_argument('--offset_objective', help='1 if subtract subreddit offsets from spans',\
            type=int, default=0)

    # Training parameters
    parser.add_argument('--lr', help='initial learning rate', type=float, default=.001)
    parser.add_argument('--n_epochs', help='number of epochs to train for', type=int, default=15)
    parser.add_argument('--num_negs', help='number of negative samples', type=int, default=50)
    parser.add_argument('--eps', help='uniqueness penalty', type=float, default=1e-3)

    args = parser.parse_args(sys.argv[1:])
    if not args.spanfile and args.metafile:
        raise ValueError("Must include path to span data, metadata, w2v, and log path")
    prefix = '/afs/cs.stanford.edu/u/wangalex/scr/rmn/' + args.prefix
    if prefix[-1] != '/': prefix += '/'
    lf = open(prefix+args.logfile+'.log', 'w')

    log_output(lf, 'loading data...')
    span_data, span_size, wmap, cmap, bmap = \
        load_data(prefix+args.spanfile, prefix+args.metafile)
    with h5py.File(prefix+args.w2v, 'r') as f:
        We = f['w2v'][:]
    norm_We = We / np.linalg.norm(We, axis=1)[:, None]
    We = np.nan_to_num(norm_We) # is it modifying the w2v in a weird way?
    descriptor_log = prefix+args.logfile+'.descriptors.log'
    trajectory_log = prefix+args.logfile+'.trajectories.log'

    # embedding/hidden dimensionality
    d_word = We.shape[1]
    d_char = args.d_char
    d_book = args.d_book
    p_drop = args.p_drop
    num_descs = args.num_descs
    offset = args.offset_objective

    # word dropout probability
    n_epochs = args.n_epochs
    lr = args.lr
    eps = args.eps
    num_negs = args.num_negs

    num_chars = len(cmap)
    num_books = len(bmap)
    num_traj = len(span_data)
    len_voc = len(wmap)
    revmap = {}
    for w in wmap:
        revmap[wmap[w]] = w

    log_output(lf, "d_w: %d, span_size: %d, n_descs: %d, n_users: %d, n_subs: %d" % \
        (d_word, span_size, num_descs, num_chars, num_books))
    log_output(lf, "eps: %.6f, n_negs: %d, lr: %.3f, p_drop: %.3f" % \
        (eps, num_negs, lr, p_drop))
    log_output(lf, 'n pairs: %d' % len(span_data))

    log_output(lf, 'compiling...')
    train_fn, traj_fn, final_layer = build_rmn(
        d_word, d_char, d_book, len_voc, num_descs, num_chars, 
        num_books, span_size, We, eps=eps, 
        freeze_words=True, lr=lr, negs=num_negs)
    log_output(lf, 'done compiling, now training...')

    # training loop
    min_cost = float('inf')
    costs = []
    for epoch in range(n_epochs):
        cost = 0.
        random.shuffle(span_data)
        start_time = time.time()

        lap_time = time.time()
        counter = 0.
        print_every = len(span_data) / 10
        test_pt = len(span_data) / 2

        for book, chars, curr, cm, in span_data:#[:test_pt]:
            ns, nm, nb = generate_negative_samples(\
                num_traj, span_size, num_negs, span_data)

            # word dropout
            drop_mask = (np.random.rand(*(cm.shape)) < (1 - p_drop)).astype('float32')
            drop_mask *= cm

            ex_cost, ex_ortho = train_fn(chars, book, curr, cm, drop_mask,
                ns, nm) # doesn't do anything with ex_ortho?
            cost += ex_cost

            counter += 1
            if not (counter % print_every):
                log_output(lf, '\tFinished %.2f%% in %d s' % \
                        (counter / len(span_data), time.time()-lap_time))
                lap_time = time.time()

        end_time = time.time()

        # save params if cost went down
        if cost < min_cost:
            min_cost = cost
            params = lasagne.layers.get_all_params(final_layer)
            p_values = [p.get_value() for p in params]
            p_dict = dict(zip([str(p) for p in params], p_values))
            cPickle.dump(p_dict, open(prefix+args.logfile+'.rmn_params.pkl', 'wb'),
                protocol=cPickle.HIGHEST_PROTOCOL)

            # compute nearest neighbors of descriptors
            R = p_dict['R']
            log = open(descriptor_log, 'w')
            for ind in range(len(R)):
                desc = R[ind] / np.linalg.norm(R[ind])
                sims = We.dot(desc.T)
                ordered_words = np.argsort(sims)[::-1]
                desc_list = [ revmap[w] for w in ordered_words[:10]]
                log.write(' '.join(desc_list) + '\n')
                log_output(lf, 'descriptor %d:' % ind)
                log_output(lf, ' '.join(desc_list))
            log.flush()
            log.close()

            # save relationship trajectories
            # TODO don't write trajectories until the end, after loading model parameters
            log_output(lf, 'writing trajectories...')
            tlog = open(trajectory_log, 'wb')
            traj_writer = csv.writer(tlog)
            traj_writer.writerow(['Subreddit', 'User', 'Span ID'] + \
                ['Topic ' + str(i) for i in range(num_descs)])
            for book, chars, curr, cm in span_data:
                c1 = cmap[chars[0]]
                bname = bmap[book[0]]

                # feed unmasked inputs to get trajectories
                traj = traj_fn(chars, book, curr, cm)
                for ind,step in enumerate(traj):
                    traj_writer.writerow([bname, c1, ind] + list(step))   
            tlog.flush()
            tlog.close()

        log_output(lf, 'done with epoch: %d, cost = %.3f, train time: %.2f, write time: %.2f' %
            (epoch, cost/len(span_data), end_time-start_time, time.time()-end_time))
        costs.append(cost/len(span_data))
    log_output(lf, "Costs:" + '\n'.join(map(str,costs)))
