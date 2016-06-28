import theano, pickle, cPickle, h5py, lasagne, random, csv, gzip, pdb
import numpy as np
import theano.tensor as T         


# convert csv into format readable by rmn code
def load_data(span_path, metadata_path):
    wmap, cmap, bmap = pickle.load(open(metadata_path, 'rb'))
    #span_data = pickle.load(open(span_path, 'rb'))
    f = h5py.File(span_path, "r")
    subs = f['subs'][:]
    users = f['user'][:]
    spans = f['spans'][:]
    masks = (spans > 0).astype(int)
    spans = spans - 1
    max_len = spans.shape[1]

    revwmap = dict((v,k) for (k,v) in wmap.iteritems())
    revbmap = dict((v,k) for (k,v) in enumerate(bmap))
    revcmap = dict((v,k) for (k,v) in cmap.iteritems())

    assert len(users) == len(subs)
    assert len(users) == len(spans)
    break_pts = [0] # find places where the users/subs change
    for i in xrange(1, len(users)):
        if users[i] != users[i-1] or subs[i] != subs[i-1]:
            break_pts.append(i)
    break_pts.append(len(users))
    break_pts = sorted(list(set(break_pts)))
    data = [] # break the data up
    for i in xrange(1, len(break_pts)):
        next_pt = break_pts[i]
        cur_pt = break_pts[i-1]
        data.append([[subs[cur_pt]], [users[cur_pt]], \
            spans[cur_pt:next_pt,:], masks[cur_pt:next_pt,:]])

    return data, max_len, wmap, cmap, bmap


def generate_negative_samples(num_traj, span_size, negs, span_data):
    inds = np.random.randint(0, num_traj, negs)
    neg_words = np.zeros((negs, span_size)).astype('int32')
    neg_masks = np.zeros((negs, span_size)).astype('float32')
    for index, i in enumerate(inds):
        rand_ind = np.random.randint(0, len(span_data[i][2]))
        neg_words[index] = span_data[i][2][rand_ind]
        neg_masks[index] = span_data[i][3][rand_ind]

    return neg_words, neg_masks

