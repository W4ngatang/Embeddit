import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
import pickle
import re
import argparse
import pdb
from collections import defaultdict, Counter

path = '/dfs/scratch0/wleif/Reddit/clean_comments/'
subreddits = os.listdir(path)
subreddits = ['funny', 'leagueoflegends', 'AdviceAnimals', 'pics', 'nfl']

#%%timeit -n1 -r1
print_every = 1

usr2ind = {}
ind2usr = {}
n_users = 0

sub2ind = {}
ind2sub = {}
n_subs = len(subreddits)

users = {}

'''
sample = np.random.random_integers(0, len(subreddits), nsubs)
print sampled_subreddits
for i, sub_idx in enumerate(sample):
    subreddit = subreddits[sub_idx]
    sub2ind[subreddit] = sub_idx
    ind2sub[sub_idx] = subreddit
'''

for i, subreddit in enumerate(subreddits):
    sub2ind[subreddit] = i
    ind2sub[i] = subreddit
    with open(path+subreddit+'.tsv', 'r') as df:
        #raw_data = np.genfromtxt(df, dtype=str, delimiter='\t')
        for row in df:
            user = row.split('\t')[-5]
            if user not in usr2ind:
                ind2usr[n_users] = user
                usr2ind[user] = n_users
                users[n_users] = defaultdict(int)
                n_users += 1
            users[usr2ind[user]][i] += 1

    
    if i % print_every == 0:
        print "Finished %d" % (i+1)


'''

Some analytics

'''

counter = [0.0] * n_subs
r_counter = [0.0] * n_subs
n_posts = 0
thresh = 3
user_pool = []

def robust_count(d, n_spans = 5):
    return sum([1 if v > n_spans else 0 for v in [v for v in d.values()] ])
        
for k, v in users.iteritems():
    n_posts += sum([uv for _, uv in v.iteritems()])
    for i in xrange(n_subs):
        if len(v) > i:
            counter[i] += 1
        if robust_count(v) > i:
            r_counter[i] += 1
    if robust_count(v) > thresh:
        user_pool.append(k)
percents = [n / n_users for n in r_counter]
for i, percent in enumerate(percents):
    print "nsubs: %d, percent: %.3f, robust count: %d, full count: %d" % (i, percent, r_counter[i], counter[i])

user_pool = set(user_pool)
print "Total number of users: %d, Number of selected users: %d" % (n_users, len(user_pool))
print "Total number of posts: %d" % n_posts


'''

Get the posts for the users that selected population of users (user_pool)

'''

print_every = [1.,1.]
posts = [[[] for x in xrange(n_users)] for y in xrange(n_subs)]

try:
    for i, subreddit in enumerate(subreddits):
        nrows = sum(1 for row in open(path+subreddit+'.tsv', 'r'))
        with open(path+subreddit+'.tsv', 'r') as df:
            sub_ind = sub2ind[subreddit]
            for j, row in enumerate(df):
                data = row.split('\t')
                user = usr2ind[data[-5]]
                if user in user_pool:
                    posts[sub_ind][user].append((data[-1], data[1]))

                if j % int(nrows/print_every[0]) == 0: ## TODO: Fix logging...
                    print '\t%.2f' % (j/float(nrows))

        if i % print_every[1] == 0:
            print "Finished %d" % (i+1)
except Exception as e:
    pdb.set_trace()


'''

Build vocab

'''


outfile = '/dfs/scratch0/wangalex/rmn/reddit5'

freqs = defaultdict(int)
doc_freqs = {}
special = ['<EOS>', '<UNK>', '<SPECIAL>', '<URL>']
lens = []
max_len = 0

print "Sorting by date and gathering vocab..."
for i in xrange(n_subs):
    for j in xrange(n_users):
        if posts[i][j] is []:
            continue
        posts[i][j].sort(key=lambda tup:tup[-1])
        for post,_ in posts[i][j]:
            clean_words = re.sub(r"[0-9]{1,}", "<NUM>", re.sub(r"<(.+?)>", "", post)).strip().split()
            for word in clean_words:
                if word not in doc_freqs:
                    doc_freqs[word] = [0 for x in xrange(nsubs)]
                doc_freqs[word][i] = 1
                freqs[word] += 1
            lens.append(len(clean_words))
            max_len = max(max_len, len(clean_words))

'''

Prune vocab
    - break up into cells because each operation is pretty costly
    
'''

min_doc_appearances = 4
remove_top_k = 500 # TODO: maybe up this?
max_vocab_size = .5

# pruning vocab borrowed from Yoon Kim
vocab = [(word, count) for word, count in freqs.iteritems()]
vocab.sort(key = lambda x: x[1], reverse = True)
doc_freqs = {word:sum(freq) for word, freq in doc_freqs.iteritems()}
no_common_words = [pair[0] for pair in vocab[remove_top_k:]]
min_doc_words = filter(lambda x: doc_freqs[x] >= min_doc_appearances, no_common_words)
if max_vocab_size <= 1:
    stop_pt = int(max_vocab_size * len(min_doc_words))
else:
    stop_pt = max_vocab_size - len(special)
pruned = min_doc_words[:stop_pt]

word2ind = {}
ind2word = {}
ind = 1 # start with 1 for easy masking
for word in special+pruned:
    word2ind[word] = ind
    ind2word[ind] = word
    ind += 1

print "Writing vocab..."
with open(outfile+'.vocab.txt', 'w') as f:
    f.write("Word Index Count DocFreq\n")
    words = [(word, idx) for word, idx in word2ind.iteritems()]
    words.sort(key = lambda x: x[1])
    for word, idx in words:
        if word in freqs:
            f.write("%s %d %d %d\n" % (word, idx, freqs[word], doc_freqs[word]))
        else:
            f.write("%s %d %d %d\n" % (word, idx, -1, -1))

with open(outfile+'.vocab.pkl', 'w') as f:
    pickle.dump((word2ind, ind2word), f)

print '\tFull vocab size: %d, pruned vocab size: %d' % (len(vocab), len(word2ind))

'''

Generate the data

'''

span_data = []
#mask_data = [] # masks to be constructed in the train now 
sub_data = []
user_data = []

max_len = min(max_len, 116)

lengths = []
unks = []

unk = word2ind['<UNK>']

for i in xrange(n_subs):
    #sub = np.array([i], dtype=np.int32)
    for j in xrange(n_users):
        if not posts[i][j]:
            continue
        #user = np.array([j], dtype=np.int32)
        spans = []
        masks = []
        for post,_ in posts[i][j]:
            clean_words = re.sub(r"[0-9]{1,}", "<NUM>", re.sub(r"<(.+?)>", "", post)).strip().split()
            span = [word2ind[word] if word in word2ind else unk for word in clean_words[:max_len]]
            
            # analytics: number of unknowns and length of phrases
            unks.append(sum(filter(lambda x: x == unk, span)))
            lengths.append(len(span))
            
            #mask = [1] * len(span) + [0] * (max_len-len(clean_words))
            span += [0]*(max_len-len(clean_words))
            #spans.append(span)
            #masks.append(mask)
            sub_data.append(i)
            user_data.append(j)
            span_data.append(span)
            #mask_data.append(masks)

'''
sub_data = np.array(sub_data, dtype=np.int32)        
user_data = np.array(user_data, dtype=np.int32)
span_data = np.array(span_data, dtype=np.int32)
mask_data = np.array(mask_data, dtype=np.int32)
'''

'''

Write data to pickles and stuff

'''
metadata_path = '/dfs/scratch0/wangalex/rmn/reddit5_meta.pkl'
span_path = '/dfs/scratch0/wangalex/rmn/reddit5_spans.hdf5'
pickle.dump((word2ind, usr2ind, sub2ind), open(metadata_path, 'wb'))
#pickle.dump(span_data, open(span_path, 'wb'))

f = h5py.File(span_path, "w")
f['subs'] = sub_data
f['user'] = user_data
f['spans'] = span_data
#f['masks'] = mask_data # can save space by doing this in train function
f.close()

'''

Build word2vec pretrained embeddings

'''

word_vecs = {}
vec_file = '/dfs/scratch0/gigawordvecs/GoogleNews-vectors-negative300.bin'
word2vec_path = '/dfs/scratch0/wangalex/rmn/glove.We'

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
        if word in word2ind:  
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
        else:
            f.read(binary_len)

embed = np.random.uniform(-0.25, 0.25, (len(vocab), len(word_vecs.values()[0])))
embed[0] = 0
for word, vec in word_vecs.items():
    embed[word2ind[word]] = vec
print "\tLoaded %d vectors" % len(word_vecs)
#pickle.dump(embed, open(word2vec_path, 'wb'))
f = h5py.File('/dfs/scratch0/wangalex/rmn/w2v.hdf5', 'w')
f['w2v'] = embed
f.close()

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--srcfiles', help='comma separated (no spaces) paths to source data files', type=str)
    parser.add_argument('--n', help='gram size', type=int, default=5)
    parser.add_argument('--split', help='train size as fraction of data', type=float, default=.8)
    parser.add_argument('--outfile', help='file prefix to write to (hdf5)', type=str)
    parser.add_argument('--vocab_size', help='max vocab size, either an absolute number or \
                                                a percentage', type=float, default=1.)
    parser.add_argument('--w2v', help='path to w2v binary', type=str, default='')
    args = parser.parse_args(arguments)
    if not args.srcfiles or not args.outfile:
        raise ValueError("srcfiles or outfile not provided")

    word2ind, ind2word = build_vocab(args)

    if args.w2v:
        print "Loading vectors from word2vec..."
        embeds = build_embeds(args.w2v, word2ind)

    print "Generating data..."
    all_inputs, all_targs = build_data(args.srcfiles, word2ind, args.n-1) # do n-1 b/c lazy indexing
    for i, (inputs, targs) in enumerate(zip(all_inputs, all_targs)):
        split_pt = int(inputs.shape[0]*args.split)
        if len(all_inputs) == 1:
            filename = args.outfile + '.hdf5'
        else:
            filename = args.outfile + '.' + str(i) + '.hdf5'

        with h5py.File(filename) as f:
            f['train_inputs'] = inputs[:split_pt]
            f['train_targets'] = targs[:split_pt]
            f['valid_inputs'] = inputs[split_pt:]
            f['valid_targets'] = targs[split_pt:]
            f['gram_size'] = np.array([args.n], dtype=np.int32)
            f['vocab_size'] = np.array([len(word2ind)], dtype=np.int32)
            if args.w2v:
                f['embeds'] = embeds
         
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
