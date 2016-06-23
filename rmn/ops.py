import numpy as np
import tensorflow as tf # why do we need to import below? try w/o
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.nn import relu
from tensorflow.python.ops.nn import softmax # may be very slow with softmax...
from tensorflow.python.ops.array_ops import concat
import pdb

class ConcatRNN(tf.nn.rnn_cell.RNNCell):
    # Modified simple RNN cell

    def __init__(self, num_units, input_size=None, alpha=.5):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._alpha = alpha # TODO make alpha a trainable parameter

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def alpha(self):
        return self.alpha

    def __call__(self, inputs, state, scope=None):
        with tf.python.ops.vs.variable_scope(scope or type(self).__name__):
            concated = concat(0, [inputs, state])
            mlp = relu(tf.nn.rnn_cell._linear(concated, self._num_units, False))
            output = self._alpha * softmax(mlp) + (1 - self._alpha) + state
        return output, output

class Model():

    def __init__(self, params, is_training=False):
        batch_size = params['batch_size'] # DATA WILL BE ARRANGED IN BATCHES
        seq_len = params['seq_len']
        span_size = params['span_size'] # number of tokens per span

        vocab_size = params['vocab_size']
        nchars = 
        nbooks = 
        ntopics =

        d_word = params['d_hid'] # also embedding dimension
        d_char
        d_book

        self.nlayers = nlayers = params['nlayers']
        drop_prob = params['drop_prob']
        
        # how to do in batch? if it's a user-subreddit pair / network, then can't
        # do batch because need sequential -> lay out each user-subreddit pair sequentially
        # TODO initializations
        # TODO dropout; masks?
        # TODO end of char x char token
        self._span = tf.placeholder(tf.int32, shape=[None, span_size])
        self._negs = tf.placeholder(tf.int32, shape=[n_negs, span_size]) 
        self._user = tf.placeholder(tf.int32, shape=[2]) # two characters?
        self._book = tf.placeholder(tf.int32, shape=[1])

        '''
        In each variable scope,
        Lookups + average/sum/reshape
        Linear + relu
            Note: original paper doesn't really concat, does it in this way
        '''
        with tf.variable_scope("span"):
            # TODO load w2v
            lookup_table = tf.get_variable("lookup_table", [vocab_size, d_word], \
                                            trainable=False) # seq x span_size
            word_embeds = tf.nn.embedding_lookup(lookup_table, self._span) # seq x span_size x d_w
            span_embeds = tf.reduce_mean(word_embeds, 1) # seq x d_word
            W_w = tf.get_variable('W', [d_word, d_word])
            b_w = tf.get_variable('b', [d_word]) # initialzed to zero
            lin_w = tf.matmul(W_w, span_embeds) + b_w

        with tf.variable_scope("char"):
            lookup_table = tf.get_variable("lookup_table", [nchars, d_char]) 
            char_embeds = tf.nn.embedding_lookup(lookup_table, self._user)
            shaped_char_embeds = tf.reduce_sum(char_embeds, [d_char, -1]) # d_char
            W_c = tf.get_variable('W', [d_char, d_word])
            lin_c = tf.matmul(W_c, shaped_char_embeds)

        with tf.variable_scope("book"):
            lookup_table = tf.get_variable("lookup_table", [nbooks, d_book]) # batch x seq
            book_embed = tf.nn.embedding_lookup(lookup_table, self._book) # batch x seq x d_b
            W_b = tf.get_variable('W', [d_book, d_word])
            lin_b = tf.matmul(W_b, book_embed)

        linear = lin_w + lin_c + lin_b
        h_t = tf.nn.relu(linear)
        # may need to reshape

        cell = ConcatRNN(d_word)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope("RNN"): 
            outputs, state = tf.nn.rnn(cell, all_embeds, dtype=tf.float32,\
                                    initial_state=self._initial_state)
        self._last_state = state
        # Output shape should be (seq_len x d_word)
        outputs = tf.reshape(tf.concat(1, outputs), [-1, d_word])

        # reconstruction
        R = tf.get_variable("descriptor_dict", [ntopics, d_word])
        recons = tf.matmul(R, outputs)

        # max-margin loss w/ similarity penalty
        pos_vecs = tf.tile(tf.reduce_sum(tf.mul(span_embeds, recons), 1), [n_negs, 1])
        neg_vecs = tf.matmul(neg_spans, tf.transpose(recons))
        J = tf.reduce_sum(tf.max(0., 1. - pos_vecs + neg_vecs))

        # The true penalty I think is this, but they use the uncommented line
        #X = tf.sqrt(tf.reduce_sum(tf.square(tf.matmul(R, tf.transpose(R)) - identity)))
        identity = tf.Variable(np.identity(ntopics), trainable=False)
        norm_R = tf.div(R, tf.sqrt(tf.reduce_sum(tf.square(R))))
        X = tf.reduce_sum(tf.square(tf.matmul(norm_R, tf.transpose(norm_R)) - identity))
        self._loss = J + unique_scale * X

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), params['max_grad_norm'])
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.debug = [logits]

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))

    @property
    def input(self):
        return self._input_ph

    @property
    def targ(self):
        return self._target_ph

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def last_state(self):
        return self._last_state
    
    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def train(self):
        return self._train_op 
