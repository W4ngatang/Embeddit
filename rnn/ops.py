import numpy as np
import tensorflow as tf
import pdb

class Model():

    def __init__(self, params, is_training=False):
        self.batch_size = batch_size = params['batch_size']
        self.seq_len = seq_len = params['seq_len']
        self.vocab_size = vocab_size = params['vocab_size']
        self.d_hid = d_hid = params['d_hid'] # also embedding dimension
        self.nlayers = nlayers = params['nlayers']
        self.drop_prob = drop_prob = params['drop_prob']
        
        self._input_ph = tf.placeholder(tf.int32, shape=[None, seq_len]) # None can be any size
        self._target_ph = tf.placeholder(tf.int32, shape=[None, seq_len]) 

        embed_table = tf.get_variable("embedding", [vocab_size, d_hid])
        embeddings = tf.nn.embedding_lookup(embed_table, self._input_ph)
        shaped_embeds = tf.unpack(tf.transpose(embeddings, [1,0,2])) # IF FETCHING THIS OP, NO LIST
        self.shaped_embeds = shaped_embeds

        if params['model'] == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell(d_hid, state_is_tuple=True) # TODO forget bias?
        elif params['model'] == 'gru':
            cell = tf.nn.rnn_cell.GRUCell(d_hid)
        elif params['model'] == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell(d_hid)
        else:
            raise ValueError("Cell type not supported")
        if is_training and drop_prob > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.-drop_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*nlayers, state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.rnn(cell, shaped_embeds, dtype=tf.float32,\
                                    initial_state=self._initial_state)
        self._last_state = state

        W = tf.get_variable("W", [d_hid, vocab_size])
        b = tf.get_variable("b", [vocab_size])
        shaped_outputs = tf.reshape(tf.concat(1, outputs), [-1, d_hid])
        logits = tf.matmul(shaped_outputs, W) + b 

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits], # hacky way to compute loss that reshapes the entire batch into one long array
            [tf.reshape(self._target_ph, [-1])],
            [tf.ones([batch_size * seq_len])], # weights; should be same shape as above 
            average_across_timesteps=True)
        self._loss = loss = tf.reduce_sum(loss) / batch_size / seq_len

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
