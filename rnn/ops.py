import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import pdb

class Model():

    def __init__(self, params):
        self.batch_size = batch_size = params.batch_size
        self.seq_len = seq_len = params.seq_len
        self.vocab_size = vocab_size = params.vocab_size
        self.d_hid = d_hid = params.d_hid # also embedding dimension
        self.n_layers = n_layers = params.n_layers
        self.drop_prob = params.drop_prob # TODO implement dropout
        
        self._input_ph = tf.placeholder(tf.int32, shape=[None, seq_len]) # None can be any size
        self._target_ph = tf.placeholder(tf.int32, shape=[None, seq_len]) 

        embed_table = tf.get_variable("embedding", [vocab_size, d_hid])
        embeddings = tf.nn.embedding_lookup(embed_table, self.input_ph)
        shaped_embeds = tf.unpack(tf.transpose(embeddings, [1,0,2])) # IF FETCHING THIS OP, NO LIST

        if params.cell_type == 'lstm':
            cell = rnn_cell.BasicLSTMCell(d_hid) # TODO forget bias?
        elif params.cell_type == 'gru':
            cell = rnn_cell.GRUCell(d_hid)
        elif params.cell_type == 'rnn':
            cell = rnn_cell.BasicRNNCell(d_hid)
        else:
            raise ValueError("Cell type not supported")
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        #cell = tf.nn.rnn_cell.MultiRNNCell([model_cell] * params.n_layers)
        # TODO dropout, multiRNN?, scope?
        outputs, states = rnn.rnn(cell, shaped_embeds, dtype=tf.float32,\
                                initial_state=self._initial_state,
                                sequence_length=seq_len)

        # TODO may need to fiddle with shapes
        W = tf.get_variable("W", [d_hid, vocab_size])
        b = tf.get_variable("b", [vocab_size])
        logits = tf.matmul(outputs, W) + b # should be list of len seq_len of sizes [batch_size x vocab_size]
        # TODO may need to fiddle with shapes
        # compute loss, returns log perplexity (NLL)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.target_ph, [-1])], # reshape to be an array of len batch_size * seq_len
            [tf.ones([batch_size * seq_len])]) # weights; should be same shape as above 

        self._loss = loss
        self._final_state = states

        # if training , get gradient of loss wrt parameters
        self._lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grad, _ = tf.gradients(cost, tvars)
        #grad, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), params.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value)

    @property
    def input(self):
        return self._input_data

    @property
    def targ(self):
        return self._targets

    @property(self):
    def initial_state(self):
        return self._initial_state
    
    @property
    def loss(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train(self):
        return self._train_op 
