import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import pdb

class Model():

    def __init__(self, params):
        self.batch_size = batch_size = params.batch_size
        self.seq_len = seq_len = params.seq_len
        self.vocab_size = vocab_size = params.vocab_size
        self.d_hid = d_hid = params.d_hid
        self.n_layers = n_layers = params.n_layers
        self.drop_prob = params.drop_prob # TODO implement dropout
        
        self.input_ph = tf.placeholder(tf.int32, shape=[]) # TODO what does None for a dimension do?
        self.target_ph = tf.placeholder(tf.int32, shape=[]) # make these methods?

        # TODO tf.device??
        embed_table = tf.get_variable("embedding", [vocab_size, d_hid])
        embeddings = tf.nn.embedding_lookup(embed_table, self.input_ph)
        shaped_embeds = # TODO reshape embeddings using tf matrix manipulation

        if params.cell_type == 'lstm':
            model_cell = rnn_cell.BasicLSTMCell(d_hid) # TODO forget bias?
        elif params.cell_type == 'gru':
            model_cell = rnn_cell.GRUCell(d_hid)
        elif params.cell_type == 'rnn':
            model_cell = rnn_cell.BasicRNNCell(d_hid)
        else:
            raise ValueError("Cell type not supported")
        #cell = tf.nn.rnn_cell.MultiRNNCell([model_cell] * params.n_layers)
        # TODO dropout, initial state
        outputs, states = rnn.rnn(cell, shaped_embeds, dtype=tf.float32) # TODO is dtype input or out?
        # TODO reshape outputs?

        W = tf.get_variable("W", [d_hid, vocab_size])
        b = tf.get_variable("b", [vocab_size])
        logits = tf.matmul(outputs, W) + b)
        # compute loss, returns log perplexity (NLL)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [targets], # TODO reshape these
            [tf.ones()]) # weights; TODO shape correctly

        # if training , get gradient of loss wrt parameters
        self.learn_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grad, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), params.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
