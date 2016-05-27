import numpy as np
import tensorflow as tf
import pdb

# model: softmax(W2*tanh(W1*x)) where x are embeddings
def model(d_hid, d_emb):
    embeddings = tf.Variable(tf.random_uniform([V, d_emb], -1.0, 1.0))
    embeds = tf.nn.embedding_lookup(embeddings, inputs) # will want to load pretrained
    with tf.name_scope('linear1'):
        weights = tf.Variable(tf.random_uniform([d_emb, d_hid], -1.0, 1.0), name='weights')
        biases = tf.Variable(tf.zeros([d_hid], name='biases'))
        linear1 = tf.matmul(embeds, weights)+biases
    activation = tf.nn.tanh(linear1)
    with tf.name_scope('linear2'):
        weights = tf.Variable(tf.random_uniform([d_hid, V], -1.0, 1.0), name='weights')
        biases = tf.Variable(tf.zeros([V], name='biases'))
        linear2 = tf.matmul(activation, weights)+biases
    return linear2

# takes in logits (pre-softmax normalized scores)
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

# TODO normalize embeddings; clip/normalize gradient?; decay learning rate
def train(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss) # TODO what does this do
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False) # counter for number of batches
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# uses the loss score to compute perplexity
def validate(cross_entropy):
    ppl = tf.exp(tf.reduce_mean(cross_entropy, name='ppl'))
    return ppl
