import numpy as np
import tensorflow as tf
import pdb

# model: softmax(W2*tanh(W1*x)) where x are embeddings
def model(inputs, params, pretrain=None):
    n = params['gram_size'] - 1 # -1 to get shape of context
    V = params['vocab_size']
    d_emb = params['emb_size']
    d_hid = params['hid_size']

    vars = []
    if pretrain is not None:
        print "\tUsing pretrained vectors..."
        embed_init = tf.constant(pretrain, dtype=np.float32, shape=[V, d_emb])
    else:
        embed_init = tf.random_uniform([V, d_emb], -5.0, 5.0)

    embeddings = tf.Variable(embed_init)
    vars.append(embeddings)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized = embeddings / norm

    embeds = tf.nn.embedding_lookup(normalized, inputs) 
    reshape = tf.reshape(embeds, [-1, n*d_emb], name='reshape')
    with tf.name_scope('linear1'):
        weights = tf.Variable(tf.random_uniform([n*d_emb, d_hid], -.1, .1), name='weights')
        biases = tf.Variable(tf.random_uniform([d_hid], -.1, .1), name='biases')
        linear1 = tf.matmul(reshape, weights)+biases
        vars += [weights, biases]
    activation = tf.nn.tanh(linear1)
    with tf.name_scope('linear2'):
        weights = tf.Variable(tf.random_uniform([d_hid, V], -.1, .1), name='weights')
        biases = tf.Variable(tf.random_uniform([V], -.1, .1), name='biases')
        linear2 = tf.matmul(activation, weights)+biases
        vars += [weights, biases]

    # ops to normalize embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized = embeddings / norm
    normalize = embeddings.assign(normalized)
    return linear2, normalize, vars

# takes in logits (pre-softmax normalized scores)
def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def train(loss, learning_rate_ph, args):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_ph)
    #train_op = optimizer.minimize(loss)
    grads_and_vars = optimizer.compute_gradients(loss)
    print_op = [tf.Print(grad,[grad]) for grad, var in grads_and_vars] # debugging...
    if args.grad_reg == 'norm':
        reg_grads_and_vars = [(tf.clip_by_norm(grad, args.max_grad), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(reg_grads_and_vars)
    elif args.grad_reg == 'clip':
        reg_grads_and_vars = [(tf.clip_by_value(grad, -args.max_grad, args.max_grad), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(reg_grads_and_vars)
    else:
        train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op, print_op

# uses the loss score to compute perplexity
# Note: not really used because need to average loss then take exp
def validate(loss):
    ppl = tf.exp(loss, name='ppl')
    return ppl
