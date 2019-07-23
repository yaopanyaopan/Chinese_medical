# coding=utf-8

import tensorflow as tf

def add_embedding(input_x,vectors):
    initial = tf.constant(vectors, dtype=tf.float32)
    WV = tf.get_variable('word_vectors', initializer=initial)
    with tf.variable_scope('encoding_layer'):
        wv = tf.nn.embedding_lookup(WV, input_x)
    return wv




