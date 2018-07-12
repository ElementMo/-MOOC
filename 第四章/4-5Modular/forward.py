#coding utf-8
import tensorflow as tf

def get_weight(shape=None, regularizer=None):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape=None):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b

def forward(x=None, regularizer=None):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2      # 输出层不激活

    return y