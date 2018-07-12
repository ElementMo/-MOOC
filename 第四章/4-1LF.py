#coding utf-8

import tensorflow as tf
import numpy as np

BATCH_SIZE = 2
SEED = 23345

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_= [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y = tf.matmul(x, w1)

loss_mse = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 2 00 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            # print("After {0} steps, weights:\n{1}".format(i, sess.run(w1)))
            print("After {0} steps, loss:{1}".format(i, total_loss))