#coding utf-8
import tensorflow as tf

w = tf.Variable(tf.constant(20, dtype=tf.float32))
loss = tf.square(w+1)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After {0} steps, w{1:.5f}, loss:{2:.5f}".format(i,w_val, loss_val))
