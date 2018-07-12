#coding utf-8
# 指数衰减学习率
import tensorflow as tf

LEARNING_RATE_BASE = 0.1    # 最初学习率
LEARNING_RATE_DECAY = 1.1  # 学习率衰减率
LEARNING_RATE_STEP = 1      #喂入多少轮BATCH_SIZE后 更新一次学习率  一般认为：总样本数/BATCH_SIZE

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w+1)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After {0} steps, global_sep:{1}, weight:{2:.5f}, learning_rate:{3:.5f}, loss:{4:.5f}".format(i, global_step_val, w_val, learning_rate_val, loss_val))
