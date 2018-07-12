#coding utf-8

import tensorflow as tf

w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
MOVING_AVERAGE_DECAY = 0.99

# 实例化滑动平均类
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)

ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 打印出当前参数w1和w1的平均值
    print(sess.run([w1, ema.average(w1)]))    

    # 参数w1的值赋为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新step和w1的值 模拟出100轮迭代后 参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))