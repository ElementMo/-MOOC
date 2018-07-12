#coding utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 120
TOTAL_DATA = 1200
STEPS = 40000
seed = 2

rdm = np.random.RandomState(seed)
X = rdm.randn(TOTAL_DATA, 2)
Y_= [int(x0*x0 + x1*x1<2) for (x0,x1) in X]   # 分类随机值
Y_c = [['red' if y else 'blue'] for y in Y_]    # 标注颜色

# 整理数据集的形状
X = np.vstack(X).reshape(-1, 2)  # 注意 这里不可以直接调用X的reshape方法  因为X此时为List类型  需要放入ndarray才可调用reshape方法
Y_= np.vstack(Y_).reshape(-1, 1)
# print(X)
# print(Y_)
# print(Y_c)
# print(X[:,0])
# print(X[:,1])
# print(np.squeeze(Y_c))

# plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
# plt.show()

def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))    # 'losses'为自定义的集合set()
    return w

def get_bias(shape):
    b =  tf.Variable(tf.constant(0.01, shape=shape))
    return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.nn.relu(tf.matmul(y1, w2)+b2)

loss_mse = tf.reduce_mean(tf.square(y-y_))                      # 均方误差的损失函数 (不含正则化)
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))   # 均方误差损失函数 + 正则化w的损失

# 反向传播算法 不包含正则化
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)

#     for i in range(STEPS):
#         start = (i*BATCH_SIZE) % TOTAL_DATA
#         end = start + BATCH_SIZE
#         sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
#         if i % 2000 == 0:
#             loss_mse_val = sess.run(loss_mse, feed_dict={x:X, y_:Y_})
#             print("After {0} steps, loss:{1}".format(i,loss_mse_val))
#     xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]     # 构建网格坐标点
#     grid = np.c_[xx.ravel(), yy.ravel()]        # 
#     probs = sess.run(y, feed_dict={x:grid})
#     probs = probs.reshape(xx.shape)
#     print("w1:",sess.run(w1))
#     print("b1:",sess.run(b1))
#     print("w2:",sess.run(w2))
#     print("b2:",sess.run(b2))

# plt.scatter(X[:,0], X[:,1], s=2, c=np.squeeze(Y_c))
# plt.contour(xx, yy, probs)
# plt.show()


# 反向传播算法 包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % TOTAL_DATA
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
        if i % 2000 == 0:
            loss_val = sess.run(loss_total, feed_dict={x:X, y_:Y_})
            print("After {0} steps, loss{1}".format(i, loss_val))
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)
    print("w1:", sess.run(w1))
    print("b1:", sess.run(b1))
    print("w2:", sess.run(w2))
    print("b2:", sess.run(b2))

plt.scatter(X[:,0], X[:,1], s=2, c=np.squeeze(Y_c))
plt.contour(xx, yy, probs)
plt.show()

