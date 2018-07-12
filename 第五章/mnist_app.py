#coding utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import mnist_forward
import mnist_backward


def restore_model(testPicArr=None):
    with tf.Graph().as_default() as tg:                                     # 重现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])    # 仅需给x占位
        y = mnist_forward.forward(x, None)                                  # 计算求得输出y
        preValue = tf.argmax(y, 1)                                          # 将y压缩为一个值

        # 实例化带滑动平均值的saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found!")
                return -1

def pre_pic(picName=None):
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50

    # 图片反色
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    # nm_arr = np.reshape(im_arr, (-1, 784))

    nm_arr = im_arr.astype(np.float32)
    nm_arr = im_arr.reshape([1, 784])
    
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    
    return img_ready

def application():
    testNum = input("input the number of test pictures:")
    # testNum = "1"
    for i in range(int(testNum)):
        testPic = input("thie path of test picture:")
        # testPic = "pic/0.png"
        testPicArr = pre_pic(testPic)
        print(testPicArr)
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)

def main():
    application()

if __name__ == '__main__':
    main()