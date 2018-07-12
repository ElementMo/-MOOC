#coding utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import mnist_forward
import mnist_backward


def restore_model(picArray=None):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

def pre_pic(picName=None):
    img = Image.open(picName)
    reIm = img.resize((28,28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50

    # 反色
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = im_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready

def application():
    testNum = input("input the number of test pictures:")
    for i in range(testNum):
        testPic = raw_input("thie path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)

def main():
    application()

if __name__ == '__main__':
    main()