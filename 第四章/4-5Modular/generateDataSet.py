#coding uft-8

import numpy as np
import matplotlib.pyplot as plt
seed = 2

def generateds(TOTAL_DATA=300):
    rdm = np.random.RandomState(seed)
    X = rdm.randn(TOTAL_DATA, 2)
    Y_= [int(x0*x0 + x1*x1<2) for (x0,x1) in X]
    Y_c = [['red' if y else 'blue'] for y in Y_]

    # 整理数据集的形状
    X = np.vstack(X).reshape(-1, 2)  # 注意 这里不可以直接调用X的reshape方法  因为X此时为List类型  需要放入ndarray才可调用reshape方法
    Y_= np.vstack(Y_).reshape(-1, 1)
    return X, Y_, Y_c

