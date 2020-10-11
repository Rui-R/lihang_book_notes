# coding=utf-8
# Date:2020-10-11

import numpy as np
import time

def load_data(file_name):
    print('start to load data!')
    data = []; label = []
    f = open(file_name, 'r')
    for line in f.readlines():
        curline = line.strip().split(',')
        if int(curline[0]) >= 5: label.append(1)
        else: label.append(-1)
        data.append([int(num) / 255 for num in curline[1:]])
    return data, label

def perceptron(data, label, iter = 50):
    print('start to train the model!')
    data_mat = np.mat(data)
    label_mat = np.mat(label).T
    m, n = np.shape(data_mat)
    w = np.zeros((1, np.shape(data_mat)[1]))
    b = 0
    h = 0.001
    for k in range(iter):
        for i in range(m):
            xi = data_mat[i]
            yi = label_mat[i]
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + h * yi * xi
                b = b + h * yi
        print('Round %d in %d Training' % (k, iter))
    return w, b

def model_test(data, label, w, b):
    data_mat = np.mat(data)
    label_mat = np.mat(label).T
    m, n = np.shape(data_mat)
    cnt = 0
    for i in range(m):
        xi = data_mat[i]
        yi = label_mat[i]
        if -1 * yi * (w * xi.T + b) >= 0: cnt = cnt + 1
    accRate = 1 - (cnt / m)
    return accRate

if __name__ == '__main__':
    start_time = time.perf_counter()
    train_data, train_label = load_data(r'D:\机器学习\李航 统计学习方法\各章代码实现\Mnist\mnist_train\mnist_train.csv')
    test_data, test_label = load_data(r'D:\机器学习\李航 统计学习方法\各章代码实现\Mnist\mnist_test\mnist_test.csv')
    w, b = perceptron(train_data, train_label, iter = 10)
    acc = model_test(test_data, test_label, w, b)
    end_time = time.perf_counter()
    print('Accuracy:', acc)
    print('Cost:', end_time - start_time)
