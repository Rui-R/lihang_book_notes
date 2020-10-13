# coding=utf-8

import numpy as np
import time


def loadData(filename):
    print('开始读取数据!')
    data = []; label = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        data.append([int(num) for num in curline[1:]])
        label.append(int(curline[0]))
    return data, label

def calcDist(x1, x2):
    # 欧氏距离
    return np.sqrt(np.sum(np.square(x1 - x2)))
    # 曼哈顿距离
    # return np.sum(x1 - x2)

def getCloest(trainDataMat, trainLabelMat, x, topK):
    distList = [0] * len(trainLabelMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist
    topKList = np.argsort(np.array(distList))[:topK]
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))

def model_test(trainData, trainLabel, testData, testLabel, topK = 25):
    print('开始测试！')
    trainDataMat = np.mat(trainData); trainLabelMat = np.mat(trainLabel).T
    testDataMat = np.mat(testData); testLabelMat = np.mat(testLabel).T
    errorCnt = 0
    # for i in range(len(testDataMat)):
    for i in range(200):
        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getCloest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]: errorCnt += 1
    # return 1 - (errorCnt / len(testDataMat))
    return 1 - (errorCnt / 200)

if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData(r'D:\机器学习\李航 统计学习方法\各章代码实现\Mnist\mnist_train\mnist_train.csv')
    testData, testLabel = loadData(r'D:\机器学习\李航 统计学习方法\各章代码实现\Mnist\mnist_test\mnist_test.csv')
    acc = model_test(trainData, trainLabel, testData, testLabel)
    end = time.time()
    print('正确率:%d' % (acc * 100), '%')
    print('时间开销:', end - start)
