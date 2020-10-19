# coding=utf-8
import numpy as np
import time


# 加载数据
def loadData(filename):
    print('开始读取数据!')
    data = []; label = []
    fr = open(filename, 'r')
    for line in fr.readlines():
        curline = line.strip().split(',')
        data.append([int(int(num) > 128) for num in curline[1:]])
        label.append(int(curline[0]))
    return data, label


# 朴素贝叶斯算法
def NaiveBayes(Py, Px_y, x):
    featureNum = 784
    classNum = 10
    P = [0] * classNum
    for i in range(classNum):
        sum = 0
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]
        P[i] = sum + Py[i]
    return P.index(max(P))


def model_test(Py, Px_y, testData, testLabel):
    print('开始测试!')
    errorCnt = 0
    for i in range(len(testData)):
        predict = NaiveBayes(Py, Px_y, testData[i])
        if predict != testLabel[i]: errorCnt += 1
    return 1 - (errorCnt / len(testData))


def getAllProbability(trainData, trainLabel):
    print('计算先验概率和条件概率!')
    featureNum = 784
    classNum = 10

    Py = np.zeros((classNum, 1))
    for i in range(classNum):
        Py[i] = (np.sum(np.mat(trainLabel) == i) + 1) / (len(trainData) + 10)
    Py = np.log(Py)

    Px_y = np.zeros((classNum, featureNum, 2))
    for i in range(len(trainLabel)):
        label = trainLabel[i]
        x = trainData[i]
        for j in range(featureNum):
            Px_y[label][j][x[j]] += 1

    for label in range(classNum):
        for j in range(featureNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    return Py, Px_y


if __name__ == '__main__':
    start = time.time()
    trainData, trainLabel = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_train\mnist_train.csv')
    testData, testLabel = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_test\mnist_test.csv')
    Py, Px_y = getAllProbability(trainData, trainLabel)
    acc = model_test(Py, Px_y, testData, testLabel)
    end = time.time()
    print('准确率为:', acc)
    print('耗时:', end - start)


