# coding=utf-8
import numpy as np
import time


def loadData(filename):
    dataList = []; labelList = []
    f = open(filename, 'r')
    for line in f.readlines():
        curLine = line.strip().split(',')
        if int(curLine[0]) > 0:
            labelList.append(1)
        else:
            labelList.append(0)
        dataList.append([int(num) / 255 for num in curLine[1:]])
    return dataList, labelList


def predict(w, x):
    wx = np.dot(w, x)
    p1 = np.exp(wx) / (1 + np.exp(wx))
    if p1 >= 0.5: return 1
    else: return 0


def logisticRegression(trainDataList, trainLabelList, iter = 200):
    for i in range(len(trainDataList)):
        trainDataList[i].append(1)
    trainDataList = np.array(trainDataList)
    w = np.zeros(trainDataList.shape[1])
    # 步长
    h = 0.001
    for i in range(iter):
        for j in range(trainDataList.shape[0]):
            wx = np.dot(w, trainDataList[j])
            xi = trainDataList[j]
            yi = trainLabelList[j]
            w += h * (yi * xi - (np.exp(wx) * xi) / (1 + np.exp(wx)))
    return w


def modelTest(testDataList, testLabelList, w):
    for i in range(len(testDataList)):
        testDataList[i].append(1)
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()
    print('start to load train data!')
    trainDataList, trainLabelList = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_train\mnist_train.csv')
    print('start to load test data!')
    testDataList, testLabelList = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_test\mnist_test.csv')
    print('start to train!')
    w = logisticRegression(trainDataList,trainLabelList)
    print('start to test!')
    acc = modelTest(testDataList, testLabelList, w)
    end = time.time()
    print('Time cost:', end - start)
    print('Accuracy:', acc)