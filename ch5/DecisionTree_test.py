import numpy as np
import time


# 加载数据
def loadData(filename):
    data = []; label = []
    f = open(filename, 'r')
    for line in f.readlines():
        curLine = line.strip().split(',')
        data.append([int(int(num) > 128) for num in curLine[1:]])
        label.append(int(curLine[0]))
    return data, label


# 找到当前标签集中占比最大的标签
def majorClass(label):
    classDict = {}
    for i in range(len(label)):
        if label[i] in classDict.keys():
            classDict[label[i]] += 1
        else:
            classDict[label[i]] = 1
    classSorted = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    return classSorted[0][0]


# 计算经验熵
def calc_H_D(trainLabel):
    H_D = 0
    trainLabelSet = set([label for label in trainLabel])
    for i in trainLabelSet:
        p = trainLabel[trainLabel == i].size / trainLabel.size
        H_D += -1 * p * np.log2(p)
    return H_D


# 计算经验条件熵
def calc_H_D_A(trainData_DevFeature, trainLabel):
    H_D_A = 0
    trainDataSet = set([label for label in trainData_DevFeature])
    for i in trainDataSet:
        H_D_A += trainData_DevFeature[trainData_DevFeature == i].size / trainData_DevFeature.size \
                * calc_H_D(trainLabel[trainData_DevFeature == i])
    return H_D_A


# 选出信息增益最大的特征
def calcBestFeature(trainDataList, trainLabelList):
    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)
    featureNum = trainDataArr.shape[1]
    max_G_D_A = -1
    maxFeature = -1
    H_D = calc_H_D(trainLabelArr)
    for feature in range(featureNum):
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
        G_D_A = H_D - calc_H_D_A(trainDataArr_DevideByFeature, trainLabelArr)
        if G_D_A > max_G_D_A:
            max_G_D_A = G_D_A
            maxFeature = feature
    return maxFeature, max_G_D_A


# 更新数据集和标签集
def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    retDataArr = []
    retLabelArr = []
    for i in range(len(trainDataArr)):
        if trainDataArr[i][A] == a:
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])
            retLabelArr.append(trainLabelArr[i])
    return retDataArr, retLabelArr

# 递归创建决策树
def createTree(*dataset):
    Epsilon = 0.1
    trainDataList = dataset[0][0]
    trainLabelList = dataset[0][1]
    print('start a node', len(trainDataList[0]), len(trainLabelList))
    classDict = {i for i in trainLabelList}
    if len(classDict) == 1:
        return trainLabelList[0]
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    treeDict = {Ag:{}}
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    return treeDict


# 预测新样本
def predict(testDataList, tree):
    while True:
        (key, value), = tree.items()
        if type(tree[key]).__name__ == 'dict':
            dataVal = testDataList[key]
            del testDataList[key]
            tree = value[dataVal]
            if type(tree).__name__ == 'int':
                return tree
        else:
            return value


# 测试准确率
def modelTest(testDataList, testLabelList, tree):
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    return 1 - errorCnt / len(testDataList)


if __name__ == '__main__':
    start = time.time()
    print('start to load trainData!')
    trainDataList, trainLabelList = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_train\mnist_train.csv')

    print('start to load testData!')
    testDataList, testLabelList = loadData(r'D:\Machine_Learning\LihangBook\Code\Mnist\mnist_test\mnist_test.csv')

    print('start to create a tree!')
    tree = createTree((trainDataList, trainLabelList))
    print('tree is:', tree)

    print('start to test!')
    acc = modelTest(testDataList, testLabelList, tree)
    print('Accuracy is:', acc)
    end = time.time()
    print('Time cost:', time)
