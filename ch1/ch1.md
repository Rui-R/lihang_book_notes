# ch1 统计学习及监督学习概论



## 1.1 统计学习

统计学习又称为统计机器学习（statistical machine learning），研究对象是**数据**，基本假设是同类数据具有一定的**统计规律性**。

实现统计学习方法的步骤如下：

1. 得到一个有限的训练数据集合；
2. 确定包含所有可能的模型的**假设空间**，即学习模型的集合；
3. 确定模型选择的准则，即学习的**策略**；
4. 实现求解最优模型的算法，即学习的**算法；**
5. 通过学习方法选择最优的模型；
6. 利用学习的最优模型对新数据进行预测或分析。



## 1.2 统计学习的分类

### 基本分类

监督学习、无监督学习、强化学习、半监督学习、主动学习等。

### 按模型分类

概率模型、非概率模型；线性模型与非线性模型；参数化模型与非参数化模型。

### 按算法分类

在线学习与批量学习(batch learning)。

### 按技巧分类

贝叶斯学习与核方法。



## 1.3 统计学习方法三要素

==三要素：**模型+策略+算法**。本书的核心。==

### 模型

定义：监督学习中，模型就是所要学习的**条件概率分布**或**决策函数**。

|              | 假设空间$\cal F$                                             | 输入空间$\cal X$ | 输出空间$\cal Y$ | 参数空间      |
| ------------ | ------------------------------------------------------------ | ---------------- | ---------------- | ------------- |
| 决策函数     | $\cal F\it =\{f |Y=f_{\theta}(x), \theta \in \bf R \it ^n\}$ | 变量             | 变量             | $\bf R\it ^n$ |
| 条件概率分布 | $\cal F\it =\{P|P_{\theta}(Y|X),\theta\in \bf R \it ^n\}$    | 随机变量         | 随机变量         | $\bf R\it ^n$ |

$\bf R\it ^n$:n维欧式空间，也称为参数空间。

### 策略

==统计学习的目标：从假设空间中选取最优模型。==

#### 损失函数与风险函数

**损失函数**度量模型**一次预测**的好坏，**风险函数**度量**平均意义**下模型预测的好坏。

- **损失函数**(loss function)或**代价函数**(cost function)
  定义：给定输入$X$时，输出的**预测值$f(X)$**和**真实值$Y$**之间的**非负实值**函数，记作$L(Y,f(X))$。

  损失函数值越小，模型就越好。

- **风险函数**(risk function)或**期望损失**(expected loss)
  定义：模型$f(X)$关于联合分布$P(X,Y)$的**平均意义下的**损失(**期望**损失)。由于$P(X,Y)$是未知的，所以

  $R_{exp}(f)$不能直接计算，监督学习成为了一个病态问题。

  $R_{exp}(f)=E_p[L(Y, f(X))]=\int_{\mathcal X\times\mathcal Y}L(y,f(x))P(x,y)\, {\rm d}x{\rm d}y$

- **经验风险**(empirical risk)或**经验损失**(empirical loss)
  定义：模型$f({X})$关于**训练数据集**的平均损失。

  $R_{emp}(f)=\frac{1}{N}\sum^{N}_{i=1}L(y_i,f(x_i))$

  根据大数定律，当样本容量N趋于无穷时，经验风险趋于期望风险。即可用经验风险估计期望风险。

- **结构风险**(structural risk)
  $R_{srm}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)$
  $J(f)$为模型复杂度,模型越复杂，其值越大；$\lambda \geqslant 0$是系数，用以权衡经验风险和模型复杂度。

#### 常用损失函数

损失函数数值越小，模型就越好

1. 0-1损失函数
   $L(Y,f(X))=\begin{cases}1, Y \neq f(X) \\0, Y=f(X) \end{cases}$

2. 平方损失函数
   $L(Y,f(X))=(Y-f(X))^2$

3. 绝对损失函数
   $L(Y,f(X))=|Y-f(X)|$

4. 对数损失函数
   $L(Y,P(Y|X))=-\log P(Y|X)$

   $P(Y|X)\leqslant 1$，对应的对数是负值，所以对数损失中包含一个负号。

#### ERM与SRM

经验风险最小化(ERM)与结构风险最小化(SRM)

经验风险最小化：认为经验风险最小的模型就是最优模型。例：极大似然估计

结构风险最小化：认为结构风险最小的模型就是最优模型。等价于正则化，平衡了经验风险和模型复杂度。例：贝叶斯估计中的最大后验概率估计。

### 算法

算法：求解最优模型的方法

## 1.4 模型评估与模型选择

#### 训练误差和测试误差

训练误差和测试误差是模型关于数据集的平均损失，测试误差小的方法具有更好的预测（泛化）能力。

#### 过拟合与模型选择

过拟合：学习时选择的模型所包含的**参数过多**，以至出现这一模型对已知数据预测得很好，但对未知数据预测得很差的现象。

![误差](D:\机器学习\李航 统计学习方法\误差.jpg)

## 1.5 正则化与交叉验证

#### 正则化

模型选择的典型方法是正则化，即加上正则化项或罚项。

符合奥卡姆剃刀原理：在所有可能选择的模型中，选择能够很好地解释已知数据并且十分简单的模型。

#### 交叉验证

另一种常用的模型选择方法是交叉验证。

- 简单
- S折
- 留一法：S=N

## 1.6 泛化能力

- 现实中采用最多的方法是通过**测试误差**来评价学习方法的泛化能力
- 统计学习理论试图从理论上对学习方法的泛化能力进行分析
- 学习方法的泛化能力往往是通过研究泛化误差的**概率上界**进行的, 简称为泛化误差上界(generalization error bound)

## 1.7生成模型与判别模型

**监督学习方法**可分为**生成方法**(generative approach)与**判别方法**(discriminative approach)

### 生成方法

- 可以还原出**联合概率分布**$P(X,Y)$
- 收敛速度快, 当样本容量增加时, 学到的模型可以更快收敛到真实模型
- 当存在隐变量时仍可使用

### 判别方法

- 直接学习**条件概率**$P(Y|X)$或者**决策函数**$f(X)$
- 直接面对预测, 往往学习准确率更高
- 可以对数据进行各种程度的抽象,  定义特征并使用特征, 可以简化学习问题

## 1.8监督学习应用

应用：分类问题、标注问题、回归问题

## END 本章概要

1. 统计学习（机器学习）：计算机基于数据构建概率统计模型并运用模型对数据进行分析与预测的一门学科。包括监督学习、无监督学习、半监督学习和强化学习。
2. 统计学习三要素：**模型+策略+算法**。
3. 监督学习：从给定有限的训练数据出发，假设数据是**独立同分布**的，假设模型属于某个假设空间，应用某一评价准则，从假设空间中选取一个最优模型，使它对已给训练数据及**未知测试数据**在给定评价标准意义下有最准确的预测。
4. 为防止**过拟合**->提高模型**泛化能力**；模型选择方法：**正则化**，**交叉验证**。
5. 监督学习研究的问题：**分类**问题、**标注**问题、**回归**问题。