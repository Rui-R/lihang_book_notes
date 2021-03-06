# ch8 提升方法

提升（boosting）方法是一种常用的统计学习方法，在分类问题中，它通过**改变训练样本的权重**，学习多个分类器，并将这些分类器线性组合，提高分类的性能。

## 8.1 提升方法 AdaBoost 算法

### 提升方法的基本思路

概率近似正确(PAC, Probably approximately correct)

在PAC学习框架下，一个概念是强可学习的**充分必要条件**是这个概念是弱可学习的。

强可学习：存在一个多项式学习算法，并且正确率很高。

弱可学习：存在一个多项式学习算法，并且正确率比随机猜测略好。

两个问题

1. 在每一轮如何改变训练数据的权值或者概率分布
2. 如何将弱分类器组合成一个强分类器

Adaboost解决方案：

1. 提高前一轮被分错的分类样本的权值，降低被正确分类的样本的权值
2. 加权多数表决的方法

#### 算法8.1

- 输入：训练数据集$T=\{(x_1,y_1), (x_2,y_2),...,(x_N,y_N)\}, x\in  \cal X\sube \R^n, y_i\in \cal Y = \{{-1, +1\}}$, 弱学习方法。
- 输出：最终分类器$G(x)$

步骤

1. 初始化训练数据的权值分布 $D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N},w_{1i}=\frac{1}{N})$
2. m = 1,2, M
   1. $G_m(x):X->{-1,+1}$
   2. 求$G_m$在训练集上的分类误差率  $e_m=\sum_{i=1}^{N}P(G_m(x_i)\ne y_i)=\sum_{i=1}^{N}w_{mi}I(G_m(x_i)\ne y_i)$
   3. 计算$G_m(x)$的系数，$\alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m}$，自然对数
   4. $w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i))$
   5. $Z_m=\sum_{i=1}^Nw_{mi}exp(-\alpha_my_iG_m(x_i))$
3. $f(x)=\sum_{m=1}^M\alpha_mG_m(x)$
4. 最终分类器$G(x)=sign(f(x))=sign(\sum_{m=1}^M\alpha_mG_m(x))$

从算法8.1的输入可以看出来，AdaBoost是个集成学习算法， 因为在它的输入中包含了**弱学习算法**。

注意这里面有个描述

> 使用具有权值分布$D_m$的训练数据集

这个怎么理解，是改变了数据么？

- 数据并没有被改变。
- 弱分类器的分类准则是错误率$e_m=\color{red}\sum_{i=1}^{N}\color{black}P(G_m(x_i)\ne y_i)=\sum_{i=1}^{N}w_{mi}I(G_m(x_i)\ne y_i)$
- 每次学习用到的数据集没有变，划分方式也没有变（比如阈值分类器中的分类点的选取方式），变的是评价每个划分错误的结果。
- 不同的权值分布上，不同样本错分对评价结果的贡献不同，**分类器**中分类错误的会被放大，分类正确的系数会减小，错误和正确的系数比值为$e^{2\alpha_m}=\frac{1-e_m}{e_m}$，这个比值是分类器分类正确的**几率**($odds$)。
- 书中对这点也有解释：误分类样本在下一轮学习中起更大的作用。==不改变所给的训练数据，而不断改变训练数据权值的分布==，使得训练数据在基本分类器的学习中起不同的作用， 这是AdaBoost的一个特点。



## 8.2 AdaBoost 算法训练误差分析

AdaBoost算法最终分类器的训练误差界为
$$
\frac{1}{N}\sum\limits_{i=1}\limits^N I(G(x_i)\neq y_i)\le\frac{1}{N}\sum\limits_i\exp(-y_i f(x_i))=\prod\limits_m Z_m
$$
这个的意思就是说**指数损失是0-1损失的上界**，然后通过递推得到了归一化系数的连乘。



## 8.3 AdaBoost 算法的解释

可以认为AdaBoost算法是**模型为加法模型，损失函数为指数函数，学习算法为前向分步算法**时的二类分类学习方法。

### 前向分步算法

#### 算法8.2

输入：训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N, y_N)}, x_i \in \cal X \sube R^n, y_i\in \{-1, 1\}$， 损失函数$L(y, f(x))$; 基函数集合$\{b(x;\gamma)\}$

输出：加法模型$f(x)$

步骤：

1. 初始化$f_0(x)=0$

2. 对$m=1,2,\dots,M$

3. 极小化损失函数
   $$
   (\beta_m,\gamma_m)=\arg\min \limits_ {\beta,\gamma}\sum_{i=1}^NL(y_i, f_{m-1}(x_i)+\beta b(x_i;\gamma))
   $$

4. 更新
   $$
   f_m(x)=f_{m-1}(x)+\beta _mb(x;\gamma_m)
   $$

5. 得到加法模型
   $$
   f(x)=f_M(x)=\sum_{m=1}^M\beta_m b(x;\gamma_m)
   $$



## 8.4 提升树

提升方法实际采用加法模型（即基函数的线性组合）与前向分步算法。

### 提升树模型

以决策树为基函数的提升方法称为提升树。

决策树$T(x;\Theta_m)$

提升树模型可以表示成决策树的加法模型
$$
f_M(x)=\sum_{m=1}^MT(x;\Theta_m)
$$

### 提升树算法

不同的问题， 主要区别在于损失函数不同：

1. 平方误差损失函数用于回归问题
2. 指数损失函数用于分类问题
3. 一般损失函数用于一般问题

#### 算法8.3

回归问题的提升树算法

输入：训练数据集

输出：提升树$f_M(x)$

步骤：

1. 初始化$f_0(x)=0$

2. 对$m=1,2,\dots,M$

   1. 计算残差

   $$
   r_{mi}=y_i-f_{m-1}(x_i), i=1,2,\dots,N
   $$

   1. **拟合残差**$r_{mi}$学习一个回归树，得到$T(x;\Theta_m)$
   2. 更新$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$

3. 得到回归问题提升树
   $$
   f(x)=f_M(x)=\sum_{m=1}^MT(x;\Theta_m)
   $$

### 梯度提升(GBDT)

#### 算法8.4

输入： 训练数据集$T={(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)}, x_i \in \cal x \sube \R^n, y_i \in \cal y \sube \R$；损失函数$L(y,f(x))$
输出：回归树$\hat{f}(x)$
步骤：

1. 初始化
   $$
   f_0(x)=\arg\min\limits_c\sum_{i=1}^NL(y_i, c)
   $$

2. $m=1,2,\dots,M$

3. $i=1,2,\dots,N$
   $$
   r_{mi}=-\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x)=f_{m-1}(x)}
   $$

4. 对$r_{mi}$拟合一个回归树，得到第$m$棵树的叶节点区域$R_{mj}, j=1,2,\dots,J$

5. $j=1,2,\dots,J$
   $$
   c_{mj}=\arg\min_c\sum_{xi\in R_{mj}}L(y_i,f_{m-1}(x_i)+c)
   $$

6. 更新
   $$
   f_m(x)=f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
   $$

7. 得到回归树
   $$
   \hat{f}(x)=f_M(x)=\sum_{m=1}^M\sum_{j=1}^Jc_{mj}I(x\in R_{mj})
   $$

这个算法里面注意，关键是`用损失函数的负梯度，在当前模型的值作为回归问题提升树算法中的残差近似值，拟合回归树`

