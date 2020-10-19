# ch4 朴素贝叶斯法

==朴素贝叶斯(Naive Bayes)法是基于**贝叶斯定理**与**特征条件独立假设**的分类方法，属于**生成模型。**==



## 4.1 朴素贝叶斯法的学习与分类

### 4.1.1 基本方法

朴素贝叶斯法通过训练数据集学习联合概率分布$P(X,Y)$，即学习以下**先验概率分布**及**条件概率分布**。

先验概率分布：$P(Y = c_k), k = 1,2,···,K$

条件概率分布：$P(X = x|Y=c_k) = P(X^{(1)}=x^{(1)},···,X^{(n)}=x^{(n)}|Y=c_k),k = 1,2,···,K$

- 条件概率分布有指数级别的参数$K\prod_{j=1}^nS_j$，因此需要很多样本来刷参数，其估计实际上是**不可行**的。
- 因此，条件概率分布作了**条件独立性**假设，这一强假设大大简化了问题，也是“朴素”二字的由来。

求$P(Y|X)$，其中$X\in\{X_1,X_2,\dots,X_n\}$，条件独立假设在给定$Y$（类别）的情况下：

1. 每一个$X_i$和其他的每个$X_k$是条件独立的
2. 每一个$X_i$和其他的每个$X_k$的子集是条件独立的

**条件独立性假设**是:
$$
\begin{align}
P(X=x|Y=c_k)&=P(X^{(1)} = x^{(1)},\dots,X^{(n)} = x^{(n)}|Y=c_k)\\
&=\prod^n_{j=1}P(X^{(j)}=x^{(j)}|Y=c_k)
\end{align}
$$
朴素贝叶斯法进行分类时，对于给定输入x，通过学习到的模型计算后验概率分布$P(Y = c_k|X = x)$，将**后验概率最大的类**作为x的类输出：
$$
P(Y = c_k|X = x) = \frac{P(X = x,Y = c_k)}{P(X = x)} = \frac{P(X = x|Y = c_k)P(Y = c_k)}{\sum P(X = x|Y = c_k)P(Y = c_k)}
$$
代入**条件独立性假设**公式可得：
$$
P(Y = c_k|X = x) = \frac{P(Y = c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_k P(Y = c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}
$$
则**朴素贝叶斯分类器**可表示为：
$$
y = f(x) = arg\max_{c_k}\frac{P(Y = c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_k P(Y = c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}
$$
由于**分母对所有的$c_k$都是相同**的（相当于全概率公式），所以可以约去：
$$
y = f(x) = arg\max_{c_k}{P(Y = c_k)\prod_{j}P(X^{(j)}=x^{(j)}|Y=c_k)}
$$


### 4.1.2 后验概率最大化

==后验概率最大化对应了期望风险最小化准则。==

书P61有推导过程，建议手推一下公式。

## 4.2 参数估计

### 4.2.1 极大似然估计

条件概率的极大似然估计：
$$
P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x_i^{j}=a_{jl},y_j=c_k)}{\sum\limits_{i=1}^NI(y_j=c_k)}
$$
先验概率的极大似然估计：
$$
P(Y = c_k) = \frac{\sum _{i=1}^NI(y_i = c_k)}{N}
$$


### 4.2.2 算法

![NaiveBayes算法_1](D:\Machine_Learning\LihangBook\NaiveBayes算法_1.jpg)

![NaiveBayes算法_2](D:\Machine_Learning\LihangBook\NaiveBayes算法_2.jpg)

### 4.2.3 贝叶斯估计

- 对于$x$的某个特征的取值没有在先验中出现的情况 ，如果用极大似然估计，这种情况的可能性就是0。

- 出现这种情况的原因通常是因为数据集不能全覆盖样本空间。
- 为了保证每一项均不为0并且概率之和仍为1，贝叶斯估计进行**平滑处理**。

条件概率的贝叶斯估计：
$$
P_{\lambda}(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum\limits_{i=1}^NI(x_i^{j}=a_{jl},y_j=c_k)+\lambda}{\sum\limits_{i=1}^NI(y_j=c_k)+S_j\lambda}
$$
先验概率的贝叶斯估计：
$$
P_{\lambda}(Y = c_k) = \frac{\sum _{i=1}^NI(y_i = c_k) + \lambda}{N + K\lambda}
$$
其中$\lambda \geqslant 0$

当$\lambda = 0$的时候，就是极大似然估计。

当$\lambda=1$的时候，这个平滑方案叫做Laplace Smoothing。**拉普拉斯平滑**相当于给未知变量给定了先验概率。