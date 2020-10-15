# ch2 感知机

==感知机（perceptron）是**二类分类**的**线性分类**模型，是神经网络和支持向量机的基础。==

## 2.1 感知机模型

输入空间：$\mathcal X\sube \bf R^n$

输出空间：$\mathcal Y=\{{+1,-1}\}$

决策函数：$f(x)=sign (w\cdot x+b)$

- $w$：权值或权值向量 b：偏置 
- 假设空间：定义在特征空间中的所有线性分类模型，即函数集合$\{{f|f(x) = w \cdot x + b}\}$
- 对应于输入空间（特征空间）的分离超平面$w\cdot x+b = 0$



## 2.2 感知机学习策略



### 数据的线性可分性

给定一个数据集 $T = \{{(x_1,y_1),(x_2,y_2),···,(x_N,y_N)}\}$，其中, $x\sube \bf\mathcal X = R^n,y_i\sube \bf \mathcal Y = \{+1,-1\}$,$\mathcal i = 1,2,···,N$,如果存在某个超平面S：$w\cdot x+b$，能够将数据集的正实例点和负实例点完全正确划分到超平面两侧，则称该数据集T线性可分。

充要条件：正实例点所构成的凸壳和负实例点所构成的凸壳互不相交。



### 学习策略

假设训练数据集是**线性可分**的，感知机的目标是求得一个能够将*训练集正实例点和负实例点*完全正确分开的分离超平面，为了找到这样的超平面，需要确定$w和b$。

确定学习策略即**定义（经验）损失函数**并将其最小化。

> 损失函数的一个自然选择是误分类点的总数，但是，这样的损失函数**不是参数$w,b$的连续可导函数，不易优化**
>
> 损失函数的另一个选择是**误分类点到超平面$S$的总距离**，这是感知机所采用的。

感知机学习的经验风险函数(损失函数)
$$
L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)
$$

- 其中$M$是误分类点的集合

- 给定训练数据集$T$，损失函数$L(w,b)$是$w$和$b$的连续可导函数

## 2.3 感知机学习算法



### 原始形式

> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal X=\bf R^n\mit , y_i\in \cal Y\it =\{-1,+1\}, i=1,2,\dots,N; \ \ 0<\eta\leqslant 1$
>
> 输出：$w,b;f(x)=sign(w\cdot x+b)$
>
> 1. 选取初值$w_0,b_0$
> 2. 训练集中选取数据$(x_i,y_i)$
> 3. 如果$y_i(w\cdot x_i+b)\leqslant 0$
>
> $$
> w\leftarrow w+\eta y_ix_i \nonumber\\
> b\leftarrow b+\eta y_i
> $$
>
> 4. 转至(2)，直至训练集中没有误分类点

- $\eta $ 是步长，又称为学习率，取值(0,1]。
- 通过随机梯度下降法(stochastic gradient descent)进行迭代，使损失函数越来越小。

- 原始形式中的迭代公式，可以对$x$补1，将$w$和$b$合并在一起，称为扩充权重向量。
- 原始形式的迭代公式对$w$进行更新，更新的是一个向量，因此计算量比较大。
- 由于采用不同的初值或选取不同的误分类点，解可以不同。



### 对偶形式

> 输入：$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}\\ x_i\in \cal{X}=\bf{R}^n , y_i\in \cal{Y} =\{-1,+1\}, i=1,2,\dots, N; 0< \eta \leqslant 1$
>
> 输出：
> $$
> \alpha ,b; f(x)=sign\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right)\nonumber\\
> \alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T
> $$
>
> 1. $\alpha \leftarrow 0,b\leftarrow 0$
> 2. 训练集中选取数据$(x_i,y_i)$
> 3. 如果$y_i\left(\sum_{j=1}^N\alpha_jy_jx_j\cdot x+b\right) \leqslant 0$
>
> $$
> \alpha_i\leftarrow \alpha_i+\eta \nonumber\\
> b\leftarrow b+\eta y_i
> $$
>
> 4. 转至(2)，直至训练集中没有误分类点



**Gram matrix**

对偶形式中，训练实例仅以内积的形式出现。

为了方便可预先将训练集中的实例间的内积计算出来并以矩阵的形式存储，这个矩阵就是所谓的Gram矩阵。
$$
G=[x_i\cdot x_j]_{N\times N} \nonumber
$$

- 实例点更新次数越多，意味着它离超平面越近，也就越难正确分类，这样的实例点对学习结果影响最大。
- 对偶形式的迭代公式对$\alpha$进行更新，更新的是一个数，因此计算量相比原始形式要小。
- Gram矩阵可以用于查询实例间的内积，减少运算量。