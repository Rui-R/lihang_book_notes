# ch7 支持向量机

**支持向量机**（support vector machine **SVM**）是一种二类分类模型，由Vapnik提出。根据学习策略的不同，可分为**线性可分支持向量机**，**线性支持向量机**和**非线性支持向量机**。

## 线性可分支持向量机

### 函数间隔与几何间隔

#### 函数间隔

对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为
$$
\hat \gamma_i=y_i(w\cdot x_i+b)
$$
定义超平面$(w,b)$关于训练数据集$T$的函数间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的函数间隔之最小值，即
$$
\hat \gamma=\min_{i=1,\cdots,N}\hat\gamma_i
$$
函数间隔可以表示分类预测的**正确性**及**确信度**。

#### 几何间隔

如果成比例地改变$w$和$b$，超平面没有改变，然而函数间隔却变成原来的2倍。因此我们引入了$||w||$进行规范化。

对于给定数据集$T$和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔为


$$
\gamma_i=y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})
$$
定义超平面$(w,b)$关于训练数据集$T$的几何间隔为超平面$(w,b)$关于$T$中所有样本点$(x_i,y_i)$的几何间隔之最小值，即
$$
\gamma=\min_{i=1,\cdots,N}\gamma_i
$$
函数间隔可以表示分类预测的**正确性**及**确信度**。

#### 硬间隔最大化

考虑如何求得一个几何间隔最大的分离超平面（下式经过变换）：
$$
\begin{align}
&\max_{w,b}\frac{\hat \gamma}{\left\|w\right\|}\\
&s.t.\ \ \ y_i(w\cdot x_i+b)\geqslant\hat \gamma,i=1,2,\dots,N\\
\end{align}
$$

#### 原始问题

等价于求解以下问题：
$$
\begin{align}
&\min_{w,b} \frac{1}{2}{\left\|w\right\|}^2\\
&s.t.\ \ \ y_i(w\cdot x_i+b) - 1\geqslant0,i=1,2,\dots,N\\
\end{align}
$$
这是一个凸二次规划问题，肯定存在全局最优解。

### 支持向量

- 在线性可分情况下，训练集样本点中与分离超平面距离最近的样本点的实例称为**支持向量**。

- 在决定分离超平面时只有支持向量起作用，其他实例点并不起作用，因此将该模型称为支持向量机。

### 算法

如果求出了上述方程的解$w^*, b^*$，就可得到

分离超平面
$$
w^*\cdot x+b^*=0
$$


相应的分类决策函数
$$
f(x)=sign(w^*\cdot x+b^*)
$$

### 对偶算法

- 对偶问题往往更容易求解（拉格朗日乘子法）
- 后续可以引入核函数，进而推广到非线性分类问题

针对每个不等式约束，定义拉格朗日乘子$\alpha_i\ge0$，定义拉格朗日函数
$$
\begin{align}
L(w,b,\alpha)&=\frac{1}{2}w\cdot w-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\left[\sum_{i=1}^N\alpha_i[y_i(w\cdot x_i+b)-1]\right]\\
&=\frac{1}{2}\left\|w\right\|^2-\sum_{i=1}^N\alpha_iy_i(w\cdot x_i+b)+\sum_{i=1}^N\alpha_i
\end{align}\\
\alpha_i \geqslant0, i=1,2,\dots,N
$$
其中$\alpha=(\alpha_1,\alpha_2,\dots,\alpha_N)^T$为拉格朗日乘子向量

**原始问题是极小极大问题**

根据**拉格朗日对偶性**，原始问题的**对偶问题是极大极小问题**:
$$
\max\limits_\alpha\min\limits_{w,b}L(w,b,\alpha)
$$
通过分别对$w$和$b$求偏导数，化简得到最终结果：
$$
\min\limits_\alpha \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
s.t. \ \ \ \sum_{i=1}^N\alpha_iy_i=0\\
\alpha_i\geqslant0, i=1,2,\dots,N
$$
求得$\alpha$的最优解$\alpha^*$后，可利用以下公式求$(w,b)$的解：
$$
\begin{align}
w^*&=\sum_{i=1}^{N}\alpha_i^*y_ix_i\\
b^*&=\color{red}y_j\color{black}-\sum_{i=1}^{N}\alpha_i^*y_i(x_i\cdot \color{red}x_j\color{black})
\end{align}
$$


## 线性支持向量机

当数据集近似线性可分时，将硬间隔最大化修改为软间隔最大化，即可扩展到线性不可分问题。

### 原始问题

引入松弛变量$\xi≥0$,得到如下原始问题：
$$
\begin{align}
\min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
&\xi_i\geqslant0,i=1,2,\dots,N
\end{align}
$$
同样地，这也是一个凸二次规划问题，肯定存在全局最优解。

### 对偶问题

$$
\begin{align}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{align}
$$

- 通过求解对偶问题， 得到$\alpha$，然后求解$w,b$的过程和线性可分支持向量机是一样的。

- 线性支持向量机的解$w^*$唯一，但$b^*$不一定唯一。



### 合页损失

另一种解释，最小化目标函数

$$\min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2$$

其中

- 第一项是经验损失或经验风险，函数$L(y(w\cdot x+b))=[1-y(w\cdot x+b)]_+$称为合页损失，可以表示成$L = \max(1-y(w\cdot x+b), 0)$
- 第二项是**系数为$\lambda$的$w$的$L_2$范数的平方**，是正则化项

书中这里通过定理7.4说明了用合页损失表达的最优化问题和线性支持向量机原始最优化问题的关系。
$$
\begin{align}
\min_{w,b,\xi} &\frac{1}{2}\left\|w\right\|^2+C\sum_{i=1}^N\xi_i\\
s.t. \ \ \ &y_i(w\cdot x_i+b)\geqslant1-\xi_i, i=1,2,\dots,N\\
&\xi_i\geqslant0,i=1,2,\dots,N
\end{align}
$$
等价于
$$
\min\limits_{w,b} \sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2
$$
证明：

令合页损失$\left[1-y_i(w\cdot x+b)\right]_+=\xi_i$，合页损失非负，所以有$\xi_i\ge0$，这个对应了原始最优化问题中的**一个约束【1】**。

还是根据合页损失非负，当$1-y_i(w\cdot x+b)\leq\color{red}0$的时候，有$\left[1-y_i(w\cdot x+b)\right]_+=\color{red}\xi_i=0$，所以有

$1-y_i(w\cdot x+b)\leq\color{red}0=\xi_i$，这对应了原始最优化问题中的**另一个约束【2】**。

所以，在满足这**两个约束【1】【2】**的情况下，有
$$
\begin{aligned}
\min\limits_{w,b} &\sum\limits_{i=1}^N\left[1-y_i(w\cdot x+b)\right]_++\lambda\left\|w\right\|^2\\
\min\limits_{w,b} &\sum\limits_{i=1}^N\xi_i+\lambda\left\|w\right\|^2\\
\min\limits_{w,b} &\frac{1}{C}\left(\frac{1}{2}\left\|w\right\|^2+C\sum\limits_{i=1}^N\xi_i\right), with \  \lambda=\frac{1}{2C}\\
\end{aligned}
$$
看下下面这个图，其中合页损失和感知机损失函数之间的关系，合页损失要求**函数间隔大于1**的时候才没有损失（loss=0），而感知机只要**函数间隔大于0**就认为没有损失，所以说合页损失对学习有更高的要求。



## 非线性支持向量机

### 核函数

**定义**：设 $\mathcal X$ 是输入空间（欧式空间**$R^n$**的子集或离散集合），又设$\mathcal H$为特征空间（希尔伯特空间），如果存在一个从$\mathcal X$到$\mathcal H$的映射：$\phi(x): \mathcal X \rightarrow\mathcal H$，使得对所有$x,z \in \mathcal X$，函数$K(x,z)$满足条件：$K(x,z) = \phi(x) ·  \phi(z)$，则称$K(x,z)$为核函数，$\phi(x)$为映射函数，$\phi(x) ·  \phi(z)$称为两映射函数的内积。

- 核技巧的想法是，在学习和预测中只定义核函数$K(x,z)$，而不显式地定义映射函数$\phi$。
- 对于给定的核$K(x,z)$，特征空间$\mathcal H$和映射函数$\phi$的取法并不唯一，可以取不同的特征空间，即便是同一特征空间里也可以取不同的映射。

### 常用核函数

#### 多项式核

$K(x, z) = (x · z + 1) ^ p$

#### 高斯核

$K(x,z) = exp(-\frac{||x-z||^2}{2\sigma^2})$



### 问题描述

~~和线性支持向量机的问题描述一样~~，注意，这里是有差异的，将**向量内积**替换成了**核函数**，而后续SMO算法求解的问题是该问题。

构建最优化问题：
$$
\begin{align}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{align}
$$
求解得到$\alpha^*=(\alpha_1^*,\alpha_2^*,\cdots,\alpha_N^*)^T$

选择$\alpha^*$的一个正分量计算
$$
b^*=y_j-\sum_{i=1}^N\alpha_i^*y_iK(x_i,x_j)
$$
构造决策函数
$$
f(x)=sign\left(\sum_{i=1}^N\alpha_i^*y_iK(x,x_i)+b^*\right)
$$




## 序列最小最优化算法

### 问题描述

$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$

这个问题中，变量是$\alpha$，一个变量$\alpha_i$对应一个样本点$(x_i,y_i)$，变量总数等于$N$。



### KKT 条件

KKT条件是该最优化问题的充分必要条件。



### SMO算法

整个SMO算法包括两**部分**：

1. 求解两个变量二次规划的解析方法
2. 选择变量的启发式方法

$$
\begin{aligned}
\min_\alpha\ &\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\ \ \ &\sum_{i=1}^N\alpha_iy_i=0\\
&0\leqslant \alpha_i \leqslant C,i=1,2,\dots,N
\end{aligned}
$$

注意，这里是**两个部分**，而不是先后的两个步骤。

#### Part I

两变量二次规划求解

选择两个变量$\alpha_1,\alpha_2$

由等式约束可以得到

$\alpha_1=-y_1\sum\limits_{i=2}^N\alpha_iy_i$

所以这个问题实质上是单变量优化问题。
$$
\begin{align}
\min_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=&\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2\nonumber\\
&-(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^Ny_i\alpha_iK_{il}+y_2\alpha_2\sum_{i=3}^Ny_i\alpha_iK_{i2}\\
s.t. \ \ \ &\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^Ny_i\alpha_i=\varsigma\\
&0\leqslant\alpha_i\leqslant C, i=1,2
\end{align}
$$
上面存在两个约束：

1. **线性**等式约束
2. 边界约束

书中的配图，其中这样一段描述**等式约束使得$(\alpha_1,\alpha_2)$在平行于盒子$[0,C]\times [0,C]$的对角线的直线上**

这句怎么理解？

$y_i\in \mathcal Y=\{+1,-1\}$所以又等式约束导出的关系式中两个变量的系数相等，所以平行于对角线。

#### Part II

变量的选择方法

1. 第一个变量$\alpha_1$
   外层循环
   违反KKT条件**最严重**的样本点
2. 第二个变量$\alpha_2$
   内层循环
   希望能使$\alpha_2$有足够大的变化
3. 计算阈值$b$和差值$E_i$

### 算法7.5

> 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),\dots, (x_N,y_N)}$，其中$x_i\in\mathcal X=\bf R^n, y_i\in\mathcal Y=\{-1,+1\}, i=1,2,\dots,N$,精度$\epsilon$
>
> 输出：近似解$\hat\alpha$
>
> 1. 取初值$\alpha_0=0$，令$k=0$
> 2. **选取**优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$，解析求解两个变量的最优化问题，求得最优解$\alpha_1^{(k+1)},\alpha_2^{(k+1)}$，更新$\alpha$为$\alpha^{k+1}$
> 3. 若在精度$\epsilon$范围内满足停机条件
>
> $$
> \sum_{i=1}^{N}\alpha_iy_i=0\\
> 0\leqslant\alpha_i\leqslant C,i=1,2,\dots,N\\
> y_i\cdot g(x_i)=
> \begin{cases}
> \geqslant1,\{x_i|\alpha_i=0\}\\
> =1,\{x_i|0<\alpha_i<C\}\\
> \leqslant1,\{x_i|\alpha_i=C\}
> \end{cases}\\
> g(x_i)=\sum_{j=1}^{N}\alpha_jy_jK(x_j,x_i)+b
> $$
>
> 则转4,否则，$k=k+1$转2
>
> 1. 取$\hat\alpha=\alpha^{(k+1)}$