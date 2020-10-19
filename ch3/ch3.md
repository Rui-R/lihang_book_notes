# ch3 K近邻法

K近邻法（k-nearest neighbor KNN）是一种分类和回归方法，可以进行多分类。K近邻法没有显式的学习过程。

==**K值的选择、距离度量和分类决策规则**==是k近邻法的三个基本要素。



## 3.1K近邻算法

输入: $T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}， x_i\in \cal{X}\sube{\bf{R}^n}, y_i\in\cal{Y}=\{c_1,c_2,\dots, c_k\}$; 实例特征向量$x$

输出: 实例所属的类$y$

步骤:

1. 根据指定的**距离度量**，在$T$中查找$x$的**最近邻的$k$个点**，覆盖这$k$个点的$x$的邻域定义为$N_k(x)$

2. 在$N_k(x)$中应用**分类决策规则**决定$x$的类别$y$
   $$
   y=\arg\max_{c_j}\sum_{x_i\in N_k(x)}I(y_i=c_j), i=1,2,\dots,N, j=1,2,\dots,K
   $$

- 公式的含义：对于每一个类$c_j$，进行$I(y_i = c_j)$求和，计算这k个点中有多少标记为$c_j$的点，最多数目的类别即为预测类别。
- k近邻法的特殊情况是k=1的情形，称为最近邻算法。最近邻算法是将与x最邻近的点的类作为x的类。



## 3.2 K近邻模型

### 模型

当**训练集、距离度量、k值和分类决策规则**确定后，对于任何一个新的输入实例，它所属的类唯一地确定。



### 距离度量

> **特征空间**中的两个实例点的距离是两个实例点相似程度的反映。

这里用到了$L_p$距离，本质上就是空间中两个实例点坐标差值的p范数。
$$
L_p(x_i, x_j)=\left(\sum_{l=1}^{n}{\left|x_{i}^{(l)}-x_{j}^{(l)}\right|^p}\right)^{\frac{1}{p}}
$$


1. $p=1$ 对应 曼哈顿距离
2. $p=2$ 对应 欧氏距离
3. p = $\infin$ 对应 各个坐标距离的最大值
4. 任意$p$ 对应 闵可夫斯基距离

![Lp距离](D:\Machine_Learning\LihangBook\Lp距离.jpg)



### k值的选择

==k值的选择会对k近邻法的结果产生重大影响==

- 如果k值较小，学习的近似误差会减小，估计误差会增大->模型变得复杂，容易发生过拟合。
- 如果k值较大，学习的近似误差会增大，估计误差会减小->模型变得简单
- k值一般取一个较小的数值，通过**交叉验证法**选取最优参数。
- 二分类问题k值取奇数可以避免平票问题。



### 分类决策规则

**多数表决规则**（Majority Voting Rule）

误分类率

$\frac{1}{k}\sum_{x_i\in N_k(x)}{I(y_i\ne c_i)}=1-\frac{1}{k}\sum_{x_i\in N_k(x)}{I(y_i= c_i)}$

如果分类损失函数是0-1损失，误分类率最低即经验风险最小。



## *3.3 kd树

k近邻法的关键是如何对训练数据进行**==快速的k近邻搜索==**，当特征空间维数大及训练数据容量大时尤其必要。

- 最简单的实现方法是线性扫描，但非常耗时。使用特殊的数据结构可以减少计算距离的次数。

- kd树是存储k维空间数据的树结构，是平衡二叉树。
- kd树的k和k近邻的k意义不同。
- 在**scipy.spatial.KDTree**中有KDTree的实现，使用起来很方便。



### 构造kd树

KDTree的构建是一个递归的过程

注意: KDTree左边的点比父节点小，右边的点比父节点大（平衡二叉树）

平衡的KDTree搜索时效率未必是最优的，这个和样本分布有关系。随机分布样本**KDTree搜索**(这里应该是**最**近邻搜索)的平均计算复杂度是$O(\log N)$，空间维数$K$接近训练样本数$N$时，搜索效率急速下降，几乎接近$O(N)$。

即如果维度比较高，搜索效率很低。当然，在考虑维度的同时也要考虑样本的规模。

> 输入：k维空间数据集$T = {x_1, x_2, ···,x_N}$，其中$x_i = (x_i^{(1)},x_i^{(2)},···,x_i^{(k)})^T$，$i = 1,2,···，N$；
>
> 输出：kd树
>
> （1）开始：构造根节点，根结点对应于包含T的k维空间的超矩形区域
>
> 选择$x^{(1)}$为坐标轴，以$T$中**所有实例的$x^{(1)}$坐标的中位数**为切分点，将根节点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴$x^{(1)}$垂直的超平面实现。
>
>   由根节点生成深度为1的左、右子结点：左子节点对应坐标$x^{(1)}$小于切分点的子区域，右子节点对应坐标$x^{(1)}$大于切分点的子区域。
>
>   将落在切分超平面上的实例点保存在根节点。
>
> （2）重复：对于深度为$j$的结点，选择$x^{(l)}$为坐标轴，$l = j(mod)k + 1$。以$T$中**所有实例的$x^{(l)}$坐标的中位数**为切分点，将根节点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴$x^{(l)}$垂直的超平面实现。
>
>   由该节点生成深度为$j+1$的左、右子结点：左子节点对应坐标$x^{(l)}$小于切分点的子区域，右子节点对应坐标$x^{(l)}$大于切分点的子区域。
>
> 将落在切分超平面上的实例点保存在该节点。
>
> （3）直到两个子区域没有实例存在时停止。从而形成kd树的区域划分。



### 搜索kd树

- 书中的kd树搜索实现的时候针对了一种$k=1$的特殊的情况，实际是**最**近邻搜索。

> 输入：已构造的kd树，目标点x；
>
> 输出：x的最近邻。
>
> （1）：在kd树中找出包含目标点x的叶结点：从根节点出发，递归地向下访问kd树。若目标点x当前维的坐标小于切分点的坐标，则移动到左子结点，否则移动到右子结点，直到子结点为叶结点为止。
>
> （2）：以此叶结点为“当前最近点”。
>
> （3）：递归地向上回退：在每个结点进行以下操作：
>
> ​           （a）如果该结点保存的实例点比当前最近点距离目标点更近，则以该结点为“当前最近点”。
>
> ​           （b）当前最近点一定存在于该结点的一个子结点对应的区域。检查该子结点的父结点的另一子结点对应的区域是否有更近的点。具体地，检查另一子结点对应的区域是否与以目标点为球心、以目标点和“当前最近点”间的距离为半径的超球体相交。
>
> ​              如果相交没可能在另一子结点对应的区域内存在距目标更近的点，移动到另一子结点，递归地进行最近邻搜索。
>
> ​              如果不相交，向上回退。
>
> （4）：当回退到根结点时，搜索结束。最后的“当前最近点”即为x的最近邻点。