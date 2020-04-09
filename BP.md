<font face="宋体">

# 基础知识

## 1. 约定

$Z^{(h)} = A^{(in)}W^{(h)}$：隐含层的净输入
$A^{(h)} = \phi(Z^{(h)})$：隐含层激活（输出）
$Z^{(out)} = A^{(h)}W^{(out)}$：输出层的净输入
$A^{(out)} = \phi(Z^{(out)})$：输出层激活（输出）

## 2. 逻辑成本函数的计算

* **成本函数**：
  1. 首先确立建立模型时想要**最大化的可能性$L$**，假设样本个体都**相互独立**，则有假设
  $L(\pmb w)=P(y|\pmb x;\pmb w) = \prod_{i=1}^nP(y^{(i)} | \pmb x^{(i)};\pmb w)=\prod_{i=1}^n(\phi(z^{(i)}))^{y^{(i)}}((1-\phi(z^{(i)}))^{1-y^{(i)}}$
  2. 求得该函数的**对数似然函数**为：
  $l(\pmb w) = \log(L(\pmb w)) = \sum_{i=1}^n[y^{(i)}\log(\phi(z^{(i)})) + (1-y^{(i)})\log(1-\phi(z^{(i)}))]$
  3. 用**梯度下降法最小化代价函数**$J$可得
  $J(\pmb w) = -\sum_{i=1}^ny^{[i]}\log(a^{[i]}) + (1-y^{[i]})\log(1-a^{[i]})$
  * 其中$y^{[i]} = \phi(z^{[i]})$：数据集中**第i个样本**用前向传播算法计算出的Sigmoid值
  * 其中上标 $i$ 为训练集中**特定样本**的索引
* **添加正则化项减少过拟合机会**：$L2 = \lambda \|\pmb w\|_2^2 = \lambda\sum_{j=1}^mw_j^2$
  * 其中 $j$ 是**某一种特征**
* **综合以上两者可得：**$J(\pmb w) = -[\sum_{i=1}^ny^{[i]}\log(a^{[i]}) + (1-y^{[i]})\log(1-a^{[i]})] + \frac{\lambda}{2}\|\pmb w\|_2^2$

## 3. 多元分类MLP的逻辑成本函数

* 对于用于多元分类的MLP，返回的是一个有$t$**个元素的输出向量**，需要与$t*1$**维独热编码表示的向量**进行比较：
$\pmb a^{(out)} = \begin{bmatrix}0.1 \\ 0.9 \\ ... \\ 0.3 \end{bmatrix},\ y = \begin{bmatrix}0 \\ 1 \\ ... \\ 0 \end{bmatrix}$
* 将逻辑成本函数推广到网络中**所有的t激活单元**，可得**成本函数**（**不包括正则化项**）为：
$J(\pmb W) = - \sum_{i=1}^n\sum_{j=1}^ty_j^{[i]}\log(\pmb a^{[i]}_j) + (1-y_j^{[i]})\log(1-\pmb a^{[i]}_j)$
  * 其中上标 $i$ 为训练集中特定样本的索引
  * 其中下标 $j$ 为某一种特征
* **惩罚项**为
$\frac{\lambda}{2}\sum_{l=1}^{L-1}\sum_{i=1}^{u_l}\sum_{i=1}^{u_{l+1}}(w_{j,i}^{(l)})^2$
  * 其中 $u_l$ 是一个给定的1层单位数
* **综上可得**：$J(\pmb W) = - \sum_{i=1}^n\sum_{j=1}^ty_j^{[i]}\log(\pmb a^{[i]}_j) + (1-y_j^{[i]})\log(1-\pmb a^{[i]}_j) + \frac{\lambda}{2}\sum_{l=1}^{L-1}\sum_{i=1}^{u_l}\sum_{i=1}^{u_{l+1}}(w_{j,i}^{(l)})^2$

## 4. 反向传播算法

* **计算误差向量**：$\pmb\delta^{(out)} = \pmb a^{(out)} - \pmb y$
* **计算隐含层的错误项**：$\pmb \delta^{(h)} = \pmb \delta^{(out)}(\pmb W^{out^{T}})\bigodot\frac{\partial \phi(z^{(h)})}{\partial (z^{(h)})}$
  * 其中$\frac{\partial \phi(z^{(h)})}{\partial (z^{(h)})}$为**sigmoid函数的导数**，即：
  $\frac{\partial \phi(z^{(h)})}{\partial (z^{(h)})} = (a^{(h)}\bigodot(1-a^{(h)}))$
  * **所以上式可变为：**：$\pmb \delta^{(h)} = \pmb \delta^{(out)}(\pmb W^{out^{T}})\bigodot((a^{(h)}\bigodot(1-a^{(h)})))$

</font>
