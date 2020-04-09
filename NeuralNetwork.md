<font face="宋体">

# 神经网络

## 1.什么是神经网络

* **定义**：具有适应性的简单单元组成的广泛并行互联的网络其组织能够模拟生物神经系统对真实世界物体所做出的交互反应
  * **组成**：**神经元模型**：兴奋——&gt;发送物质——&gt;引起相连神经元电位变化——&gt;达到阈值——&gt;激活——&gt;兴奋
    * **组成部分**
      * n个其他神经元的**输入信号**
      * n个**带权重的连接**
      * **阈值**
      * **激活函数**及其产生的**输出**

## 2.为什么要用神经网络

* **仿生学**原理：**模仿人脑的结构**，获得最好的表现

## 3.神经网络怎么训练：误差逆传播算法BP

### 3.1 误差逆传播算法组成

* 适用范围：多层前馈神经网络和其他神经网络
* **关键思想：从后往前，层层求导**
* **组成**
  * **输入层***d*个输入：接收输入$x_i$
  * **隐含层***q*个神经元：接收输入$\alpha_h = \sum_{i=1}^dv_{ih}x_i$，其中$v_{ih}$为权重，输入经过激活函数以后和阈值$\gamma_h$进行比较，输出为$b_h$
  * **输出层***l*个神经元：接收输入$\beta_j = \sum_{h=1}^qw_{hj}b_h$，其中$w_{hj}$为权重，输入经过激活函数以后和阈值$\theta_j$进行比较，输出为$y_i$
* 一般**输入个数为参数个数，输出个数为标记个数**
* **参数个数**：$d*q$个权重$+q$个阈值$+q*l$个权重$+l$个阈值
* **输出**：假定$\hat{\pmb {y}_k} = (\hat{y_1^k}, \hat{y_2^k}, ..., \hat{y_l^k})$，其中$\hat{y_j^k} = f(\beta_j - \theta_j)$
* **优化目标：均方误差** 对第k个样例求所有l个属性值的均方误差。1/2求导时使用
  $E_k = \frac{1}{2}\sum_{j=1}^l(\hat{y}_j^k - {y_j^k})$
* **假设隐层和输出层神经元都是用Sigmoid函数（实际上是对数函数）**

### 3.2 参数更新方法（以隐层到输出层的连接权$w_{hj}$为例）

* **策略：梯度下降** 以目标的的负梯度方向进行参数调整
* 对均方误差$E_k$和学习率$\eta$**求第二层权重的改变量**：**均方误差对权重求导的过程**
  1. **总目标**：$\varDelta w_{hj} = -\eta \frac{\partial E_k}{\partial w_{hj}}$
  2. 由**链式求导法则**可得：
  $\frac{\partial E_k}{\partial w_{hj}} = \frac{\partial E_k}{\partial \hat{y}_j^k} \cdot \frac{\partial \hat{y}_j^k}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial w_{hj}}$
  3. 根据$\beta_j$的**定义**可得：
  $\frac{\partial \beta_j}{\partial w_{hj}} = b_h$
  4. **Sigmoid函数的性质**
  $f'(x) = f(x)(1 - f(x))$
  5. 根据*定义*
  $f(\beta_j - \theta_j) = \hat{y_j^k}$
  6. 由[1], [3], [4], [5]可得**输出层神经网络的梯度项**
  $\begin{aligned} \pmb g_j &= -\frac{\partial E_k}{\partial \hat{y}_j^k} \cdot \frac{\partial \hat{y}_j^k}{\partial \beta_j} \\ &= -(\hat{y}_j^k - {y_j^k})f'(\beta_j - \theta_j) \qquad (各自求导) \\ &= \hat{y}_j^k(1 - \hat{y}_j^k)({y_j^k} - \hat{y}_j^k) \qquad (sigmoid函数的性质) \end{aligned}$
  7. 综上所述，可得
  $\varDelta w_{hj} = \eta\pmb g_jb_h$
  $\varDelta\theta_j = -\eta\pmb g_j$
  8. 类似可得**隐含层神经网络的梯度项**
  $\begin{aligned} e_h &= - \frac{\partial E_k}{\partial b_h} \cdot \frac{\partial b_h}{\partial a_h} \\ & = -\sum_{j=1}^l \frac{\partial E_k}{\partial \beta_j} \cdot \frac{\partial \beta_j}{\partial b_h} f'(a_h-\gamma_h) \qquad (利用上一层的梯度项)\\ & = \sum_{j=1}^lw_{hj}\pmb g_j f'(a_h - \gamma_h)\qquad (将\pmb g_j代入) \\ & = b_h(1 - b_h)\sum_{j=1}^lw_{hj}\pmb g_j \qquad (sigmoid函数的性质) \end{aligned}$
  9. 综上所述可得
  $\varDelta v_{ih} = \eta\pmb e_hx_i$
  $\varDelta\theta_j = -\eta\pmb e_h$
* **结果分析**
  * $\varDelta w_{hj} = \eta\pmb g_jb_h$ 和下层的权重$\pmb g_j$和本层的输出$b_h$有关
  * $\varDelta\theta_j = -\eta\pmb g_j$ 本层的阈值只和梯度项有关
  * $\varDelta v_{ih} = \eta\pmb e_hx_i$ 类似
  * $\varDelta\theta_j = -\eta\pmb e_h$

### 3.3 算法流程

* **输入**
  * 将输入示例提供给输入层神经元
  * 逐层将信号向前传
  * 产生输出层的结果$y_j^k$
* **迭代**
  * 计算输出层的误差$\pmb g_j$
  * 将误差逆向传播至隐层神经元$\pmb e_h$
  * 根据隐层神经元的误差来对连接权和阈值进行调整
    * 更新阈值
      * $\varDelta\theta_j = -\eta\pmb g_j$
      * $\varDelta\theta_j = -\eta\pmb e_h$
    * 更新权重
      * $\varDelta w_{hj} = \eta\pmb g_jb_h$
      * $\varDelta v_{ih} = \eta\pmb e_hx_i$
* **判断当前的累积误差是否已经足够小**
  $E = \frac{1}{m}\sum_{k=1}^mE_k, \qquad (E_k为单个样本的均方误差)$

### 3.4 算法实现

* **输入**：$训练集D = \{(\pmb x_k, \pmb y_k)\}_{k=1}^m，学习率\eta$
* **输出：连接权与阈值确定的多层前馈神经网络**
* **伪代码**

```python
"""在(0,1)范围内随机初始化网络中所有连接权和阈值"""
do{
  for (所有（xk,yk）属于集合D){
    """
    1.计算输出层的误差
      根据当前参数和式计算当前样本的输出yk
      计算输出层神经元的梯度项gj

    2.将误差逆向传播至隐层神经元
      计算隐层神经元的梯度项eh

    3.更新连接权和阈值
    """
    }
}while(没达到停止条件)
```

### 3.5 算法分类

* **标准BP算法**
  * **优化目标**：**均方误差**$E_k$
  * **更新参数的单位**：每一个**样例**进行一次
  * 可能出现**抵消效果**
* **累计BP算法**
  * **优化目标**：**累积误差**
  * **更新参数的单位**：针对整个**训练集**
  * 到了**一定迭代次数后进一步下降很缓慢**，不如标准快

### 3.6 过拟合针对

* **早停**
  * 将数据分成**训练集**和**验证集**
    * 训练集用来**计算梯度、更新连接权和阈值**
    * 验证集用来**估计误差**
    * 若**训练误差降低但验证误差升高**，则停止训练直接返回

* **正则化**
  * **基本思想**：在误差目标函数中增加一个用于**描述网络复杂度**的部分
    * 例如：**连接权与阈值的平方和**
    * 取$\omega_i$表示连接权和阈值，则误差目标函数改变为
    $E = \lambda\frac{1}{m}\sum_{k=1}^mE_k + (1-\lambda)\sum_i\omega_i^2$
      * $\lambda$取值为$(0,1)$
      * 用于对经验误差和网络复杂度这两项进行折中，通过交叉验证法来估计

### 3.7 参数择优

* **目标：累计误差最小，是关于$\omega$和$\theta$的函数，全局最小**
  * 局部最小：邻域中最小
  * 全局最小：整个参数空间中最小
* **负梯度方向：函数值下降最快的方向**
* 当梯度项为0时，达到局部最小，并停止更新，但**可能有多个局部最小，避免参数最优陷入局部最小**
* **跳出局部极小**，进一步接近全局最小
  1. 以**多组不同的参数值初始化**，取其中误差最小的参数
  2. **模拟退火**：每一步都**以一定概率接受次优解**。并且**这个概率随着时间的推移而逐渐降低**，以保证算法稳定
  3. 使用**随机梯度下降**：在计算梯度下降时加入随机因素
  4. **遗传算法**

</font>
