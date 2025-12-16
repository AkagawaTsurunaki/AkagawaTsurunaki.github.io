# S4 部分公式推导与详解

> Paper: [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC)
> 
> Github: https://github.com/state-spaces/s4
> 
> 本博客作者：赤川鹤鸣

## 离散时间的 SSM 推导

S4 论文给出了这样的公式

$$
x_{k} = \bar{\boldsymbol{A}} x_{k-1} + \bar{\boldsymbol{B}} u_{k} \quad 
\bar{\boldsymbol{A}} = \left( \boldsymbol{I} - \Delta / 2 \cdot \boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \Delta / 2 \cdot \boldsymbol{A} \right)
\\
y_{k} = \bar{\boldsymbol{C}} x_k \quad 
\bar{\boldsymbol{B}} = \left( \boldsymbol{I} - \Delta / 2 \cdot \boldsymbol{A} \right)^{-1} \Delta \boldsymbol{B}
\quad 
\bar{\boldsymbol{C}} = \boldsymbol{C}
$$

它是怎么来的呢？

其实，Arnold Tustin 在 1947 年的论文中提出了一种将**连续时间线性系统**转换为**离散时间系统**的方法，也叫做 Tustin 方法. 

我们从一个标准的**连续**时间状态空间模型出发


$$
\dot{x}(t) = \boldsymbol{A} x(t) + \boldsymbol{B} u(t)
$$

我们希望将其**离散化**为如下形式

$$
x_k = \boldsymbol{A} x_{k-1} + \boldsymbol{B} u_k
$$

从 $t = (k-1)\Delta $ 到 $t = k\Delta$（刚好一个时间步），对状态方程积分

$$
x_k = x_{k-1} + \int_{(k-1)\Delta}^{k\Delta} \left[ \boldsymbol{A} x(t) +\boldsymbol{B} u(t) \right] \mathrm{d}t
$$

用**梯形积分法**（上底 $x_{k-1}$ 加下底 $x_k$ 乘高 $\Delta$ 除以 2）来近似这个积分，得

$$
\int_{(k-1)\Delta}^{k\Delta} \dot{x}(t) \mathrm{d}t \approx \frac{\Delta}{2} \left( \dot{x}_{k-1} + \dot{x}_k \right)
$$
代入状态方程
$$
x_k = x_{k-1} + \frac{\Delta}{2} \left[ \boldsymbol{A} x_{k-1} + \boldsymbol{B} u_{k-1} + \boldsymbol{A} x_k + \boldsymbol{B} u_k \right]
$$


将此式进行变形整理，将带 $x_{k}$ 的项移到左边，其余放在右边，得
$$
\left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right) x_{k} = \left( \boldsymbol{I} + \dfrac{\Delta}{2}  \boldsymbol{A} \right) x_{k-1} +  \dfrac{\Delta}{2}  \boldsymbol{B}  \left(  u_k + u_{k-1} \right)
$$
若 $ \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A}$ 的逆矩阵存在，那么对上式左右同时左乘 $\left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right)^{-1}$，得
$$
x_{k} = \left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \dfrac{\Delta}{2}  \boldsymbol{A} \right) x_{k-1} + \left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right)^{-1}  \dfrac{\Delta}{2}  \boldsymbol{B}  \left(  u_k + u_{k-1} \right)
$$

在时间步很小的情况下，这里视为 $\dfrac{u_k + u_{k-1}}{2} \approx u_k$，因此上式变为
$$
x_{k} \approx \underbrace{\left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \dfrac{\Delta}{2}  \boldsymbol{A} \right)}_{\boldsymbol{\bar{A}}} x_{k-1} 
+ \underbrace{\left( \boldsymbol{I} - \dfrac{\Delta}{2}  \boldsymbol{A} \right)^{-1} \Delta \boldsymbol{B}}_{\boldsymbol{\bar{B}}} u_k
$$
这样我们就得到了 $\boldsymbol{\bar{A}}$，$\boldsymbol{\bar{B}}$ 的表达式，而 $\boldsymbol{\bar{C}}$ 就直接等于 $\boldsymbol{C}$，至此证明了原文的离散时间的 SSM 推导. 

## SSM 生成函数推导

文中提到的 SSM 卷积函数是
$$
\mathcal{K} (\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
= (\bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}} \bar{\boldsymbol{B}}, \dots)
$$

通常采用生成函数来表达 SSM 卷积函数中的卷积核，这样我们可以得到 **SSM 生成函数**（在 SSM 卷积函数上加了个 `\hat` 标记），生成函数的含义将在下一节详细介绍
$$
\hat{\mathcal{K}} (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}) \in \mathbb{C}
:= \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{B}} z + \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}} \bar{\boldsymbol{B}} z^2 + \cdots = \sum_{i=0}^{\infty} \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}}^{i} \bar{\boldsymbol{B}} z^{i}
$$
这是一个矩阵的无穷级数，而且包含了无穷次矩阵 $\bar{\boldsymbol{A}}$ 的幂操作，幂操作是非常消耗计算资源的操作. 还好，我们可以利用 **Neumann 定理**处理这个无穷级数. Neumann 定理指出
$$
\sum_{i=0}^{\infty} \boldsymbol{M}^{k} = (\boldsymbol{I} - \boldsymbol{M})^{-1}
$$
这样，无穷次矩阵幂操作化为了一次矩阵求逆操作，计算复杂度一下子就降低了很多. 

我们可以令 $\boldsymbol{M} = \bar{\boldsymbol{A}} z$，则 SSM 生成函数化为
$$
\hat{\mathcal{K}} (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})= \sum_{i=0}^{\infty} \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}}^{i} \bar{\boldsymbol{B}} z^{i}
= \sum_{i=0}^{\infty} \bar{\boldsymbol{C}}^{*} (\bar{\boldsymbol{A}}^{i}z^{i}) \bar{\boldsymbol{B}} 
= \bar{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}} z)^{-1} \bar{\boldsymbol{B}}
$$

但 SSM 卷积函数是一个无穷长度的卷积核序列，实际上计算机只能计算有限个，所以只需要前 $L$ 个就足够了，因此得到截断 SSM 卷积函数为
$$
\mathcal{K}_L (\bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
= (\bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}} \bar{\boldsymbol{B}}, \dots, \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}}^{L-1} \bar{\boldsymbol{B}})
$$

同理，也可以得到截断 SSM 生成函数
$$
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})= \sum_{i=0}^{L-1} \bar{\boldsymbol{C}}^{*} \bar{\boldsymbol{A}}^{i} \bar{\boldsymbol{B}} z^{i}
= \bar{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}}^L z^L) (\boldsymbol{I} - \bar{\boldsymbol{A}} z)^{-1} \bar{\boldsymbol{B}}
$$
因为单位根 $z^L = 1$（后面解释），所以
$$
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
= \bar{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}}^L) (\boldsymbol{I} - \bar{\boldsymbol{A}} z)^{-1} \bar{\boldsymbol{B}}
$$
为了方便起见，令 $\tilde{\boldsymbol{C}}^{*} = \bar{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}}^L) $，这样有
$$
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
= \tilde{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}} z)^{-1} \bar{\boldsymbol{B}}
$$
实际上，这就是 S4 论文卷积核算法中的第一步. 

第一节时，我们利用 Tustin 方法得到了 $\bar{\boldsymbol{A}} = \left( \boldsymbol{I} - \Delta / 2 \cdot \boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \Delta / 2 \cdot \boldsymbol{A} \right)$ 和 $\bar{\boldsymbol{B}} = \left( \boldsymbol{I} - \Delta / 2 \cdot \boldsymbol{A} \right)^{-1} \Delta \boldsymbol{B}$，从而得
$$
\begin{align*}
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
&= \tilde{\boldsymbol{C}}^{*} (\boldsymbol{I} - \bar{\boldsymbol{A}} z)^{-1} \bar{\boldsymbol{B}} \\
&= \tilde{\boldsymbol{C}}^{*} \left[ \boldsymbol{I} - \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \frac{\Delta}{2}\boldsymbol{A} \right) z\right]^{-1} \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right)^{-1} \Delta \boldsymbol{B} \\
&= \tilde{\boldsymbol{C}}^{*} \left[ \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right) - \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right)^{-1} \left( \boldsymbol{I} + \frac{\Delta}{2}\boldsymbol{A} \right) z \right]^{-1} \overline{\boldsymbol{B}} \\
&= \tilde{\boldsymbol{C}}^{*} \underbrace{\left[ \left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right) - \left( \boldsymbol{I} + \frac{\Delta}{2}\boldsymbol{A} \right) z \right]^{-1} }_{合并同类项} \underbrace{\left( \boldsymbol{I} - \frac{\Delta}{2}\boldsymbol{A} \right) \overline{\boldsymbol{B}}}_{根据 \bar{\boldsymbol{B}} = \left( \boldsymbol{I} - \Delta / 2 \cdot \boldsymbol{A} \right)^{-1} \Delta \boldsymbol{B}} \\
&= \tilde{\boldsymbol{C}}^{*} \left[ \boldsymbol{I}(1 - z) - \frac{\Delta}{2}\boldsymbol{A}(1 + z) \right]^{-1} \Delta \boldsymbol{B} \\
&= \frac{\Delta}{1 - z} \tilde{\boldsymbol{C}}^{*} \left[ \boldsymbol{I} - \frac{\Delta \boldsymbol{A}}{2 \frac{1 - z}{1 + z}} \right]^{-1} \boldsymbol{B} \\
&= \frac{2\Delta}{1 + z} \tilde{\boldsymbol{C}}^{*} \left[ 2 \frac{1 - z}{1 + z} \boldsymbol{I} - \Delta \boldsymbol{A} \right]^{-1} \boldsymbol{B} \\
&= \frac{2}{1 + z} \tilde{\boldsymbol{C}}^{*} \left[ \frac{2}{\Delta} \frac{1 - z}{1 + z} \boldsymbol{I} -  \boldsymbol{A} \right]^{-1} \boldsymbol{B}
\end{align*}
$$
即便如此，该式子包含了一个矩阵逆运算，需要想办法将它化为其他形式以简化运算. S4 论文提出了 NPLR 方法可以将矩阵 $\boldsymbol{A}$ 分解为正则项和低秩项之和，也就是
$$
\boldsymbol{A} = \boldsymbol{\Lambda} + \boldsymbol{P} \boldsymbol{Q}^{*}
$$
需要注意的是，这里的矩阵 $\boldsymbol{A}$ 是未经过离散化的矩阵，一旦离散化，我们会在它上面加一个横线符号来表示. 而且，NPLR 并不会加速 SSM 生成函数的计算速度，它的目的是保持**数值稳定**. 

因此，SSM 生成函数又可以化为
$$
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}})
= \frac{2}{1 + z} \tilde{\boldsymbol{C}}^{*} \left[ \frac{2}{\Delta} \frac{1 - z}{1 + z} \boldsymbol{I} - \boldsymbol{\Lambda} + \boldsymbol{P} \boldsymbol{Q}^{*} \right]^{-1} \boldsymbol{B}
$$
根据 Woodbury 恒等式（注意下式中的符号和上下文中的 SSM 无关）：
$$
(\boldsymbol{A} + \boldsymbol{UCV})^{-1} = \boldsymbol{A}^{-1} - \boldsymbol{A}^{-1} \boldsymbol{U} (
\boldsymbol{C}^{-1} + \boldsymbol{V}\boldsymbol{A}^{-1}\boldsymbol{U})^{-1} \boldsymbol{V} \boldsymbol{A}^{-1}
$$
对照 SSM 生成函数里方括号内的各项，令 $\boldsymbol{R}(z; \boldsymbol{\Lambda}) = \left( \frac{2}{\Delta} \frac{1-z}{1+z} - \boldsymbol{\Lambda} \right)^{-1}$（因为取逆好套公式），而我们又知道 $\boldsymbol{PI} \boldsymbol{Q}^{*} = \boldsymbol{P} \boldsymbol{Q}^{*}$，所以按照 Woodbury 恒等式，可以得到
$$
\left[ \boldsymbol{R}^{-1}(z) + \boldsymbol{PI} \boldsymbol{Q}^{*} \right]^{-1}
= \boldsymbol{R}(z) - \boldsymbol{R}(z) \boldsymbol{P} (\boldsymbol{I} + \boldsymbol{Q}^{*} \boldsymbol{R}(z) \boldsymbol{P})^{-1} \boldsymbol{Q}^{*} \boldsymbol{R}(z)
$$
把这个结果代入 SSM 生成函数中
$$
\begin{align*}
\hat{\mathcal{K}}_L (z; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}) 
&=
\frac{2}{1 + z} \tilde{\boldsymbol{C}}^{*} \left[ \boldsymbol{R}(z) - \boldsymbol{R}(z) \boldsymbol{P} (\boldsymbol{I} + \boldsymbol{Q}^{*} \boldsymbol{R}(z) \boldsymbol{P})^{-1} \boldsymbol{Q}^{*} \boldsymbol{R}(z) \right] \boldsymbol{B} \\
&= \frac{2}{1 + z} \left( \tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(z) \boldsymbol{B} - \tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(z) \boldsymbol{P} (\boldsymbol{I} + \boldsymbol{Q}^{*} \boldsymbol{R}(z) \boldsymbol{P})^{-1} \boldsymbol{Q}^{*} \boldsymbol{R}(z) \boldsymbol{B}\right)
\end{align*}
$$

这个结果与原文的唯一差别是 $\boldsymbol{I}$，原文中这里是 $1$，因为它假设输入信号是 1 维度的，因此低秩矩阵的秩其实就是 1. 

这样，实际上我们可以把 SSM 生成函数中各项用 $k_{00}(\omega)$，$k_{01}(\omega)$，$k_{10}(\omega)$，$k_{11}(\omega)$ 来简洁地表示，也就是这样

$$
\hat{\mathcal{K}}_L (z=\omega; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}) 
= \frac{2}{1 + \omega} ( \underbrace{\tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(\omega) \boldsymbol{B}}_{k_{00}(\omega)} - \underbrace{\tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(\omega) \boldsymbol{P}}_{k_{01}(\omega)} (\boldsymbol{I} + \underbrace{\boldsymbol{Q}^{*} \boldsymbol{R}(\omega) \boldsymbol{P}}_{k_{11}(\omega)} )^{-1} \underbrace{\boldsymbol{Q}^{*} \boldsymbol{R}(\omega) \boldsymbol{B}}_{k_{10}(\omega)} )
$$

这样其实我们就得到了 S4 卷积核算法中的第二步和第三步，

$$
\hat{\boldsymbol{K}}(\omega) \leftarrow \hat{\mathcal{K}}_L (z = \omega; \bar{\boldsymbol{A}}, \bar{\boldsymbol{B}}, \bar{\boldsymbol{C}}) 
= \frac{2}{1 + \omega} ( k_{00}(\omega) - k_{01}(\omega) (\boldsymbol{I} + k_{11}(\omega) )^{-1} k_{10}(\omega) )
$$

如果按照矩阵形式排列，其实就和原文一样了

$$
\begin{bmatrix}
k_{00}(\omega) & k_{01}(\omega) \\
k_{10}(\omega) & k_{11}(\omega)
\end{bmatrix}
= 
\begin{bmatrix}
\tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(\omega) \boldsymbol{B} & \tilde{\boldsymbol{C}}^{*} \boldsymbol{R}(\omega) \boldsymbol{P} \\
\boldsymbol{Q}^{*} \boldsymbol{R}(\omega) \boldsymbol{B} & \boldsymbol{Q}^{*} \boldsymbol{R}(\omega) \boldsymbol{P}
\end{bmatrix}
=
\begin{bmatrix} \tilde{\boldsymbol{C}}^{*} \\ \boldsymbol{Q}^{*} \end{bmatrix}
\boldsymbol{R}(\omega) \begin{bmatrix} \boldsymbol{B} & \boldsymbol{P}\end{bmatrix}
=
\begin{bmatrix} \tilde{\boldsymbol{C}} & \boldsymbol{Q} \end{bmatrix}^{*} \boldsymbol{R}(\omega) \begin{bmatrix} \boldsymbol{B} & \boldsymbol{P}\end{bmatrix}
$$

这里先总结一下该算法前两步的思路，首先 SSM 原本是一个迭代式，对于训练来说计算效率太低，于是将其化为卷积式，由卷积式定义了卷积函数和生成函数，但是生成函数中仍然存在无穷级数，因此使用 Neumann 定理将级数化为一个只需要求出逆矩阵的式子，但是求逆矩阵也是一个计算复杂度很高的操作，同时我们还需要对各个参数矩阵进行离散化，因此顺理成章地使用 Tustin 方法对参数矩阵进行等价变形，带入到原先的生成函数中，此时生成函数中可以用 NPLR 方法将矩阵分解，并进一步应用 Woodbury 恒等式进行分解，最后就得到了只含有上述矩阵形式的式子，利用矩阵进行训练比直接循环迭代的效率更高. 

## 离散傅里叶变换

傅里叶变换本质上就是将原本在时域上的问题转化到频域上去解决. S4 先计算生成函数在频率域（单位根上）的值，然后利用逆 FFT 高效地恢复时间域的卷积核，这避免了直接计算矩阵幂的高复杂度，这也是为什么上一步要花那么多步骤不断变化生成函数的形式. 

当然，FFT 和逆 FFT 是线性变换，本质是矩阵乘法（DFT 矩阵），因此梯度可以正常传播. 

对于一个长度为 $L$ 的序列 $\boldsymbol{K} = (K_0, K_1, \ldots, K_{L-1})$，其离散傅里叶变换定义为另一个序列 $\hat{\boldsymbol{K}} = (\hat{K}_0, \hat{K}_1, \ldots, \hat{K}_{L-1})$，其中
$$
\hat{\boldsymbol{K}}_j = \sum_{k=0}^{L-1} \bar{\boldsymbol{K}}_k \exp\left(-2\pi i \frac{jk}{L}\right)
$$
这里，$i$ 是虚数单位，$j$ 是频率，$\omega_k = e^{-2\pi i jk/L}$是单位根（后面会解释）. DFT 将时间域序列 $K$ 转换为频率域序列 $\hat{K}$. 

还记得上节我们计算的 SSM 生成函数吗？它正是这种级数求和形式，只不过我们最后将其化为更为简单的矩阵形式了，其实我们当时引入了一个变量 $z$，把卷积函数变成了生成函数，其实就是为了这一步运用傅里叶变换，我们引入的 $z $ 其实就是后面的 $\exp\left(-2\pi i \frac{jk}{L}\right)$，我们当时做的 $z = \omega$ 其实就是在取单位根. 

同理，我们既然能从时域到频域，就也能从频域回到时域，从频率域恢复时间域序列的逆运算为

$$
\bar{\boldsymbol{K}}_k = \frac{1}{L} \sum_{j=0}^{L-1} \hat{\boldsymbol{K}}_j \exp\left(2\pi i \frac{jk}{L}\right)
$$

快速傅里叶变换是离散傅里叶变换的高效算法，计算复杂度为 $O(L \log L)$，比直接计算离散傅里叶变换的 $O(L^2)$ 快得多. 当然，逆傅里叶变换就是离散逆傅里叶变换的高效实现. 

在复数平面上，单位根是方程 $ω^L=1$ 的解，即对于 $ k=0,1,…,L−1$，有 $ω=\exp(2πik/L) $. 这些单位根均匀分布在单位圆上，是离散傅里叶变换的核心. 

在实际训练的过程中，只需要把上述单位根的值带入到下面的式子即可
$$
\hat{\boldsymbol{K}}(\omega) \leftarrow \frac{2}{1 + \omega} ( k_{00}(\omega) - k_{01}(\omega) (\boldsymbol{I} + k_{11}(\omega) )^{-1} k_{10}(\omega) ) \\
\hat{\boldsymbol{K}}  = \left\{ \hat{\boldsymbol{K}}(\omega) : \omega = \exp(2\pi i k/L)  \right\}
$$
其中，$\hat{\boldsymbol{K}}$ 是 SSM 生成函数在点 $ω$ 的评估值，本质是卷积核的生成函数在频率域的表示. 

至于为什么 $j$ 没有了，记得我们的生成函数已经不带级数形式了，它被替换成了**一个更抽象的变量 $\omega$**，这个 $\omega$ 依次遍历了所有 $L$ 个单位根，从而囊括了所有的 $j$. 所有的单位根具有闭式解，可以直接通过下式得到
$$
\Omega_L = \left\{ \omega \mid \omega = \exp\left(2\pi i \frac{k}{L}\right), \quad k=0, 1, \dots, L-1 \right\}
$$
在数学上，生成函数在单位根上的评估值 $\hat{\boldsymbol{K}}(\omega)$ 正好是卷积核 $\boldsymbol{K}$ 的离散傅里叶变换，即
$$
\hat{\boldsymbol{K}} = \mathcal{F}_L K
$$
最后我们把频域上的卷积核通过逆傅里叶变换还原到时域上
$$
\bar{\boldsymbol{K}} = \mathrm{iFFT}(\hat{\boldsymbol{K}})
$$

> 参考资料
> 
> 1. [Conjugate of a Matrix: Definition, Properties, Formula, Examples](https://www.careers360.com/maths/conjugate-of-a-matrix-topic-pge)
>2. Arnold Tustin. *A method of analysing the behaviour of linear systems in terms of time series*. Journal of the Institution of Electrical Engineers - Part IIA: Automatic Regulators and Servo Mechanisms, 94(1): 130–142, 1947.
> 3. [普通生成函数 - OI Wiki](https://oi-wiki.org/math/poly/ogf/)
> 4. [什么是纽曼级数，具体展开式是啥，在哪些书中可以查到相关内容 ？ - 知乎](https://www.zhihu.com/question/319129479)
> 5. [快速傅里叶变换 - OI Wiki](https://oi-wiki.org/math/poly/fft/)