# 球坐标下的 Laplace 算子推导

> Ciallo～(∠・ω< )⌒★ 我是赤川鹤鸣！在学习球谐函数的时候，第一次听说**球坐标**下的 **Laplace 算子**这一概念. 在查阅了一些资料后，现在整理出球坐标下拉普拉斯算子的推导公式.

## 1. 球坐标

我们非常熟悉的坐标系是初中时学习过的具有 $2$ 个维度 $x$ 和 $y$ 的平面直角坐标系. 然后在高中，我们将平面直角坐标系拓展到了具有 $3$ 个维度的空间直角坐标系，下图（引用自[球坐标系 - 小时百科](https://wuli.wiki/online/Sph.html)）就是用 $x$、$y$ 和 $z$ 三个互相垂直的维度表示的空间直角坐标系.

<img src="/images/xiaoshibaike-spherical-coordinates.svg" alt="球坐标系" width="300">

如果想定义球坐标，在我们熟悉的这个空间直角坐标系中建立定义是最简便的.

在空间直角坐标系中存在某一点 $P$，那么

| 名称                      | 符号              | 定义                                                          | 约束                                         |
| ------------------------- | ----------------- | ------------------------------------------------------------- | -------------------------------------------- |
| 位矢（position vector）   | $\boldsymbol {r}$ | 坐标原点$O$（球心）到点 $P$ 的向量 $\vec{OP}$.                | -                                            |
| 位矢的模                  | $r$               | 位矢$\boldsymbol {r}$ 的模长，即点 $P$ 与坐标原点 $O$ 的距离. | $r \geq 0$                                   |
| 极角（polar angle）       | $\theta$          | 位矢$\boldsymbol {r}$ 与 $z$ 轴的夹角.                        | $\theta \in [0, \pi] $                       |
| 方位角（azimuthal angle） | $\phi$            | $\boldsymbol {r}$ 在 $xOy$ 平面上的投影与 $x$ 轴的夹角.       | $\phi \in [0, \ 2 \pi) $ 或 $ (−\pi,\ \pi] $ |

因此，点 $P$ 可以用 $(r, \ \theta, \ \phi)$ 这 $3$ 个有序实数来表示，称为该点的**球坐标（spherical coordinates）**.

当然，我们可以把球坐标系中的坐标转化到空间直角坐标系中的坐标.

根据图中角与向量的关系，我们可以得到位矢在空间直角坐标系中的分量

$$
\boldsymbol{x} = \sin{\theta} \cos{\phi} \boldsymbol{\hat{x}}
$$

$$
\boldsymbol{y} = \sin{\theta} \sin{\phi} \boldsymbol{\hat{y}}
$$

$$
\boldsymbol{z} = \cos{\theta} \boldsymbol{\hat{z}}
$$

其中，$\boldsymbol{\hat{x}} = (1, \ 0, \ 0)^T$，$\boldsymbol{\hat{y}} = (0, \ 1, \ 0)^T$，$\boldsymbol{\hat{z}} = (0, \ 0, \ 1)^T$，它们是空间直角坐标系中的一组单位基.

因此，单位位矢 $\boldsymbol {\hat{r}}$ 可以表达为

$$
\boldsymbol {\hat{r}} = \boldsymbol {x} + \boldsymbol {y} + \boldsymbol {z}
$$

那么引入常量 $r$，位矢就可以表达为

$$
\boldsymbol{r} = r \cdot \boldsymbol {\hat{r}}
$$

## 2. 拉梅系数

球坐标系虽然定义在空间坐标系中，但是空间坐标系的 $x$，$y$，$z$ 在三个方向上都可以分别从负无穷取到正无穷；然而，在球坐标系中，$r$ 必须大于或等于 $0$，$\theta$ 和 $\phi$ 的定义域也都有自己的限制，从这个角度上说，它们都没法取到实数域上的每一个值.

如果我们按照三个维度等分整个空间坐标系会发生什么？比如说，我们按照 $0.01$ 为边长，把整个空间坐标系按 $x$，$y$，$z$ 三个维度切割后，就会得到很多很多边长为 $0.01$ 的完全相同的小块.

然而在球坐标系中，两个带角度的维度的稍稍变化会在球面上划出一个曲面，而稍稍拉长一下位矢，但不改变位矢的方向，就会让这个曲面向外微微膨胀，但这个小块不是一个正方体，而是像一个稍稍掰弯的六面体.

可见，球坐标系看起来不太寻常！空间坐标系看起来是“均匀”的，但球坐标系是“不均匀”的.

请观察我绘制的这个小块的图像. 在空间坐标系下，假设点 $P$ 产生了 $(\mathrm{d} x, \ \mathrm{d}y, \ \mathrm{d}z)$ 的位移，那么沿着这三个维度的形成了一个边长（弧长）分别为 $ \mathrm{d} l_1 ,\mathrm{d} l_2, \mathrm{d} l_3 $ 的六面体，称为**微元**.

<img src="/images/coordinates-differentia-element.jpg" alt="坐标系的微元" width="300">

我们知道球坐标系和平面坐标系存在着差异，因此它们的微元弧长也存在某种关系.

以 $u_i$ 这个基（轴）为例，弧长 $\mathrm{d} l_i$ 满足下式

$$
\mathrm{d} l_i = H_i \mathrm{d} u_i
$$

如果在 $n$ 个基上定义了位矢 $\boldsymbol{r} = (u_1, u_2, \dots, u_n)^T$ ，那么就有

$$
\mathrm{d} \boldsymbol{r} = \sum_{i=1}^n \dfrac{\partial \boldsymbol{r}}{\partial u_i} \mathrm{d} u_i
$$

其中 $\dfrac{\partial \boldsymbol{r}}{\partial u_i}$ 称为**切矢**.

我们通常希望用一个单位基向量来表示切矢，那么就有

$$
\dfrac{\partial \boldsymbol{r}}{\partial u_i} = H_i \boldsymbol{\hat{u}_i}
$$

其中 $\boldsymbol{\hat{u}_i}$ 是单位基向量，$H_i$ 称为**拉梅系数**.

实际上，拉梅系数表示了在这个单位基上微元的“边长”. 特别地，空间直角坐标系的拉梅系数为 $H_x = H_y = H_z = 1$，这也解释了它的“均匀性”.

## 3. 正交曲线坐标系

对球坐标系中的位矢 $\boldsymbol {r}$ 求微分

$$
\mathrm{d} \boldsymbol{r} =
    \dfrac{\partial \boldsymbol{r}}{\partial r} \mathrm{d}r + \dfrac{\partial \boldsymbol{r}}{\partial \theta} \mathrm{d} \theta + \dfrac{\partial \boldsymbol{r}}{\partial \phi} \mathrm{d} \phi
$$

将式代入得

$$
\mathrm{d} \boldsymbol{r} =
    \left( \boldsymbol {x} + \boldsymbol {y} + \boldsymbol {z} \right) \mathrm{d} r + r \left( \cos{\theta} \cos{\phi} \boldsymbol{\hat{x}} + \cos{\theta} \sin{\phi} \boldsymbol{\hat{y}} - \sin{\theta} \boldsymbol{\hat{z}} \right) \mathrm{d} \theta + r \sin{\theta} \left( -\sin{\phi} \boldsymbol{\hat{x}} +  \cos{\phi} \boldsymbol{\hat{y}} \right) \mathrm{d} \phi \\
$$

观察式子里的这三项

$$
\boldsymbol {\hat{r}} = \boldsymbol {x} + \boldsymbol {y} + \boldsymbol {z}
$$

$$
\boldsymbol {\hat{\theta}} = \cos{\theta} \cos{\phi} \boldsymbol{\hat{x}} + \cos{\theta} \sin{\phi} \boldsymbol{\hat{y}} - \sin{\theta} \boldsymbol{\hat{z}}
$$

$$
\boldsymbol {\hat{\phi}} = -\sin{\phi} \boldsymbol{\hat{x}} +  \cos{\phi} \boldsymbol{\hat{y}}
$$

计算其模长发现

$$
\begin{align*}
    | \boldsymbol {\hat{r}}| &= \sqrt{ \sin^2{\theta} \cos^2{\phi} + \sin^2{\theta} \sin^2{\phi} + \cos^2{\theta}} \\
    &= \sqrt{ \sin^2{\theta} (\cos^2{\phi} + \sin^2{\phi}) + \cos^2{\theta}} \\
    &= \sqrt{ \sin^2{\theta} + \cos^2{\theta}} \\
    &= 1
\end{align*}
$$

$$
\begin{align*}
    | \boldsymbol {\hat{\theta}}| &= \sqrt{ \cos^2{\theta} \cos^2{\phi} + \cos^2{\theta} \sin^2{\phi} + \sin^2{\theta}} \\
    &= \sqrt{ \cos^2{\theta} (\cos^2{\phi} + \sin^2{\phi}) + \sin^2{\theta}} \\
    &= \sqrt{ \cos^2{\theta} + \sin^2{\theta}} \\
    &= 1
\end{align*}
$$

$$
\begin{align*}
    | \boldsymbol {\hat{\phi}}| &= \sqrt{ \sin^2{\theta} + \cos^2{\theta}} \\
    &= 1
\end{align*}
$$

且

$$
\begin{align*}
    \boldsymbol {\hat{r}} \cdot \boldsymbol {\hat{\theta}} &= \sin{\theta} \cos{\theta} \cos^2{\phi} + \sin{\theta} \cos{\theta} \sin^2 -{\phi} - \sin{\theta} \cos{\theta} \\
    &= \sin{\theta} \cos{\theta} \left( \sin^2{\theta} + \cos^2{\theta} \right) - \sin{\theta} \cos{\theta} \\
    &= \sin{\theta} \cos{\theta} - \sin{\theta} \cos{\theta} \\
    &= 0
\end{align*}
$$

$$
\begin{align*}
    \boldsymbol {\hat{r}} \cdot \boldsymbol {\hat{\phi}} &= - \sin{\theta} \sin{\phi} \cos{\phi} + \sin{\theta} \sin{\phi} \cos{\phi}  \\
    &= 0
\end{align*}
$$

$$
\begin{align*}
    \boldsymbol {\hat{r}} \cdot \boldsymbol {\hat{\phi}} &= - \cos{\theta} \cos{\phi} \sin{\phi} + \cos{\theta} \sin{\phi} \cos{\phi}  \\
    &= 0
\end{align*}
$$

因此可知，$\boldsymbol {\hat{r}}$，$\boldsymbol {\hat{\theta}}$，$\boldsymbol {\hat{\phi}}$ 是单位正交向量.

若对空间中任意一点，式中的三个矢量都两两正交，那么这个曲线坐标系就是**正交曲线坐标系（orthogonal curvilinear coordinate system）**. 球坐标系、柱坐标系、抛物线坐标系和椭圆坐标系，乃至直角坐标系都是一种曲线坐标系.

因此式可化为

$$
\mathrm{d} \boldsymbol{r} = \boldsymbol{\hat{r}} \mathrm{d}r + r \boldsymbol{\hat{\theta}} \mathrm{d}\theta + r \sin{\theta} \boldsymbol{\hat{\phi}} \mathrm{d}\phi
$$

易知拉梅系数分别为 $H_1 = 1$，$H_2 = r$，$H_2 = r \sin{\theta}$.

## 4. 梯度与 Nabla 算子

我们在正交曲线坐标系中定义**梯度**

$$
\nabla u = \dfrac{1}{H_1} \dfrac{\partial u}{\partial u_1} + \dfrac{1}{H_2} \dfrac{\partial u}{\partial u_2} + \dfrac{1}{H_3} \dfrac{\partial u}{\partial u_3}
$$

注意，正是因为在正交曲线坐标系中，我们才带上了拉梅系数，回想起空间直角坐标系中我们的拉梅系数均为 $1$.

因此，根据式求出的拉梅系数，球坐标上对函数 $u$ 的梯度为

$$
\nabla u = \dfrac{\partial u}{\partial r} + \dfrac{1}{r} \dfrac{\partial u}{\partial \theta}+ \dfrac{1}{r \sin{\theta}} \dfrac{\partial u}{\partial \phi}
$$

其中，$\nabla$ 为 **Nabla 算子**

$$
\nabla = \dfrac{\partial}{\partial r}+ \dfrac{1}{r} \dfrac{\partial}{\partial \theta} + \dfrac{1}{r \sin{\theta}} \dfrac{\partial}{\partial \phi}
$$

## 5. 散度与 Laplace 算子

我们在正交曲线坐标系中定义**散度**

$$
\Delta u = \dfrac{1}{H_1 H_2 H_3} \left[ \dfrac{\partial }{\partial u_1} \left( \dfrac{H_2 H_3}{H_1} \dfrac{\partial u}{\partial u_1} \right) +  \dfrac{\partial  }{\partial u_2} \left( \dfrac{H_1 H_3}{H_2} \dfrac{\partial u}{\partial u_2} \right) +\dfrac{\partial}{\partial u_3} \left( \dfrac{H_1 H_2}{H_3} \dfrac{\partial u}{\partial u_3}  \right) \right]
$$

注意 $H_1 H_2 H_3$ 相当于微元体的体积，且拉梅系数 $H_i$ 不一定是常数.

因此，根据式求出的拉梅系数，球坐标上对函数 $u$ 的散度为

$$
\Delta u = \dfrac{1}{r^2 \sin{\theta}} \left[ \dfrac{\partial  }{\partial r} \left( r^2 \sin{\theta} \dfrac{\partial u}{\partial r} \right) +  \dfrac{\partial  }{\partial \theta} \left( \sin{\theta} \dfrac{\partial u}{\partial \theta} \right) +\dfrac{\partial}{\partial \phi} \left( \dfrac{1}{\sin{\theta}} \dfrac{\partial u}{\partial \phi} \right) \right]
$$

整理上式，得

$$
\Delta u = \dfrac{1}{r^2} \dfrac{\partial}{\partial r} \left( r^2 \dfrac{\partial u}{\partial r} \right) + \dfrac{1}{r^2 \sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial u}{\partial \theta} \right) + \dfrac{1}{r^2 \sin^2{\theta}}\dfrac{\partial^2 u}{\partial \phi^2}
$$

注意求偏导的时候，与其无关的变量都可以视为常数提出. 当然，你可以把括号里的每一项都求偏导打开，但是这样整个式子会非常长且不优雅.

其中，$\Delta$ 为 **Laplace 算子**，也可以写作是 $\nabla^2$，即

$$
\Delta =\nabla^2= \dfrac{1}{r^2} \dfrac{\partial  }{\partial r} \left( r^2 \dfrac{\partial }{\partial r} \right) + \dfrac{1}{r^2 \sin{\theta}}\dfrac{\partial  }{\partial \theta} \left( \sin{\theta} \dfrac{\partial }{\partial \theta} \right) + \dfrac{1}{r^2 \sin^2{\theta}}\dfrac{\partial^2}{\partial \phi^2}
$$

现在我们求出了球坐标系下的 Laplace 算子，下一期我们来推导球坐标系中 Laplace 方程的通解，从而得到球谐函数的表达式.

> 【参考资料/文献】
>
> - [球谐函数 - 小时百科](https://wuli.wiki/online/SphHar.html)
> - [球坐标系 - 小时百科](https://wuli.wiki/online/Sph.html)
> - [正交曲线坐标系 - 小时百科](https://wuli.wiki/online/CurCor.html)
> - [浅谈：拉梅系数那些事儿 - 知乎](https://zhuanlan.zhihu.com/p/194241346)
> - [数学基础 | 正交曲线坐标系中梯度、散度、旋度的理解与记忆 - 知乎](https://zhuanlan.zhihu.com/p/452461912)
