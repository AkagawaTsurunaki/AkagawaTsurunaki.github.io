# 神经辐射场 NeRF 相关公式推导

> Ciallo～(∠・ω< )⌒★ 我是赤川鹤鸣!
> 学习了 [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934) 这篇论文后, 看到里面的一些公式, 思考着它们是怎么来的, 同时查询了很多资料和博客, 现在将它们的推导汇总起来.

## 论文中的相关概念

**体积密度** $\sigma (\mathbf{x}) $ 可认为是光线终止于一个位于 $\mathbf{x}$ 处的无限小粒子的微分概率.

相机光线 $ \mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$ 的期望颜色为 $C(\mathbf{r})$, 其近界和远界分别为 $t_n$ 和 $t_f$.

函数 $T(t)$ 指的是光线从 $t_n$ 到 $t$ 的**累计通透率**, 即光线从 $t_n$ 传播到 $t$ 而没有与任何其他粒子碰撞的概率.

## 累计通透率 $T(t)$ 的公式推导

光线从 $t_n$ 到 $t + \Delta s$ 的累计通透率 $T (t+\Delta s)$, 可以认为是光线传播到 $t$ 时的累计通透率 $T(n)$ 与光线在接下来的 $\Delta s$ 的传播过程中撞不到任何粒子的累计概率的乘积.

不妨反过来思考, 光线在位置 $t + \Delta s$ 时撞到粒子的概率是

> 思考概率密度函数和累计概率函数的差别, 我们这里类似于计算的是 $P(\tau = t + \Delta s)$.

$$
\sigma \left( \mathbf{r}(t + \Delta s) \right)
$$

在一个极小的 $[t, t + \Delta s]$ 区间内, 对概率密度函数 $\sigma$ 进行近似的积分

$$
\int_{t}^{t+\Delta s} \sigma \left( \mathbf{r}(t) \right) \mathrm{d} t = \Delta s \sigma \left( \mathbf{r}(t + \Delta s) \right)
$$

> 思考概率密度函数和累计概率函数的差别, 我们这里类似于计算的是 $P(t \leq \tau \leq t + \Delta s)$.

因此得到

$$
T (t+\Delta s) = T(t) \left(1-\Delta s \sigma \left( \mathbf{r}(t + \Delta s) \right)  \right)
$$

> 注意要用 1 减去, 因为我们要的是不碰撞的那一部分.

回想起导数的定义, 不妨求解 T(t) 的导数,

$$
\begin{align*}
        \lim_{\Delta s \rightarrow 0} \dfrac{T (t+\Delta s) - T(t)}{\Delta t}
        &= \lim_{\Delta s \rightarrow 0} \dfrac{T(t) \left(1-\Delta s \sigma \left( \mathbf{r}(t + \Delta s) \right)  \right) - T(t)}{\Delta s} \\
        & = \lim_{\Delta s \rightarrow 0} -T(t) \sigma \left( \mathbf{r}(t) + \Delta s\right) \\
        \dfrac{\text{d}T}{\text{d}s} &= -T(t) \sigma \left( \mathbf{r}(t)\right)
    \end{align*}
$$

整理等式两侧

$$
\dfrac{1}{T(t)} \text{d}T = - \sigma \left( \mathbf{r}(t)\right) \text{d}s
$$

积分

$$
\int_{t_n}^{t} \dfrac{1}{T(t)} \text{d}T = - \int_{t_n}^{t} \sigma \left( \mathbf{r}(t)\right) \text{d}s
$$

回想起$\log(x)$的导数是$\dfrac{1}{x}$,

$$
\begin{align*}
        \log T(t) |_{t_n}^t &= - \int_{t_n}^{t} \sigma \left( \mathbf{r}(t)\right) \text{d}s \\
        \log \dfrac{T(t)}{T(t_n)} &= - \int_{t_n}^{t} \sigma \left( \mathbf{r}(t)\right) \text{d}s
    \end{align*}
$$

两侧取指数, 则上式化为

$$
\dfrac{T(t)}{T(t_n)} = \exp \left(- \int_{t_n}^{t} \sigma \left( \mathbf{r}(t)\right) \text{d}s \right)
$$

令 $T(t_n) = 1$ 因为 $\sigma \left( \mathbf{r}(t_n)\right) = 0$ 得

$$
T(t)= \exp \left(- \int_{t_n}^{t} \sigma \left( \mathbf{r}(t)\right) \text{d}s \right)
$$

## 期望颜色 $C(\mathbf{r})$ 的推导

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma \left( \mathbf{r}(t)\right) \mathbf{c} \left( \mathbf{r}(t), \mathbf{d} \right) \text{d} t
$$

> 我们可以直观地想象这个情景, 例如你在一个雾蒙蒙的天气看见一颗红色的苹果, 此时 $T$ 就是在说来自苹果光线有多大的通透率（穿越迷雾）传到你的眼睛里. $\sigma$ 说明了苹果表面上的某一个粒子有多大概率是正好是终结光线（光线不能穿透苹果本体）的那个粒子.

这个公式源自 [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf).

## 黎曼和、定积分

离散化这个积分之前, 我们先要了解什么是**黎曼和**、**定积分**.

### 黎曼和

<img src="/images/arbitrary-partitions.gif" width="500" alt="黎曼和">

设函数 $f(x)$ 在 $[a,b]$ 上有定义, 在 $[a,b]$ 上任意插入若干个分点

$$
a = x_0 < x_1 < x_2\cdots < x_n = b
$$

这些分点的集合 $ P=\{x*0, x_1, x_2, \cdots, x*{n}\} $ 称为 $[a,b]$ 的一个**划分**.

划分 $P$ 定义了 $n$ 个子区间

$$
[x_0, x_1], [x_1, x_2], \cdots, [x_{n-1}, x_n]
$$

它们的长度依次为

$$
\Delta x_1=x_1-x_0,\Delta x_2=x_2-x_1,\cdots,\Delta x_n=x_n-x_{n-1}
$$

<img src="/images/select-xi-for-height.gif" width="500" alt="选择 xi 的高">

在每个子区间 $[x_{k-1},x_k]$ 上任取选取一个数 $\xi_k$, 以 $[x_{k-1},x_k]$ 为底, $f(\xi_i)$ 为高构造矩形, 这些矩形的和

$$
A_n=\sum_{k=1}^{n}f(\xi_k)\Delta x_k
$$

称为函数 $f$ 在区间 $[a,b]$ 上的**黎曼和**.

### 定积分

<img src="/images/definite-integral.gif" width="500" alt="定积分">

设函数 $f(x)$ 在 $[a,b]$ 上有定义, 对于 $[a,b]$ 上的任意划分 $P$, $\xi_k$ 为子区间 $[x_{k-1},x_k]$ 上任意选取的数, 子区间 $[x_{k-1},x_k]$ 的长度为 $\Delta x_k$, 记

$$
\lambda=max\{\Delta x_1,\Delta x_2,...,\Delta x_n\}
$$

> 实际上, $\lambda$ 就是你所划分的那些块中最大长度的那块, 目的是令最大的那块趋近于无穷小.

如果下述极限存在

$$
I=\lim_{\lambda\to0}\sum_{k=1}^{n}f(\xi_k)\Delta x_k
$$

则称被积函数 $f(x)$ 在积分区间 $[a,b]$ 可积 , $a$ 为**积分下限**, $b$ 为**积分上限**, $I$ 为 $f(x)$ 在 $[a,b]$ 上的**定积分**, $x$ 为**积分变量**, 可以标记如下

$$
I=\int_{a}^{b}f(x)dx
$$

## 期望颜色 $C(\mathbf{r})$ 的积分离散化

> 下图是我自己做的, 有 svg 非位图版本, 若有需要请邮箱联系.

<img src="/images/discrete-different.svg" width="1000" alt="积分离散化">

结合论文的内容, 以及上图, 积分区间为 $[t_n, t_f]$, 将其划分为 $N$ 个区间, 得到划分 $P$, 在这个划分 $P$ 中, 第 $i$ 个区间中任取一点 $t_i$, 且满足

$$
t_i \sim \mathcal{U} \left[ t_n + \dfrac{i-1}{N} \left( t_f - t_n \right), t_n + \dfrac{i}{N} \left( t_f - t_n \right) \right], \ i = 1,  \ 2, \ 3, \cdots,\ N
$$

这样第 $i$ 个区间所对应的矩形记为 $\hat{C}(\mathbf{r})_i$, 而根据期望颜色的公式可得

$$
C(\mathbf{r})_{i} =\int_{t_i}^{t_{i+1}}T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\mathrm{d}t
$$

注意到, $\sigma$ 和 $\mathbf{c}$ 与积分变量 $t$ 有关, 又因为这段区间极小, 可以认为介质近似均匀, 所以

$$
\sigma \left( \mathbf{r}(t) \right) = \sigma_i
$$

$$
\mathbf{c} \left(\mathbf{r} (t), \mathbf{d}\right) = \mathbf{c}_i
$$

视为常数, 由多层感知机给出结果.

然而 $T(t)$ 与积分变量 $t$ 和 $s$ 都有关, 无法忽略积分变量在积分区间内造成的影响, **不能视为常数**.

> 进一步为什么不能将它近似视为$T_i$？
>
> $T(t,s)$ 可视为二元函数, 对它积分就类比于求体积, 而 $\sigma(t)$ 是一元函数, 对它积分类比于求面积, 我们不能把体积视为面积, 因为它们具有不同的维度.

因此, 代入并将常量提到积分号之前

$$
\begin{align*}
        \hat{C}(\mathbf{r})_{i} &= \int_{t_i}^{t_{i+1}}\exp\left(-\int_{t_n}^t\sigma(s)\mathrm{d}s\right)\sigma_i\mathbf{c}_i\mathrm{d}t \\
                          &= \sigma_i\mathbf{c}_i \int_{t_i}^{t_{i+1}} \exp\left(-\int_{t_n}^t\sigma(s)\mathrm{d}s\right) \mathrm{d}t
    \end{align*}
$$

将嵌套的积分拆开得

$$
\begin{align*}
        \hat{C}(\mathbf{r})_{i} &= \sigma_i\mathbf{c}_i \int_{t_i}^{t_{i+1}} \exp \left( - \left( \int_{t_n}^{t_i}\sigma(s)\mathrm{d}s + \int_{t_i}^{t}\sigma(s)\mathrm{d}s \right)\right) \mathrm{d}t \\
                          &= \sigma_i\mathbf{c}_i \int_{t_i}^{t_{i+1}} \exp \left(-\int_{t_n}^{t_i}\sigma(s)\mathrm{d}s\right) \exp \left(-\int_{t_i}^{t}\sigma(s)\mathrm{d}s\right) \mathrm{d}t
    \end{align*}
$$

不难发现, $\int_{t_n}^{t_i}\sigma(s)\mathrm{d}s$ 中已经不存在积分变量 $t$, 因此根据之前的公式, 可记常量

$$
T_{i} = \exp \left( - \int_{t_n}^{t_i}\sigma(s)\mathrm{d}s\right) = \exp\left(-\sum_{j=1}^{i-1}\sigma_{j} \left(t_{j+1} - t_j\right)\right)
$$

而区间 $[t_i, t]$ 也是极小的, 故 $\int_{t_i}^{t}\sigma(s)\mathrm{d}s $ 也可以视为长为 $t - t_i$ , 高为 $\sigma_i$ 的矩形, 即

$$
\int_{t_i}^{t}\sigma(s)\mathrm{d}s = \sigma_i \left( t - t_i\right)
$$

代入得

$$
\begin{align*}
        \hat{C}({\bf r})_{i} &= \sigma_{i}{\bf c}_{i}T_{i} \int_{t_{i}}^{t_{i+1}}\exp(-\sigma_{i}(t-t_{i}))\mathrm{d}t \\
                    &= \sigma_{i}\,\mathbf{c}_{i}\,T_{i}\,{\frac{e^{-\sigma_{i}(t-t_{i})}}{-\sigma_{i}}}{\Big|}_{t_{i}}^{t_{i+1}} \\
                    &= \mathbf{c}_{i} T_i \left( 1 - e^{-\sigma_i \left(t_i -t_{i-i}\right)} \right)
    \end{align*}
$$

记第 $i$ 个区间 $[t_i, t_{i+1}]$ 的长度 $ \delta*i = t_i -t*{i-i}$, 那么

$$
\hat{C}({\bf r})_{i} = \mathbf{c}_{i} T_i \left( 1 - e^{-\sigma_i \delta_i} \right)
$$

将每个区间的 $\hat{C}({\bf r})_{i}$ 加起来, 就得到了离散化的积分

$$
\hat{C}(\mathbf{r})=\sum_{i=1}^N \hat{C}(\mathbf{r})_i
$$

即

$$
\hat{C}({\bf r})=\sum_{i=1}^{N}T_{i}(1-\exp(-\sigma_{i}\delta_{i})){\bf c}_{i}\,,\;\;{\text{where}}\;\;T_{i}=\exp\left(-\sum_{j=1}^{i-1}\sigma_{j}\delta_{j}\right)\,
$$

完毕.

> 【文献/资料引用】
>
> - [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934)
> - [Optical Models for Direct Volume Rendering](https://courses.cs.duke.edu/spring03/cps296.8/papers/max95opticalModelsForDirectVolumeRendering.pdf)
> - [【论文精读】NeRF中的数学公式推导](https://blog.csdn.net/YuhsiHu/article/details/124318473)
> - [NERF公式推导与理解](https://blog.csdn.net/zfkdsghnb/article/details/140125493)
> - [NeRF：用深度学习完成3D渲染任务的蹿红](https://zhuanlan.zhihu.com/p/390848839)
> - [什么是黎曼和？什么是定积分？](https://www.zhihu.com/tardis/zm/art/49577150)
