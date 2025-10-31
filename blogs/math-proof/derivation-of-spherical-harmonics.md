# 球谐函数公式推导

> Ciallo～(∠·ω< )⌒★ 我是赤川鹤鸣！本期将会从拉普拉斯方程开始推导**球谐函数**，本文尽可能把所有步骤都列出来，至少有高等数学的基础知识就可以看懂.

## 1 分解 Laplace 方程

上一期，我们推导了球坐标系下的拉普拉斯算子，

$$
\Delta u = \dfrac{1}{r^2} \dfrac{\partial}{\partial r} \left( r^2 \dfrac{\partial u}{\partial r} \right) + \dfrac{1}{r^2 \sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial u}{\partial \theta} \right) + \dfrac{1}{r^2 \sin^2{\theta}}\dfrac{\partial^2 u}{\partial \phi^2}
$$

那么拉普拉斯方程为

$$
\dfrac{1}{r^2} \dfrac{\partial}{\partial r} \left( r^2 \dfrac{\partial u}{\partial r} \right) + \dfrac{1}{r^2 \sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial u}{\partial \theta} \right) + \dfrac{1}{r^2 \sin^2{\theta}}\dfrac{\partial^2 u}{\partial \phi^2} \tag{1} = 0
$$

将拉普拉斯方程 $(1)$ 两侧同时乘以 $r^2$ 可得

$$
\dfrac{\partial}{\partial r} \left( r^2 \dfrac{\partial u}{\partial r} \right) + \dfrac{1}{\sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial u}{\partial \theta} \right) + \dfrac{1}{ \sin^2{\theta}}\dfrac{\partial^2 u}{\partial \phi^2} = 0 \tag{2}
$$

令 $u(r, \theta, \phi) = R(r) Y(\theta, \ \phi)$，则

$$
\dfrac{\partial}{\partial r} \left( r^2 \dfrac{\partial }{\partial r}  R(r) Y(\theta, \ \phi) \right) + \dfrac{1}{\sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial}{\partial \theta}  R(r) Y(\theta, \ \phi) \right) + \dfrac{1}{ \sin^2{\theta}}\dfrac{\partial^2}{\partial \phi^2}  R(r) Y(\theta, \ \phi) \tag{3} = 0
$$

提出式 $(3)$ 中的系数，

$$
Y(\theta, \ \phi) \dfrac{\mathrm{d}}{\mathrm{d} r} \left( r^2 \dfrac{\mathrm{d} R}{\mathrm{d} r} \right) + \dfrac{R(r)}{\sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial Y}{\partial \theta} \right) + \dfrac{R(r)}{ \sin^2{\theta}}\dfrac{\partial^2 Y}{\partial \phi^2} = 0 \tag{4}
$$

将式 $(4)$ 两侧同时除以 $R(r) Y(\theta, \ \phi)$，分离变量

$$
\dfrac{1}{R(r)} \dfrac{\mathrm{d}}{\mathrm{d} r} \left( r^2 \dfrac{\mathrm{d} R}{\mathrm{d} r} \right) + \dfrac{1}{Y(\theta, \ \phi)}\dfrac{1}{\sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial Y}{\partial \theta} \right) + \dfrac{1}{Y(\theta, \ \phi)} \dfrac{R(r)}{ \sin^2{\theta}}\dfrac{\partial^2 Y}{\partial \phi^2} = 0 \tag{5}
$$

式 $(5)$ 中可分离出两个分别含有参数 $r$ 的式 $(6)$ 和含有参数 $\theta, \phi$ 的式 $(7)$

$$
- \dfrac{1}{R(r)} \dfrac{\mathrm{d}}{\mathrm{d} r} \left( r^2 \dfrac{\mathrm{d} R}{\mathrm{d} r} \right) = -k \tag{6}
$$

$$
\dfrac{1}{Y(\theta, \ \phi)}\dfrac{1}{\sin{\theta}}\dfrac{\partial}{\partial \theta} \left( \sin{\theta} \dfrac{\partial Y}{\partial \theta} \right) + \dfrac{1}{Y(\theta, \ \phi)} \dfrac{R(r)}{ \sin^2{\theta}}\dfrac{\partial^2 Y}{\partial \phi^2} = -k \tag{7}
$$

## 2 求解径向方程

### 2.1 柯西-欧拉方程

这是一个二阶变系数常微分方程

$$
x^2 \dfrac{\mathrm{d}^2 y}{d x^2} + b x \dfrac{\mathrm{d}y}{\mathrm{d} x} + cy = 0 \tag{8}
$$

其中 $b$，$c$ 为常数.

观察可能的解，显然我们从多项式开始猜方程 $(9)$ 的一个特解.

令特解 $y=x^{\lambda}$，那么

$$
x^2 \left( \lambda \left( \lambda - 1 \right) x^{\lambda-2} \right) + b x \left( \lambda x^{\lambda-1} \right) + c x^\lambda = 0
$$

整理，得

$$
\lambda \left( \lambda - 1 \right) x^\lambda + b \lambda x^\lambda + cx^\lambda = 0
$$

因式分解，得

$$
\left( \lambda^2 + \left( b-1 \right)\lambda + c \right) x^\lambda = 0
$$

$$
\left\{
    \begin{array}{lr}
		x^\lambda = 0 \\
		\lambda^2 + \left( b-1 \right)\lambda + c = 0
    \end{array}
\right.
\tag{9}
$$

方程组 $(9)$ 中，当且仅当 $x = 0$ 时，$x^\lambda=0$ 成立；而对于特征方程 $\lambda^2 + \left( b-1 \right)\lambda + c = 0 $ 则可知它的两个解为

$$
\lambda_{1,\ 2} = \dfrac{1 - b  \pm \sqrt{(b-1)^2 - 4c}}{2}
$$

当 $\lambda_{1} \neq \lambda_{2}$ 时，方程 $(9)$ 的通解为 $y = C_1 x^{\lambda_1} + C_2 x^{\lambda_1}$，其中 $C_1$，$C_2$ 是常数。

当 $\lambda_{1} = \lambda_{2} = \dfrac{1-b}{2}$ 时，方程 $(9)$ 的一个特解为 $y=x^{\lambda} \ln x$，代入方程 $(9)$ 中得

$$
x^2 \dfrac{\mathrm{d}^2}{d x^2} \left( x^{\lambda} \ln x \right) + b x \dfrac{\mathrm{d}}{\mathrm{d} x} \left( x^{\lambda} \ln x \right) + c \left( x^{\lambda} \ln x \right) = 0 \tag{10}
$$

其中，

$$
\dfrac{\mathrm{d}}{d x} \left( x^{\lambda} \ln x \right) = \lambda x^{\lambda - 1} \ln x + x^{\lambda- 1} = (\lambda \ln x + 1) x^{\lambda - 1} \tag{11}
$$

$$
\dfrac{\mathrm{d}^2}{d x^2} \left( x^{\lambda} \ln x \right) = \dfrac{\mathrm{d}}{d x} \left( \lambda x^{\lambda - 1} \ln x + x^{\lambda- 1} \right) = \lambda \left(\lambda - 1\right) x^{\lambda- 2} \ln x + \lambda x^{\lambda - 2} + (\lambda - 1) x^{\lambda - 2} = \left(\lambda (\lambda - 1) \ln x + 2 \lambda - 1\right) x^{\lambda - 2} \tag{12}
$$

将式 $(10)$ 和 $(12)$ 代入到 $(10)$ 中，可得

$$
x^2 \left(\lambda (\lambda - 1) \ln x + 2 \lambda - 1\right) x^{\lambda - 2} + b x (\lambda \ln x + 1) x^{\lambda - 1}  + c \left( x^{\lambda} \ln x \right) = 0
$$

整理，并提出 $x^{\lambda}$，得

$$
x^{\lambda} \left[ \left(\lambda (\lambda - 1) \ln x + 2 \lambda - 1\right)  + b (\lambda \ln x + 1) + c \ln x \right]  = 0
$$

进一步整理方括号内的部分

$$
x^{\lambda} \left[ \ln x \left( {\lambda}^2 + (b - 1) \lambda + c \right) +2 \lambda + b - 1 \right]  = 0
$$

那么解为

$$
\left\{
    \begin{array}{lr}
		x^\lambda = 0 \\
		\ln x \left( {\lambda}^2 + (b - 1) \lambda + c \right) +2 \lambda + b - 1 = 0
    \end{array}
\right.
\tag{13}
$$

由方程组 (9) 可知，$\lambda^2 + \left( b-1 \right)\lambda + c = 0 $，因此方程组 $(13)$ 可化为

$$
2 \lambda + b - 1 = 0
$$

变形为

$$
\lambda = \dfrac{1-b}{2}
$$

恰好满足分类讨论的预设条件，因此 $y=x^{\lambda} \ln x$ 确实是方程组 $(9)$ 的特解.

此时方程 $(9)$ 的通解为

$$
y = C_1 x^{\lambda} + C_2 x^{\lambda} \ln x
$$

### 2.2 径向方程的通解

由径向方程 $(6)$ 可得

$$
\dfrac{\mathrm{d}}{\mathrm{d} r} \left( r^2 \dfrac{\mathrm{d} R}{\mathrm{d}r} \right) = k R(r)
$$

进一步展开求导得

$$
r^2 \dfrac{\mathrm{d}^2 R}{\mathrm{d} r^2} + 2r \dfrac{\mathrm{d} R}{\mathrm{d} r} - k R(r) = 0
$$

由柯西-欧拉方程的通解可知其特征方程的根为

$$
\lambda_{1, \ 2} = \dfrac{1 - 2 \pm \sqrt{(2-1)^2+4k}}{2}  = \dfrac{-1 \pm \sqrt{1+4k}}{2}
$$

根据韦达定理（二元一次方程的根与系数的关系）

$$
\lambda_1 + \lambda_2 = - \dfrac{2}{2} = -1
$$

移项得

$$
\lambda_1 = - \lambda_2 -1
$$

如果令 $l = \lambda_1$ 那么

$$
R(r) = C_1 r^l + C_2 r^{-(l + 1)}
$$

## 4 分解角向方程、方位方程

设 $ Y(\theta, \ \phi) = \Theta{\theta}\Phi(\phi) $，则有

$$
\dfrac{1}{\Theta{(\theta)}\Phi(\phi)} \dfrac{1}{\sin{\theta}} \dfrac{\partial}{\partial \theta} \left[ \sin{\theta} \dfrac{\partial}{\partial \theta} \left( \Theta{(\theta)}\Phi(\phi) \right) \right] + \dfrac{1}{\Theta{(\theta)}\Phi(\phi)} \dfrac{1}{\sin^2{\theta}} \dfrac{\partial^2}{\partial \theta^2} \left( \Theta{(\theta)}\Phi(\phi) \right) = -k
$$

观察上式，将偏导数中的部分无关因式作为常数提出并消去，这样化偏导数为导数

$$
\dfrac{1}{\Theta{(\theta)} \sin{\theta}} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \Theta{(\theta)} \right) + \dfrac{1}{\Phi(\phi) \sin^2(\theta)} \dfrac{\mathrm{d}^2 \Phi}{\mathrm{d} \phi^2} = -k
$$

将上式两侧同时乘 $\sin^2{\theta}$，得

$$
\dfrac{\sin{\theta}}{\Theta{(\theta)} } \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \Theta{(\theta)} \right) + \dfrac{1}{\Phi(\phi)} \dfrac{\mathrm{d}^2 \Phi}{\mathrm{d} \phi^2} = -k \sin^2{\theta}
$$

移项得

$$
\dfrac{\sin{\theta}}{\Theta{(\theta)} } \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \Theta{(\theta)} \right) + k \sin^2{\theta} = -\dfrac{1}{\Phi(\phi)} \dfrac{\mathrm{d}^2 \Phi}{\mathrm{d} \phi^2}
$$

这样将上式两端分别设为一个数 $M$，即

$$
\dfrac{\sin{\theta}}{\Theta{(\theta)} } \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \Theta{(\theta)} \right) + k \sin^2{\theta} = M
$$

$$
-\dfrac{1}{\Phi(\phi)} \dfrac{\mathrm{d}^2 \Phi}{\mathrm{d} \phi^2} = M
$$

其中上式叫做方位方程，整理后得

$$
\dfrac{\mathrm{d}^2 \Phi}{\mathrm{d} \phi^2} + M\Phi(\phi) = 0
$$

## 5 求解方位方程

### 5.1 二阶常系数齐次线性方程及其通解

形如

$$
\dfrac{\mathrm{d} y}{\mathrm{d} x} + a \dfrac{\mathrm{d} y}{\mathrm{d} x} + b y = 0
$$

设特征方程

$$
\lambda^2 + a \lambda + b = 0
$$

当 $\lambda_1 \neq \lambda_2 \in \mathbb{R}$ 时，方程的通解为 $y = C_1 e^{\lambda_1 x} + C_2 e^{\lambda_2 x}$.

当 $\lambda_1 = \lambda_2 \in \mathbb{R}$ 时，方程的通解为 $y = (C_1 + C_2 x) e^{\lambda_1 x}$.

当 $\lambda_{1, \ 2} = \alpha \pm i \beta \in \mathbb{C}$ 时，方程的通解为 $y = e^{\alpha x} \left(C_1 \cos{\beta x } + C_2 \sin{\beta x }\right)$.

根据欧拉公式 $e^{i x} = \cos{x} + i \sin{x}$，有

$$
\sin{x} = \dfrac{e^{ix} - e^{-i x}}{2i}
$$

$$
\cos{x} = \dfrac{e^{ix} + e^{-i x}}{2}
$$

那么通解可以用复数表示为

$$
y = e^{\alpha x} \left(C_1 \cos{\beta x } + C_2 \sin{\beta x }\right)
  = e^{\alpha x} \left(C_1 \dfrac{e^{i \beta x} + e^{-i \beta x}}{2} +  C_2 \dfrac{e^{i\beta x} - e^{-i \beta x}}{2i} \right)
$$

将常数吸收，通解进一步化简为

$$
y = e^{\alpha x} \left(C_1 e^{i \beta x}  + C_2 e^{ - i \beta x} \right)
  = C_1 e^{(\alpha + i \beta) x } + C_2 e^{(\alpha - i \beta) x }
$$

### 5.2 方位方程的通解

根据角向方程的特征方程

$$
\lambda^2 + M = 0
$$

解得特征根

$$
\lambda_{1, \ 2} = \pm \sqrt{M} i
$$

根据二阶常系数齐次线性方程的通解可知 $ \alpha = 0 $，$\beta = \sqrt{M}$.

这样角向方程的通解为

$$
\Phi(\phi) = C_1 e^{i \sqrt{M} \phi} + C_2 e^{- i \sqrt{M} \phi}
$$

其中 $C_1$，$C_2$ 依赖于 $M$.

## 6 求解角向方程

### 6.1 连带 Legendre 方程

令 $M = m^2$，角向方程化为

$$
\dfrac{\sin{\theta}}{\Theta(\theta)} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} \right) + k \sin^2{\theta } = m^2
$$

方程两侧乘 $\Theta(\theta)$，移项整理得

$$
\sin{\theta} \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \sin{\theta} \dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} \right) + \left(k \sin^2{\theta } - m^2 \right) \Theta(\theta) = 0
$$

展开求导，得

$$
\sin^2{\theta} \dfrac{\mathrm{d}^2 \Theta}{\mathrm{d} \theta^2} + \sin{\theta} \cos{\theta} \dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} + \left(k \sin^2{\theta } - m^2 \right) \Theta(\theta) = 0
$$

方程两侧除以 $\sin^2(\theta)$，移项整理得

$$
\dfrac{\mathrm{d}^2 \Theta}{\mathrm{d} \theta^2} + \dfrac{\cos{\theta}}{\sin{\theta}} \dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} + \left(k  - \dfrac{m^2}{\sin^2{\theta}} \right) \Theta(\theta) = 0
$$

为了代换方程中存在的三角函数，设 $\Theta(\theta) = P(\cos\theta)$，令 $x=\cos{\theta}$，那么 $\Theta(\theta) = P(x)$.

因为

$$
\dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} = \dfrac{\mathrm{d} P}{\mathrm{d} x} \dfrac{\mathrm{d} x}{\mathrm{d} \theta} = - \dfrac{\mathrm{d} P}{\mathrm{d} x} \sin{\theta}
$$

$$
\dfrac{\mathrm{d}^2 \Theta}{\mathrm{d} \theta^2}
= \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \dfrac{\mathrm{d} \Theta}{\mathrm{d} \theta} \right)
= - \dfrac{\mathrm{d}}{\mathrm{d} \theta} \left( \dfrac{\mathrm{d} P}{\mathrm{d} x}  \right) \sin{\theta} - \dfrac{\mathrm{d} P}{\mathrm{d} x}  \cos{\theta}
= \sin^2{\theta}  \dfrac{\mathrm{d^2} P}{\mathrm{d} x^2}  - \cos{\theta} \dfrac{\mathrm{d} P}{\mathrm{d} x}
$$

所以代入到角向方程后得

$$
\sin^2{\theta}  \dfrac{\mathrm{d^2} P}{\mathrm{d} x^2}  - \cos{\theta} \dfrac{\mathrm{d} P}{\mathrm{d} x} + \dfrac{\cos{\theta}}{\sin{\theta}} \left( - \dfrac{\mathrm{d} P}{\mathrm{d} x} \sin{\theta} \right) + \left(k  - \dfrac{m^2}{\sin^2{\theta}} \right) P(x) = 0
$$

整理得

$$
\sin^2{\theta}  \dfrac{\mathrm{d^2} P}{\mathrm{d} x^2}  - 2\cos{\theta} \dfrac{\mathrm{d} P}{\mathrm{d} x} +  \left(k  - \dfrac{m^2}{\sin^2{\theta}} \right) P(x) = 0
$$

又因为 $x = \cos{\theta}$，$\sin^2{\theta} + \cos^2{\theta} = \sin^2{\theta} + x^2 = 1 $，所以 $\sin^2{\theta} = 1 - x^2$.

代换掉三角函数，得

$$
\left( 1 - x^2\right) \dfrac{\mathrm{d^2} P}{\mathrm{d} x^2} - 2x \dfrac{\mathrm{d} P}{\mathrm{d} x} + \left(k  - \dfrac{m^2}{1 - x^2} \right) P(x) = 0
$$

这个方程被称为**连带 Legendre 方程**.

如果令 $m=0$，即轴对称时，则得到 **Legendre 方程**

$$
\left( 1 - x^2\right) \dfrac{\mathrm{d^2} P}{\mathrm{d} x^2} - 2x \dfrac{\mathrm{d} P}{\mathrm{d} x} + k P(x) = 0
$$

### \*6.2 Legendre 多项式

我们将 $P(x)$ 视为一个无穷级数，即

$$
P(x) = \sum_{n=0}^{\infty} a_n x^n
$$

那么它的一阶导数

$$
\dfrac{\mathrm{d} P}{\mathrm{d} x} = \sum_{n=0}^{\infty} n a_n x^{n-1}
$$

二阶导数

$$
\dfrac{\mathrm{d}^2 P}{\mathrm{d} x^2} = \sum_{n=0}^{\infty} n (n-1) a_n x^{n-2}
$$

将其代入三角代换后的角向方程，并假设 $m = 0$，得

$$
\left( 1 - x^2\right) \sum_{n=0}^{\infty} n (n-1) a_n x^{n-2} - 2x \sum_{n=0}^{\infty} n a_n x^{n-1} + k \sum_{n=0}^{\infty} a_n x^n = 0
$$

$$
\sum_{n=0}^{\infty} n (n-1) a_n x^{n-2}  - \sum_{n=0}^{\infty} n (n-1) a_n x^{n} - 2 \sum_{n=0}^{\infty} n a_n x^{n} + k \sum_{n=0}^{\infty} a_n x^n = 0
$$

以 $n \leftarrow n + 2$，变更下标

$$
\sum_{n=-2}^{\infty} (n+2) (n+1) a_{n+2} x^{n} - \sum_{n=0}^{\infty} n (n-1) a_n x^{n} - 2 \sum_{n=0}^{\infty} n a_n x^{n} + k \sum_{n=0}^{\infty} a_n x^n = 0
$$

由于 $n$ 可以取到无穷大，所以下标的变换等价于

$$
\sum_{n=0}^{\infty} (n+2) (n+1) a_{n+2} x^{n} - \sum_{n=0}^{\infty} n (n-1) a_n x^{n} - 2 \sum_{n=0}^{\infty} n a_n x^{n} + k \sum_{n=0}^{\infty} a_n x^n = 0
$$

提出 $x^n$，从而

$$
\sum_{n=0}^{\infty} \left[ (n+2) (n+1) a_{n+2} - n(n-1) a_n - 2n a_n + k a_n \right] x^n
$$

整理括号内的式子

$$
\sum_{n=0}^{\infty} \left[ (n+2) (n+1) a_{n+2} - \left( n(n+1) -k \right) a_n \right] x^n
$$

若上式成立，那么就有

$$
(n+2) (n+1) a_{n+2} - \left( n(n+1) -k \right) a_n = 0
$$

整理上式得到如下的递推公式

$$
a_{n+2} = \dfrac{n(n+1) -k}{(n+2) (n+1)} a_n
$$

由此可知，若 $a_n = 0$，那么 $a_{n+2} = 0$，从而使得对于任意的 $k \in \mathbb{N}$，有 $a_{n+2k} = 0$.

又因为 $x = \cos{\theta} \in [-1, \ 1]$，当 $x=\pm 1 $ 时，这个级数是**发散**的，除非满足以下条件：

从 $a_0 = 0$ 或 $a_1 = 0$ 开始，存在正整数 $l$，使得 $a_l \neq 0$，$a_{l+2} = 0$.

$$
a_{l+2} = 0 = \dfrac{l(l+1) -k}{(l+2) (l+1)} a_l
$$

解得 $k = l(l+1)$，那么

$$
a_{n+2} = \dfrac{n(n+1) - l(l+1)}{(n+2) (n+1)} a_n
$$

如果 $l$ 是偶数，此时令所有奇数项为 $0$，那么这个级数是**收敛**的；如果 $l$ 是奇数，此时令所有偶数项为 $0$，那么这个级数也是**收敛**的.

在级数收敛的条件下，有

$$
P(x) = \sum^{\infty}_{l=0} C_l P_l (x)
$$

这里 $P_l( x)$ 称为**勒让德多项式**，$C_l$ 是常数.

## 7 Laplace 方程的通解

令 $m$ 表示阶数，那么 Laplace 的方程通解可表示为

$$
\varphi(r, \ \theta, \ \phi) = \sum_{l=0}^{\infty} \sum_{m=-\infty}^{\infty} \varphi_l^m(r, \ \theta, \ \phi)
$$

其中

$$
\varphi_l^m(r, \ \theta, \ \phi) = R_l^m(r) \Theta_l^m(\theta) \Phi^m(\phi)
$$

如果代入径向方程、方位方程和角向方程的通解，那么可以得到

$$
\varphi_l^m(r, \ \theta, \ \phi) = \left( A_{lm} r^l + B_{lm} r^{-(l+1)} \right) \cdot P_l^m( \cos{\theta}) \cdot e^{im \phi}
$$

> **问**：为什么 $\Phi(\phi) = C_1 e^{i \sqrt{M} \phi} + C_2 e^{- i \sqrt{M} \phi}$ 在这里会化为 $e^{im \phi}$？
>
> **答**：$e^{i \sqrt{M} \phi}$ 的轨迹在复平面上的一个半径为 1 的单位圆，所以 $ C_1 e^{i \sqrt{M} \phi} $ 的轨迹是一个半径为 $C_1$ 的圆，同理 $C_2 e^{- i \sqrt{M} \phi}$ 的轨迹是一个半径为 $C_2$ 的圆，这两个圆上的任意一点与复平面原点形成的两个向量，如果将它们加和在一起，等价于一个在常量 $C_3$ 半径下的新圆上的向量，其中常量 $C_3$ 依赖于 $C_1$，$C_2$. 因此可以认为，$C_1 e^{i \sqrt{M} \phi} + C_2 e^{- i \sqrt{M} \phi} = C_3 e^{im \phi}$，而在通解公式中，这个 $C_3$ 可以被前面的项吸收.

我们令 $\theta = 0$，这意味着我们的位矢指向球坐标系的顶点.

当 $m = 0$ 时，

$$
\varphi^0_l (r, 0, \phi) = \left( A_{l0} r^l + B_{l0} r^{-(l+1)} \right) \cdot P_l^0( \cos{0}) \cdot e^{0}
= \left( A_{l0} r^l + B_{l0} r^{-(l+1)} \right)
$$

当 $m \neq 0$ 时，

$$
\varphi_l^m(r, \ 0, \ \phi) = \left( A_{lm} r^l + B_{lm} r^{-(l+1)} \right) \cdot P_l^m(1) \cdot e^{im \phi}
$$

然而，在球坐标系下，点 $(R, 0, A)$ 点 $(R, 0, B)$ 实际上是同一点，但是上面两个式子的结果却不同.

> 你可以想象你正在仰头看着顶点（$\theta = 0$），保持头的朝向不变，无论你如何原地转动自己的身体（$\phi$），你的视线始终在同一个顶点上.

如果想要上面两个式子相等，意味着 $m = 0$，或者 $\Theta_l^m (0) = P_l^m(1) = 0$.

令 $\theta = \pi$，这意味着我们的位矢指向球坐标系的底点，同理可知 $\Theta_l^m (0) = P_l^m(\pi) = 0$.

为了满足上述条件，将 $\Theta_l^m(\theta)$ 解的形式表示为

$$
\Theta_l^m(\theta) = \sin^m(\theta) g_l^m (\theta)
$$

回想起之前我们的代换，

$$
{x=\cos\theta}
$$

这样 $\Theta_l^m(\theta)$ 的解中的因式 $g_l^m (\theta)$ 又可表示为

$$
{{g_{l}^{m}(\theta)=f_{l}^{n}(x)}}
$$

$$
\Theta_l^m(\theta) = (1 - \cos^2{\theta})^{m/2} g_l^m (\theta) = (1 - x^2)^{m/2} f_l^m (x)
$$

令 ${n={\dfrac{m}{2}}}$，则

$$
\Theta_l^m(\theta) = (1 - x^2)^{n} f_l^m (x)
$$

对 $P_l^m$ 求一阶和二阶导数

$$
\frac{\mathrm{d} P_{l}^{m}}{\mathrm{d} x} = n\left(1-x^{2}\right)^{n-1}\left(-2x\right)f_{l}^{m}(x)+\left(1-x^{2}\right)^{n}\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}
$$

$$
{\frac{\mathrm{d}^{2}P_{l}^{m}}{\mathrm{d} x^{2}}}
=\left(1-x^{2}\right)^{n}{\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}}-4n x\left(1-x^{2}\right)^{n-1}{\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}} + \left(4(n-1)n x^{2}\left(1-x^{2}\right)^{n-1}-2n\left(1-x^{2}\right)^{n}\right)f_{l}^{m}(x)
$$

整理上式，得

$$
\begin{align*}
\left(1-x^{2}\right)\frac{\mathrm{d}^{2}P_{l}^{m}}{\mathrm{d} x^{2}}
&= \left(1-x^{2}\right)^{n+1}{\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}}-4n x\left(1-x^{2}\right)^{n}{\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}} + \left(4(n-1)n x^{2}\left(1-x^{2}\right)^{n-1}-2n\left(1-x^{2}\right)^{n}\right)f_{l}^{m}(x) \\
&= \left(1-x^{2}\right)^{n-1}\left(\left(1-x^{2}\right)^{2}{\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}}-4n x\left(1-x^{2}\right){\frac{\mathrm{d} f_{l}^{n}}{\mathrm{d} x}} + \left(4(n-1)n x^{2}-2n\left(1-x^{2}\right)\right)f_{r}^{m}(x) \right)
\end{align*}
$$

$$
- 2x\frac{\mathrm{d} P_{l}^{m}}{\mathrm{d} x}=\left(1-x^{2}\right)^{n-1}\left(4n x^{2}f_{l}^{m}(x)-2x\left(1-x^{2}\right)\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}\right)
$$

将 $P_l^m$ 变形为

$$
\left(l(l+1)-\frac{m^{2}}{1-x^{2}}\right)P_{l}^{m}(x)=\left(1-x^{2}\right)^{n-1}\left(l(l+1)\left(1-x^{2}\right)-m^{2}\right)f_{l}^{m}(x)
$$

代入连带 Legendre 方程

$$
\left(1-x^{2}\right)\frac{\mathrm{d}^{2}P_{l}^{m}}{\mathrm{d} x^{2}}-2x\frac{\mathrm{d} P_{l}^{m}}{\mathrm{d} x}+\left(l(l+1)-\frac{m^{2}}{1-x^{2}}\right)P_{l}^{m}(x) = 0
$$

得

$$
\left(1-x^{2}\right)^{n-1}\left(\left(1-x^{2}\right)^{2}{\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}}-4n x\left(1-x^{2}\right){\frac{\mathrm{d} f_{l}^{n}}{\mathrm{d} x}} + \left(4(n-1)n x^{2}-2n\left(1-x^{2}\right)\right)f_{r}^{m}(x) \right) \\ +
\left(1-x^{2}\right)^{n-1}\left(4n x^{2}f_{l}^{m}(x)-2x\left(1-x^{2}\right)\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}\right) + \left(1-x^{2}\right)^{n-1}\left(4n x^{2}f_{l}^{m}(x)-2x\left(1-x^{2}\right)\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}\right) = 0
$$

提取因式整理得

$$
\left(1-x^2 \right)^{n-1} \left[
\textcolor{green}{\left(1-x^2 \right)^{2}} \frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}
 \textcolor{red}{-4nx  \left(1-x^2 \right)} \frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x} \\
+ \textcolor{blue}{\left(4(n-1)n x^{2}-2n\left(1-x^{2}\right) \right)} f_{l}^{n}(x)  \\
+ \textcolor{blue}{4 n x^{2}}f_{l}^{m}(x) \\
 \textcolor{red}{-2x\left(1-x^{2}\right)}{\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}}
+ \textcolor{blue}{\left( l(l+1) \left(1-x^2\right) - m^2\right)} f_l^m(x)
\right] = 0
$$

按照绿色、蓝色、红色整理系数，得

$$
\left(1-x^2 \right)^{n-1} \left[
\textcolor{green}{\left(1-x^2 \right)^{2}} \frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}
- \textcolor{red}{2x  \left(1-x^2 \right) (2n+1)} \frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x} \\
+ \textcolor{blue}{\left( \left(4(n-1)n x^{2}-2n\left(1-x^{2}\right) \right) + \left( l(l+1) \left(1-x^2\right) - m^2\right) \right)} f_{l}^{n}(x)  \\
\right] = 0
$$

进一步整理蓝色系数，用 ${n={\dfrac{m}{2}}}$ 代换，则

$$
\begin{align*}
& \textcolor{blue}{4(n-1)n x^{2}-2n\left(1-x^{2}\right) + l(l+1) \left(1-x^2\right) - m^2} \\
&= 4 n^2 x^2 -2n \left( 1- x^2 \right) + l (l+1) \left(1-x^2 \right) - m^2 \\
&= m^2 x^2 -m (1-x^2) + l(l+1)(1-x^2) - m^2 \\
&= m^2(x^2 - 1) - m(1-x^2) + l(l+1)(1-x^2) \\
&= (1-x^2)m(1+m) + l(l+1)(1-x^2) \\
&= \left(1-x^2\right) \left( l(l+1) - m(m+1) \right)
\end{align*}
$$

提出 $(1-x^2)$，连带 Legendre 方程最终化为

$$
\left(1-x^2 \right)^{n} \left[
\left(1-x^2 \right) \frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}
-2(m+1)x \frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x} \\
+ \left( l(l+1) -m(m+1) \right) f_{l}^{n}(x)  \\
\right] = 0
$$

解这个方程，我们知道 $x=\pm 1$ 时方程成立，而另一个因式成立时，即

$$
\left(1-x^2 \right) \frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}
-2(m+1)x \frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}
+ \left( l(l+1) -m(m+1) \right) f_{l}^{n}(x)
= 0
$$

此时令 $m=0$，得到 Legendre 方程，那么其求导

$$
\frac{\mathrm{d}}{\mathrm{d} x}\left((1-x^{2})\,\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}-2(m+1)x\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}+\left(l(l+1)-m(m+1)\right)f_{l}^{m}(x)\right)=0
$$

$$
(1-x^{2})\,\frac{\mathrm{d}^{3}f_{l}^{m}}{\mathrm{d} x^{3}}-2x\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}-2(m+1)\frac{\mathrm{d}_{l}^{m}}{\mathrm{d} x}-2(m+1)x\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}+\left(l(l+1)-m(m+1)\right)\frac{\mathrm{d} f_{l}^{m}}{\mathrm{d} x}=0
$$

$$
\bigl(1-x^{2}\bigr)\,\frac{\mathrm{d}^{3}f_{l}^{m}}{\mathrm{d} x^{3}}
 -2(m+2)\,x\,\frac{\mathrm{d}^{2}f_{l}^{m}}{\mathrm{d} x^{2}}
 -\bigl(l(l+1)-(m+1)(m+2)\bigr)\frac{\mathrm{d}f_{l}^{m}}{\mathrm{d} x}=0
$$

对比求导前和求导后的式子，不难发现这样的规律，即

$$
\dfrac{\mathrm{d} f_l^m}{\mathrm{d} x} = f_l^{m+1}(x)
$$

又 $f^0_l(x) = P_l (x)$，代入 $P_l^m(x)$ 的解的表达式可得

$$
P_l^m(x) = \left( 1 - x^2 \right)^{m/2} \dfrac{\mathrm{d}^m P_l}{\mathrm{d} x^m}
$$

这些函数被称为**连带勒让德多项式**. 因为 Legendre 多项式的阶数为 $l$，意味着对其最多求导 $l$ 次，即 $m≤l$.

至此，我们求得了 Laplace 方程的通解为

$$
\varphi(r, \ \theta, \ \phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} \left( A_{lm} r^l + B_{lm} r^{-(l+1)} \right) \cdot P_l^m( \cos{\theta}) \cdot e^{im \phi}
$$

## 8 球谐函数

由 Laplace 方程的通解定义**球谐函数**

$$
Y_{lm}(1, \theta, \phi) = C_{lm} \cdot P_l^m (\cos{\theta}) \cdot e^{im\phi}
$$

其中，$l \geq 0$，$-l \leq m \leq l$，$C_{lm}$ 是归一化系数，满足

$$
\int \left| Y_{lm} (\theta, \phi) \right|^2 \mathrm{d} \Omega = 1
$$

> 参考资料：
>
> https://wuli.wiki/online/SphHar.html
>
> https://zhuanlan.zhihu.com/p/488242089
