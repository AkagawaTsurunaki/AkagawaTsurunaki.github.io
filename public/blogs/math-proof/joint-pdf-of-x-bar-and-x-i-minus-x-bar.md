# $\bar{X}$ 与 $X_i - \bar{X}$ 的联合概率密度函数

设 $X_1, X_2, \dots , X_n$ 为来自总体 $X$ 的简单随机样本，$X \sim N(\mu, \sigma^2)$，求出 $\bar{X}$ 与 $X_i - \bar{X}$ 的联合概率密度函数. 

---

对于两个服从正态分布的样本 $\bar{X}, X_i - \bar{X}$，协方差 $\mathrm{Cov}(\bar{X}, X_i - \bar{X})$ 为 $0$ 时即二者独立. 

> 注意，对于服从其他分布的两个随机变量，其协方差为 0 不一定独立，只能证明没有线性关系，它们可能独立，也可能存在某种非线性关系. 

首先，证明 $\mathrm{Cov}(\bar{X}, X_i - \bar{X}) = 0$
$$
\mathrm{Cov}(\bar{X}, X_i - \bar{X}) = E \left(\bar{X} (X_i - \bar{X}) \right) - E(\bar{X}) E(X_i - \bar{X}) \tag{1}
$$
注意到式 $(1)$ 中的最后一项，有

$$
\begin{align*}

E(X_i - \bar{X}) &= E(X_i) - E(\bar{X}) 
\\ &= E(X_i) - E(\frac{1}{n}\sum_i^n X_i) 
\\ &= E(X_i) - \frac{1}{n}E(\sum_i^n X_i) 
\\ &= E(X_i) - \frac{1}{n}\sum_i^n E( X_i) 
\\ &= \mu - \frac{1}{n} n \mu 
\\ &= 0

\end{align*}
$$
从而简化协方差的式子为
$$
\mathrm{Cov}(\bar{X}, X_i - \bar{X}) = E \left(\bar{X} (X_i - \bar{X}) \right) \tag{2}
$$
这样只需证明 $E \left(\bar{X} (X_i - \bar{X}) \right) = 0$ 即可. 
$$
E \left(\bar{X} (X_i - \bar{X}) \right) = E \left(\bar{X} X_i - \bar{X}^2 \right) = E(\bar{X} X_i) - E( \bar{X}^2 ) \tag{3}
$$
在式 $(3)$ 中，我们先计算 $E(\bar{X} X_i)$，
$$
\begin{align*}

E(\bar{X} X_i) &= E \left[ \left(\frac{1}{n}\sum_i^n X_i\right) \cdot X_i \right] 
\\ &= \frac{1}{n} E \left[ \left(\sum_i^n X_i\right) \cdot X_i \right] 
\\ &= \frac{1}{n} E \left[ X_1X_i + X_2 X_i + \cdots +X_i^2 + \cdots X_n X_i \right] 
\\ &= \frac{1}{n} \left[ E(X_1X_i) + E(X_2 X_i) + \cdots + E(X_i^2) + \cdots E(X_n X_i) \right]

\end{align*} \tag{4}
$$
当 $i \neq j $ 时，$X_i$ 与 $X_j$ 独立，那么 $E(X_i X_j) = E(X_i) E(X_j) = \mu^2$，注意到式 $(4)$ 中除了 $E(X_i^2)$ 外一共有 $n-1$ 个 $E(X_i X_j)$
$$
E(\bar{X} X_i) = \frac{1}{n} \left[ (n-1)\mu^2 + E(X_i^2) \right] \tag{5}
$$
再计算 $X_i$ 的二阶矩，它可以通过方差与期望间接求取，已知 $D(X_i) = \sigma^2$
$$
E(X_i^2) = D(X_i) + (E(X_i))^2 = \sigma^2 + \mu^2 \tag{6}
$$
将式 $(6)$ 代入 $(5)$，可得
$$
E(\bar{X} X_i) =  \frac{1}{n} \left[ (n-1)\mu + \sigma^2 + \mu^2 \right] = \mu^2 + \frac{\sigma^2}{n} \tag{7}
$$
同理，
$$
E(\bar{X}^2) = D(\bar{X}) + (E(\bar{X}))^2 = \frac{\sigma^2}{n} + \mu^2 \tag{8}
$$
将式 $(7)$ 和 $(8)$ 代入式 $(3)$ 得
$$
E(\bar{X} X_i) - E( \bar{X}^2 ) = 0 \tag{9}
$$
这样我们证明了 $\mathrm{Cov}(\bar{X}, X_i - \bar{X}) = 0$，即 $\bar{X}, X_i - \bar{X}$ 独立. 

接着，求出 $X_i - \bar{X}$ 的方差
$$
\begin{align*}

D(X_i - \bar{X}) &= D \left(-\frac{1}{n} X_1 -\frac{1}{n} X_2 + \cdots + \left(1-\frac{1}{n}\right) X_i + \cdots -\frac{1}{n} X_n \right)
\\ &= \dfrac{1}{n^2} D (X_1) + \dfrac{1}{n^2} D (X_2) + \cdots + \left(1-\frac{1}{n}\right)^2 D(X_i) + \cdots + \dfrac{1}{n^2} D (X_n)
\\ &= \dfrac{1}{n^2} (n-1)\sigma^2 + \left(1-\frac{1}{n}\right)^2 \sigma^2
\\ &= \dfrac{n-1}{n} \sigma^2

\end{align*}
$$

因此 $X_i - \bar{X} \sim N(0,  \frac{n-1}{n} \sigma^2)$. 

不难证明，$\bar{X} \sim N(\mu, \frac{\sigma^2}{n})$. 

因此，$\bar{X}, X_i - \bar{X}$ 的联合分布函数就是各自的概率密度函数的乘积. 
$$
f_{\bar{X}}(\bar{x}) = \dfrac{1}{\sqrt{2 \pi \frac{\sigma^2}{n} }} \exp \left( {- \frac{ (\bar{x} - \mu)^2 }{2 \frac{\sigma^2}{n}} } \right)
$$

$$
f_{X_i - \bar{X}}(x_i - \bar{x}) = \dfrac{1}{\sqrt{2 \pi \frac{n-1}{n} \sigma^2 }} \exp \left( {- \frac{ (x_i - \bar{x})^2 }{2 \frac{n-1}{n} \sigma^2 } } \right)
$$

$$
\begin{align*}

f(\bar{X}, X_i - \bar{X}) &= f_{\bar{X}}(\bar{x}) \cdot f_{X_i - \bar{X}}(x_i - \bar{x})
\\ &= \dfrac{n}{2\pi \sqrt{n-1} \sigma^2} \exp \left[ - \dfrac{n (\bar{x} - \mu)^2}{2 \sigma^2} - \dfrac{n(x_i - \bar{x})^2 }{2(n-1) \sigma^2} \right]

\end{align*}
$$

> 用线性代数的做法可能会更简单，这里只是给出最一般的、较为全面的推导. 
