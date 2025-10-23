# Scaled Dot-Product Attention 的公式中为什么要除以 $\sqrt{d_k}$？

在学习 Scaled Dot-Product Attention 的过程中，遇到了如下公式

$$
\mathrm{Attention} (\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \mathrm{softmax} \left( \dfrac{\boldsymbol{Q} \boldsymbol{K}}{\sqrt{d_k}} \right) \boldsymbol{V}
$$

不禁产生疑问，其中的 $\sqrt{d_k}$ 为什么是这个数，而不是 $d_k$ 或者其它的什么值呢？

[Attention Is All You Need](https://paperswithcode.com/paper/attention-is-all-you-need) 中有一段解释

> We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\sqrt{d_k}$.

这说明，两个向量的点积可能很大，导致 softmax 函数的梯度太小，因此需要除以一个因子，但是为什么是 $\sqrt{d_k}$ 呢？

文章中的一行注释提及到

> To illustrate why the dot products get large, assume that the components of $\boldsymbol{q}$ and $\boldsymbol{k}$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $\boldsymbol{q} \cdot \boldsymbol{k} = \sum_{i=1}^{d_k} q_i k_i $ has mean $0$ and variance $d_k$.

本期，我们将基于上文的思路进行完整的推导，以证明 $\sqrt{d_k}$ 的在其中的作用.

## 基本假设

假设独立随机变量 $U_1 ,\ U_2 ,\ \dots ,\ U_{d_k}$ 和独立随机变量 $V_1 ,\ V_2 ,\ \dots ,\ V_{d_k}$ 分别服从期望为 $0$，方差为 $1$ 的分布，即
$$
 E \left(U_i \right) = 0 ,\ \mathrm{Var} \left(U_i \right) = 1 
$$

$$
E \left(V_i \right) = 0 ,\ \mathrm{Var} \left(V_i \right) = 1
$$

其中 $i = 1, 2, \dots ,\ d_k$，$d_k$ 是个常数. 

## 计算 $U_i V_i $ 的方差

由随机变量方差的定义可得 $ U_i V_i $ 的方差为

$$
\begin{align*}
    \mathrm{Var} \left( U_i V_i \right) &= E \left[ \left( U_i V_i - E \left( U_i V_i \right)  \right)^2\right] \\
                                        &= E \left[ \left(U_i V_i \right)^2 - 2U_i V_i E \left( U_i V_i \right) + E^2 \left( U_i V_i \right)\right] \\
    &= E \left[ \left( U_i V_i \right)^2 \right] - 2 E \left[ U_i V_i E \left( U_i V_i \right) \right] + E^2 \left(U_i V_i\right) \\
    &= E \left( U_i^2 V_i^2 \right) - 2 E \left( U_i V_i \right) E \left( U_i V_i \right) + E^2 \left(U_i V_i\right) \\
    &= E \left( U_i^2 V_i^2 \right) - E^2 \left( U_i V_i \right)
\end{align*}
$$

因为 $U_i$ 和 $V_i$ 是独立的随机变量，所以

$$
E \left( U_i V_i \right) = E \left( U_i \right) E \left( V_i \right)
$$

从而

$$
\begin{align*}
    \mathrm{Var} \left( U_i V_i \right) &= E\left(U_i^2\right) E\left(V_i^2\right) - \left(E\left(U_i\right) E\left(V_i\right) \right)^2 \\
    &= E\left(U_i^2\right) E\left(V_i^2\right) - E^2\left(U_i\right) E^2\left(V_i\right)
\end{align*}
$$

又因为 $E(U_i) = E(V_i) = 0$，所以

$$
\mathrm{Var} \left( U_i V_i \right) = E(U_i^2) E(V_i^2)
$$

## 计算 $E(U_i^2)$

因为
$$
    E \left( U_i \right) = 0
$$
$$
\mathrm{Var} \left( U_i \right) = 1
$$
$$
\mathrm{Var} \left( U_i \right) = E \left( U_i^2 \right) - E^2 \left( U_i \right)
$$
所以
$$
E(U_i^2) = 1
$$

同理，

$$
E(V_i^2) = 1
$$

## 计算 $\boldsymbol{q} \boldsymbol{k}$ 的方差

如果 $\boldsymbol{q} = \left[U_1, U_2, \cdots, U_{d_k} \right]^T$，$\boldsymbol{k} = \left[V_1, V_2, \cdots, V_{d_k} \right]^T$，那么

$$
\boldsymbol{q} \mathbf{k} = \sum_{i=1}^{d_k} U_i V_i
$$

$\boldsymbol{q} \boldsymbol{k}$ 的方差

$$
\begin{align*}
    \mathrm{Var}\left(  \boldsymbol{q} \mathbf{k}  \right)
    &= \mathrm{Var}\left( \sum_{i=1}^{d_k} U_i V_i \right)  \\
    &= \sum_{i=1}^{d_k} \mathrm{Var} \left( U_i V_i \right) \\
    &= \sum_{i=1}^{d_k} E \left(U_i^2\right) E \left(V_i^2\right) \\
    &= \sum_{i=1}^{d_k} 1 \cdot 1 \\
    &= d_k
\end{align*}
$$

到这里就可以解释为什么在最后要除以 $\sqrt{d_k}$，因为

$$
\begin{align*}
    \mathrm{Var}\left( \dfrac{\boldsymbol{q} \boldsymbol{k} }{\sqrt{d_k}} \right) &= \dfrac{\mathrm{Var}\left(  \boldsymbol{q} \boldsymbol{k}  \right)}{d_k} \\
    &= \dfrac{d_k}{d_k} \\
    &= 1
\end{align*}
$$

可见这个因子的目的是让 $\boldsymbol{q} \boldsymbol{k}$ 的分布也归一化到期望为 $0$，方差为 $1$ 的分布中，增强机器学习的稳定性. 


> 参考文献/资料
> - [Attention Is All You Need](https://paperswithcode.com/paper/attention-is-all-you-need)