# 离散情况下的贝叶斯估计量求解

设总体 $X \sim B(N, P)$，$N$ 已知，$X_1, X_2, \dots , X_n$ 为 $X$ 的样本，如果 $p$ 具有先验分布律 $P \{p=0.2 \}=0.8$，$P \{p=0.9 \}=0.2$，求 $p$ 的贝叶斯估计量. 

---


根据贝叶斯公式
$$
P(p_i |X_1, X_2, \dots , X_n ) = \dfrac{P(X_1, X_2, \dots , X_n | p_i) P(p_i)}{P(X_1, X_2, \dots , X_n)}
$$

首先，由于 $X_1, X_2, \dots, X_n$ 独立同分布，写出样本 $X$ 与参数 $p$ 的联合分布
$$
P(X_1, X_2, \dots , X_n | p_i) P(p_i) = P(X_1|p_i) P(X_2|p_i) \cdot \cdots  \cdot P(X_n|p_i)  P(p_i) 
\\ =
\left( \prod_{j=1}^{n} \binom{N}{x_j} p_i^{x_j} \left( 1- p_i \right)^{N - x_j} \right) \cdot \left( 0.8^{\mathbb{1}(p_i=0.2)} 0.2^{\mathbb{1} (p_i=0.9) } \right)
$$
> 这里用了指示函数 $\mathbb{1}(\cdot)$，你也可以分开写. 

接着，对上式进行整理，保留带参数 $p$ 的部分，其它部分吸收记为常数 $C$
$$
C \left(  p_i^{\sum_{j=1}^{n} x_j} {\left( 1- p_i \right)}^{nN - \sum_{j=1}^{n} x_j} \right) \cdot \left( 0.8^{\mathbb{1}(p_i=0.2)} 0.2^{\mathbb{1} (p_i=0.9) } \right)
$$
因为 $\bar{X} = \sum_{j=1}^{n} X_j $，所以上式化为
$$
P(X_1, X_2, \dots , X_n | p_i) P(p_i) = C \left(  p_i^{n \bar{x}} {\left( 1- p_i \right)}^{nN - n \bar{x}} \right) \cdot \left( 0.8^{\mathbb{1}(p_i=0.2)} 0.2^{\mathbb{1} (p_i=0.9) } \right)
$$
那么由

$$
P(p_i |X_1, X_2, \dots , X_n ) \propto P(X_1, X_2, \dots , X_n | p_i) P(p_i)
$$

可知
$$
P(p_i |X_1, X_2, \dots , X_n ) \propto \left(  p_i^{n \bar{x}} {\left( 1- p_i \right)}^{nN - n \bar{x}} \right) \cdot \left( 0.8^{\mathbb{1}(p_i=0.2)} 0.2^{\mathbb{1} (p_i=0.9) } \right)
$$
具体地，因为参数 $p_i$ 可以取 $0.2$ 或 $0.9$，所以
$$
P(p_i =0.2 |X_1, X_2, \dots , X_n ) \propto \left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right)
$$

$$
P(p_i =0.9 |X_1, X_2, \dots , X_n ) \propto \left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)
$$

这样，我们记上面两个式子和为 $\Delta$，即
$$
\Delta = P(X_1, X_2, \dots , X_n) = C \left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right) + C \left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)
$$

> 写正比于式子的时候可以忽略常数 $C$，但是写等式的时候别忘了（虽然最后约掉了）. 

这样带入贝叶斯公式，就有
$$
P(p_i = 0.2 |X_1, X_2, \dots , X_n) = \dfrac{C \left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right)}{\Delta}
$$
因为 $\Delta$ 中含有因子 $C$，所以可以化简上式
$$
P(p_i = 0.2 |X_1, X_2, \dots , X_n) = \dfrac{C \left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right)}{C \left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right) + C \left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)} \\
 = \dfrac{\left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right)}{\left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right) + \left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)}
$$
同理，也可以得到
$$
P(p_i = 0.9 | X_1, X_2, \dots, X_n) = \dfrac{\left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)}{\left(  0.2^{n \bar{x}} {0.8}^{nN - n \bar{x}} \right) \cdot \left( 0.8 \right) + \left(  0.9^{n \bar{x}} {0.1}^{nN - n \bar{x}} \right) \cdot \left( 0.2 \right)}
$$
假设损失函数是平方损失，那么 $p$ 的贝叶斯统计量就是 $p|X_1, X_2, \dots, X_n$ 的数学期望
$$
\tilde{p}_B = E(p|X_1, X_2, \dots, X_n) = \sum_{i=1}^{2} p_i P(p_i | X)
$$

> 计算时记得把 $x$ 替换成 $X$. 

$$
\tilde{p}_B = 0.2 \times P(p_i = 0.2 |X_1, X_2, \dots , X_n) + 0.9 \times P(p_i = 0.9 | X_1, X_2, \dots, X_n)
\\
= 0.2 \times \dfrac{\left(  0.2^{n \bar{X}} {0.8}^{nN - n \bar{X}} \right) \cdot \left( 0.8 \right)}{\left(  0.2^{n \bar{X}} {0.8}^{nN - n \bar{X}} \right) \cdot \left( 0.8 \right) + \left(  0.9^{n \bar{X}} {0.1}^{nN - n \bar{X}} \right) \cdot \left( 0.2 \right)} \\ + 0.9 \times \dfrac{\left(  0.9^{n \bar{X}} {0.1}^{nN - n \bar{X}} \right) \cdot \left( 0.2 \right)}{\left(  0.2^{n \bar{X}} {0.8}^{nN - n \bar{X}} \right) \cdot \left( 0.8 \right) + \left(  0.9^{n \bar{X}} {0.1}^{nN - n \bar{X}} \right) \cdot \left( 0.2 \right)}
$$

上下约分整理， 得到 $p$ 的贝叶斯估计量为
$$
\tilde{p}_B = \dfrac{0.2^{n \bar{X}} 0.8^{nN-n\bar{X}+1} +0.9^{n\bar{X}+1} 0.1^{nN-n\bar{X}} }{0.2^{n \bar{X}-1} 0.8^{nN-n\bar{X}+1} + 0.9^{n\bar{X}} 0.1^{nN-n\bar{X}} }
$$

> 本题来自孙荣恒编写的《应用数理统计（第三版）》103页的39题的 (1)，书后答案貌似不正确. 
