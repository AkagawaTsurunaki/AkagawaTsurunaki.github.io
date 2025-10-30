# 为什么“总体方差 ≠ 偏样本方差”？

> Ciallo～(∠・ω< )⌒★ 我是赤川鹤鸣！这一次是在推导强化学习公式的时候遇到的一个式子的转换，由此延伸出了总体方差和偏样本方差的区别. 

已知样本均值 $ \bar{x} = \frac{1}{n} \sum_{i=1}^n x_i $，偏样本方差 $\sigma_{x}^2 = \frac{1}{n} \sum_{i=1}^n \left( x_{i} - \bar{x} \right)^2$，总体方差 $s^2 = \frac{1}{n} \sum_{i=1}^n \left( x_{i} - \mu \right)^2$

试证明
$$
s^2 = \sigma_x^{2} + \left( \bar{x} - \mu\right)^2
$$

----

原式可化为
$$
\dfrac{1}{n} \sum_{i=1}^n \left( x_{i} - \mu \right)^2 = \dfrac{1}{n} \sum_{i=1}^n \left( x_{i} - \bar{x} \right)^2 + \left( \bar{x} - \mu \right)^2 \tag{1}
$$
两侧同时乘 $n$，得
$$
\sum_{i=1}^n \left( x_{i} - \mu \right)^2 = \sum_{i=1}^n \left( x_{i} - \bar{x} \right)^2 + n \left( \bar{x} - \mu \right)^2 \tag{2}
$$


令
$$
A = \sum_{i=1}^n \left( x_{i} - \mu \right)^2 - \sum_{i=1}^n \left( x_{i} - \bar{x} \right)^2 - n \left( \bar{x} - \mu \right)^2 \tag{3}
$$
只需要证明 $A=0$ 即可.

将 $(3)$ 中的乘方打开，得
$$
A = \sum_{i=1}^{n} x_i^2 - 2 \mu \sum_{i=1}^{n} x_i + n\mu^2 - \sum_{i=1}^{n} x_i^2 + 2\bar{x} \sum_{i=1}^{n} x_i - n \bar{x}^2 - n \left( \bar{x} - \mu \right)^2 \tag{4}
$$
将 $(4)$ 中的 $\sum_{i=1}^{n} x_i^2$ 抵消，得
$$
A =  - 2 \mu \sum_{i=1}^{n} x_i + n\mu^2 + 2\bar{x} \sum_{i=1}^{n} x_i - n \bar{x}^2 - n \left( \bar{x} - \mu \right)^2 \tag{5}
$$
将 $(5)$ 中的乘方进一步打开，合并同类项，得
$$
\begin{aligned}
A &=  - 2 \mu \sum_{i=1}^{n} x_i + n\mu^2 + 2\bar{x} \sum_{i=1}^{n} x_i - n \bar{x}^2 - n \bar{x}^2 + 2 n \bar{x} \mu  - n \mu^2
 \\ &= - 2 \mu \sum_{i=1}^{n} x_i + n\mu^2 + 2\bar{x} \sum_{i=1}^{n} x_i -2 n \bar{x}^2 + 2 n \bar{x} \mu - n \mu^2
 \\ &= - 2 \mu \sum_{i=1}^{n} x_i + 2\bar{x} \sum_{i=1}^{n} x_i -2 n \bar{x}^2 + 2 n \bar{x} \mu
\end{aligned}
\tag{6}
$$
又因为 $ \bar{x} = \dfrac{1}{n} \sum_{i=1}^n x_i $，所以代入式子 $(6)$ 得
$$
\begin{aligned}
A &= -2n\mu \bar{x} + 2n \bar{x} ^2 -2 n \bar{x}^2 + 2 n \bar{x} \mu
\\ &= -2n\mu \bar{x} + 2 n \bar{x} \mu
\\ &= 0
\end{aligned}
\tag{7}
$$
证明完毕.

这说明通常情况下，**总体方差不等于偏样本方差，它们之间的误差是** $\left( \bar{x} - \mu\right)^2$.
