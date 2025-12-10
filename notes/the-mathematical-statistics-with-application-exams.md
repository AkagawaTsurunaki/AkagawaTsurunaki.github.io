# 应用数理统计历年真题
作者：赤川鹤鸣_Channel | Author: AkagawaTsurunaki

> [!NOTE] 
>
> 所有问题及其答案已经过人工和 AI （Kimi、Deepseek 和 Gemini）双重评判，但仍可能存在错误，请保持严谨细心的态度看待. 
>
> 根据作者对历年真题的分析，从 2020 年开始，无论是证明题还是计算题，都开始更加多元化、灵活化、技巧化，需要考生掌握知识点的融会贯通，但繁琐计算（如大量加减乘除）的运算量减少. 
>
> 根据往年出题规律：第一章，一道题，第二章，贝叶斯估计一道题、传统估计两道题；第三章，参数检验一道题，非参数检验一道题；第四章单因素方差分析，一道题；第五章一元线性回归模型，一道题.

我们约定：
1. 样本都是简单随机样本. 样本均值与样本方差分别定义为

$$
\bar{X} = \dfrac{1}{n} \sum_{i=1}^{n} X_k , \quad S^2 = \dfrac{1}{n-1} \sum_{i=1}^{n} (X_k - \bar{X})^2
$$

2. 分位点 $Q_\alpha$ 取为上侧分位点，即：$P(X>Q_\alpha)=\alpha$. 

## 模拟卷 I

> [!WARNING]
>
> 模拟卷由作者通过分析历年真题，在 AI 的辅助下完成出题，本套不是考试真题，仅供学习参考. 只要不带“模拟卷”三个字的都是真题卷. 

### 一、计算分析题（10分）

设 $X_1, X_2, \dots, X_n$ 相互独立，且 $X_i \sim N(\mu, \sigma^2_i)$，$i=1,2,\dots,n$，$S_\sigma = \sum_{i=1}^{n} \frac{1}{\sigma_i}$. 试证：
$$
U = \sum_{i=1}^{n} \dfrac{X_i}{\sigma_i S_\sigma} \ 与 \ V = \sum_{i=1}^{n} \left[ \dfrac{X_i - \mu}{\sigma_i} - \dfrac{U - \mu}{n} S_\sigma \right]^2
$$
相互独立，且 $U$ 服从正态分布，$V \sim \chi^2(n-1)$. 

> [!TIP]
>
> 本题来自书后第一章习题，需要使用书中 1.2 节的多元正态分布与正态二次型的相关知识. 

**解：**

由于 $U$ 是 $X_1, X_2, \dots, X_n$ 的线性组合，所以 $U$ 一定服从正态分布. 

计算 $U$ 的期望和方差
$$
E(U) = \dfrac{1}{S_\sigma} \sum_{i=1}^{n} E \left( \dfrac{X_i}{\sigma_i} \right)
= \dfrac{1}{S_\sigma} \sum_{i=1}^{n} \dfrac{1}{\sigma_i}  E \left( X_i \right)
= \dfrac{1}{S_\sigma} \sum_{i=1}^{n} \dfrac{1}{\sigma_i} \mu
= \dfrac{1}{S_\sigma}\mu \sum_{i=1}^{n} \dfrac{1}{\sigma_i} 
= \dfrac{1}{S_\sigma}\mu S_\sigma 
= \mu 
\\
D(U) = \dfrac{1}{S_\sigma^2} D \left( \sum_{i=1}^{n} \dfrac{X_i}{\sigma_i} \right)
= \dfrac{1}{S_\sigma^2} \sum_{i=1}^{n} \dfrac{1}{\sigma_i^2} D \left(X_i \right)
= \dfrac{1}{S_\sigma^2} \sum_{i=1}^{n} \dfrac{1}{\sigma_i^2} \sigma_i^2
= \dfrac{n}{S_\sigma^2}
$$
对于 $V$ 来说，设 $Z_i = \dfrac{X_i-\mu}{\sigma_i} \sim N(0,1)$，$\bar{Z} = \dfrac{U - \mu}{n} S_\sigma = \dfrac{U-\mu}{\sqrt{n/S_\sigma^2}} \sim N \left(0, \dfrac{1}{n} \right)$，可以得到
$$
V = \sum_{i=1}^{n} \left( Z_i - \bar{Z} \right)^2
$$
设 $W_i = Z_i - \bar{Z}$，计算 $W_i$ 的均值与方差
$$
E(W_i) = E(Z_i - \bar{Z}) = E(Z_i) - E(\bar{Z}) = 0-0 = 0 \\
D(W_i) = D(Z_i - \bar{Z}) = D(Z_i) + D(\bar{Z}) - 2\mathrm{Cov} (Z_i , \bar{Z}) = 1 + \dfrac{1}{n} - 2\mathrm{Cov} (Z_i , \bar{Z})
$$
接下来计算 $\mathrm{Cov} (Z_i , \bar{Z})$，即
$$
\begin{align*}
\mathrm{Cov} (Z_i , \bar{Z}) 
&= \mathrm{Cov} \left(\dfrac{X_i-\mu}{\sigma_i} ,\dfrac{U - \mu}{n} S_\sigma \right) \\
&= \dfrac{S_\sigma}{n \sigma_i} \mathrm{Cov} \left(X_i-\mu , U - \mu \right) \\
&= \dfrac{S_\sigma}{n \sigma_i} \mathrm{Cov} \left(X_i, U \right) \\
&= \dfrac{S_\sigma}{n \sigma_i} \mathrm{Cov} \left(X_i, \sum_{j=1}^{n} \dfrac{X_j}{\sigma_j S_\sigma} \right) \\
&= \dfrac{S_\sigma}{n \sigma_i} \mathrm{Cov} \left(X_i, \dfrac{X_i}{\sigma_i S_\sigma} \right) \\
&= \dfrac{1}{n \sigma_i^2} \mathrm{Cov} \left(X_i, X_i \right) \\
&= \dfrac{1}{n \sigma_i^2} D(X_i) \\
&= \dfrac{1}{n \sigma_i^2} \sigma_i^2 \\
&= \dfrac{1}{n} 
\end{align*}
$$
所以 $D(W_i) = 1 + \dfrac{1}{n} - 2\mathrm{Cov} (Z_i , \bar{Z}) = 1 -  \dfrac{1}{n}$. 

所以 $W_i = Z_i - \bar{Z} \sim N \left(0, 1 - \dfrac{1}{n} \right)$. 

设随机变量组成的向量 $\boldsymbol{Z} = \left(Z_1, Z_2, \dots, Z_n \right)^T$，$\bar{Z} \boldsymbol{1}^T \boldsymbol{1} = (\bar{Z},\bar{Z},\dots,\bar{Z})^T$，则可以把 $V$ 改写成
$$
V = (\boldsymbol{Z} - \bar{Z} \boldsymbol{1}^T \boldsymbol{1})^T (\boldsymbol{Z} - \bar{Z} \boldsymbol{1}^T \boldsymbol{1}) = (W_1, W_2, \dots, W_n) (W_1, W_2, \dots, W_n)^T
$$
对分量进行标准化
$$
V = \left(1-\dfrac{1}{n} \right) \left(\dfrac{\boldsymbol{Z} - \bar{Z} \boldsymbol{1}^T \boldsymbol{1}}{\sqrt{1-\dfrac{1}{n}}} \right)^T \boldsymbol{I}_n \left(\dfrac{\boldsymbol{Z} - \bar{Z} \boldsymbol{1}^T \boldsymbol{1}}{\sqrt{1-\dfrac{1}{n}}} \right)
$$
这步操作的目的是得到了
$$
V = \left(1-\dfrac{1}{n} \right) \boldsymbol{Y}^T \boldsymbol{I}_n \boldsymbol{Y}, \quad \boldsymbol{Y} \sim N(\boldsymbol{0},\boldsymbol{1} )
$$
根据 Cochran 定理可知，如果存在一个对称阵 $\boldsymbol{A}$，那么 $\boldsymbol{Y}^T \boldsymbol{A} \boldsymbol{Y} \sim \chi^2(\mathrm{rank} \left(\boldsymbol{A}) \right)$，在这里 $\boldsymbol{A} = \left(1-\dfrac{1}{n} \right) \boldsymbol{I}_n$. 

所以就有
$$
\mathrm{rank} \left(\boldsymbol{A} \right) = n \left(1-\dfrac{1}{n}\right) = n - 1\\
V \sim \chi^2(n-1)
$$
> [!TIP]
>
> **可测函数独立性判别**
>
> 如果 $X$ 与 $Y$ 独立，且 $f, g$ 是可测函数，那么随机变量 $U = f(X)$ 与 $V = g(Y)$ 也独立. 一般地，我们学过的初等函数都是可测的.  

由于样本均值 $\bar{Z}$ 与样本平方和 $\sum_{i=1}^n (Z_i - \bar{Z})^2$ 是独立的，所以 $U$ 与 $V$ 也是独立的. 

### 二、计算题（15分）

设随机样本 $ X_1,\dots ,X_n $ 来自指数分布 $ X_i\mid \theta \sim E(\theta),\ \theta>0$，取 $\theta$ 的先验分布为 Gamma 分布 $ \theta\sim \operatorname{Gamma}(a,b)$，$a>0,\ b>0$. 

1. 给定观测样本 $x_1,\dots ,x_n$，在平方误差损失下，求 $\theta$ 的贝叶斯估计 $\hat{\theta}_B$. 

2. 现考虑对指数分布的均值 $\mu=\dfrac{1}{\theta}$ 进行估计，在平方误差损失下，求 $\mu$ 的贝叶斯估计 $\hat{\mu}_B$. 

提示：设 $X>0$ 是连续随机变量，概率密度函数为 $f_X(x)$，令 $Y=1/X$，则
$$
f_Y(y)=f_X\!\left(\frac{1}{y}\right)\cdot\frac{1}{y^{2}},\qquad y>0
$$
**解1：**

似然函数
$$
L(\theta)=\prod_{i=1}^{n}\theta e^{-\theta x_i}=\theta^{n}e^{-\theta\sum x_i}
$$
先验分布 $\theta\sim\mathrm{Gamma}(a,b)$ 密度
$$
\pi(\theta)=\frac{b^{a}}{\Gamma(a)}\theta^{a-1}e^{-b\theta},\quad\theta>0
$$
后验分布
$$
\pi(\theta\mid\boldsymbol{x})\propto L(\theta)\pi(\theta)\propto\theta^{n}e^{-\theta\sum x_i}\cdot\theta^{a-1}e^{-b\theta}=\theta^{n+a-1}e^{-(b+\sum x_i)\theta}
$$
即
$$
\theta\mid\boldsymbol{x}\sim\mathrm{Gamma}(n+a,\,b+\sum x_i)
$$
平方误差损失下贝叶斯估计为后验均值
$$
\hat{\theta}_B=E[\theta\mid\boldsymbol{x}]=\frac{n+a}{b+\sum_{i=1}^{n}x_i}
$$
**解2：**

由提示，令 $Y=1/\theta$，则后验密度
$$
f_Y(y\mid\boldsymbol{x})=f_{\theta\mid\boldsymbol{x}}\!\left(\frac{1}{y}\right)\cdot\frac{1}{y^{2}},\quad y>0
$$
代入 $\theta\mid\boldsymbol{x}\sim\mathrm{Gamma}(n+a,\,b+\sum x_i)$ 密度：
$$
f_Y(y\mid\boldsymbol{x})=\frac{(b+\sum x_i)^{n+a}}{\Gamma(n+a)}\left(\frac{1}{y}\right)^{n+a-1}e^{-(b+\sum x_i)/y}\cdot\frac{1}{y^{2}}
=\frac{(b+\sum x_i)^{n+a}}{\Gamma(n+a)}y^{-(n+a+1)}e^{-(b+\sum x_i)/y}
$$
这是逆 Gamma 分布 $Y\sim\mathrm{I}\Gamma(n+a,\,b+\sum x_i)$，其均值为
$$
E[Y\mid\boldsymbol{x}]=\frac{b+\sum x_i}{n+a-1},\quad(n+a>1)
$$
因此
$$
\hat{\mu}_B=E\left[\frac{1}{\theta}\Bigm|\boldsymbol{x}\right]=\frac{b+\sum_{i=1}^{n}x_i}{n+a-1}\quad(a+n>1)
$$

### 三、计算分析题（15分）

一组简单随机样本 $X_1, X_2, \dots X_n$ 服从 Weibull 分布，Weibull 分布的概率密度函数为
$$
f(x) = \frac{α}{β} \left(\frac{x}{β}\right)^{α-1} e^{-(x/β)^α}, \quad x>0
$$
其中 $\alpha >0$，$ \beta>0$，$\alpha$ 是已知参数. 

1. 求参数 $\beta$ 的极大似然估计 $ \hat{\beta}$. 
2. 当 $k$ 为何值时（用 $\alpha$ 表示），$k \bar{X}$ 是 $\beta$ 的无偏估计？
3. 利用 Cramér-Rao 不等式证明：对于 $\beta$ 的任何无偏估计量 $\varphi$ 都满足 $D(\varphi) \geq \dfrac{\beta^2}{n \alpha^2}$. 

**解1：**

对数似然函数
$$
\ln L(\beta) = n\ln\alpha - n\alpha\ln\beta + (\alpha-1)\sum_{i=1}^{n}\ln X_i - \sum_{i=1}^{n}\left(\frac{X_i}{\beta}\right)^{\alpha}
$$
对 $\beta$ 求导并令为零
$$
\frac{\mathrm{d}\ln L(\beta)}{\mathrm{d}\beta} = -\frac{n\alpha}{\beta} + \frac{\alpha}{\beta}\sum_{i=1}^{n}\left(\frac{X_i}{\beta}\right)^{\alpha} = 0
\;\Longrightarrow\;
\sum_{i=1}^{n}X_i^{\alpha} = n\beta^{\alpha}.
$$
解得
$$
\hat{\beta} = \left(\frac{1}{n}\sum_{i=1}^{n}X_i^{\alpha}\right)^{1/\alpha}
$$
**解2：**

计算
$$
E(X)=\int_{0}^{\infty} x\,f(x)\,\mathrm{d}x
=\int_{0}^{\infty} x\,\frac{\alpha}{\beta}\left(\frac{x}{\beta}\right)^{\alpha-1}
e^{-(x/\beta)^{\alpha}}\mathrm{d}x
$$
令换元
$$
t=\left(\frac{x}{\beta}\right)^{\alpha}\quad\Longrightarrow\quad
x=\beta\,t^{1/\alpha},\quad
\mathrm{d}x=\frac{\beta}{\alpha}\,t^{\frac{1}{\alpha}-1}\mathrm{d}t.
$$
代入积分
$$
E(X)=\frac{\alpha}{\beta}\int_{0}^{\infty}
\beta\,t^{1/\alpha}\,t^{1-\frac{1}{\alpha}}\,
e^{-t}\,\frac{\beta}{\alpha}\,t^{\frac{1}{\alpha}-1}\mathrm{d}t
=\beta\int_{0}^{\infty}t^{1/\alpha}\,e^{-t}\,\mathrm{d}t.
$$
识别 Gamma 函数
$$
\int_{0}^{\infty}t^{1/\alpha}\,e^{-t}\,\mathrm{d}t
=\Gamma\!\left(1+\frac{1}{\alpha}\right).
$$
故
$$
E(X)=\beta\,\Gamma\!\left(1+\frac{1}{\alpha}\right).
$$
无偏估计量的数学期望为待估参数本身
$$
E(k\bar{X}) = k\beta\,\Gamma\!\left(1+\frac{1}{\alpha}\right) = \beta
$$
所以
$$
k = \frac{1}{\Gamma\!\left(1+\frac{1}{\alpha}\right)}
$$
**解3：**


Fisher 信息量的公式为
$$
\mathcal I(\beta)=-E\!\left[\frac{\partial^2\ln L(\beta)}{\partial\beta^2}\right]
$$

对数似然函数
$$
\ln L(\beta)=n\ln\alpha-n\alpha\ln\beta+(\alpha-1)\sum\ln X_i-\sum\Bigl(\frac{X_i}{\beta}\Bigr)^{\alpha}
$$

先求一阶导数
$$
\frac{\partial\ln L(\beta)}{\partial\beta}=-\frac{n\alpha}{\beta}+\frac{\alpha}{\beta}\sum\Bigl(\frac{X_i}{\beta}\Bigr)^{\alpha}
$$

再求二阶导数
$$
\frac{\partial^{2}\ln L(\beta)}{\partial\beta^{2}}
=\frac{n\alpha}{\beta^{2}}-\frac{\alpha(\alpha+1)}{\beta^{2}}\sum\Bigl(\frac{X_i}{\beta}\Bigr)^{\alpha}
$$

然后，我们需要计算
$$
E \left[\sum\Bigl(\frac{X_i}{\beta}\Bigr)^{\alpha}\right]=n E\left[\Bigl(\frac{X}{\beta}\Bigr)^{\alpha}\right] = \dfrac{n}{\beta^\alpha} E(X^{\alpha})
$$

其次，我们计算 $E(X^{\alpha})$
$$
E(X^{\alpha})=\int_0^{\infty} x^{\alpha}\,\frac{\alpha}{\beta}\Bigl(\frac{x}{\beta}\Bigr)^{\alpha-1}e^{-(x/\beta)^{\alpha}}\mathrm{d}x
$$

令 $t=(x/\beta)^{\alpha}\Rightarrow x=\beta t^{1/\alpha},\;\mathrm{d}x=\frac{\beta}{\alpha}t^{\frac{1}{\alpha}-1}\mathrm{d}t$，代入得
$$
E(X^{\alpha})=\frac{\alpha}{\beta}\int_0^{\infty}
\beta^{\alpha}t\,t^{1-\frac{1}{\alpha}}\,e^{-t}\,
\frac{\beta}{\alpha}t^{\frac{1}{\alpha}-1}\mathrm{d}t
=\beta^{\alpha}\int_0^{\infty} t\,e^{-t}\mathrm{d}t
=\beta^{\alpha}\Gamma(2)=\beta^{\alpha}
$$

因此
$$
E\!\left[\sum\Bigl(\frac{X_i}{\beta}\Bigr)^{\alpha}\right]=\frac{n}{\beta^{\alpha}}\cdot\beta^{\alpha}=n
$$

代入二阶导期望
$$
-E\!\left[\frac{\partial^{2}\ln L(\beta)}{\partial\beta^{2}}\right]
=\frac{\alpha(\alpha+1)}{\beta^{2}}\,n-\frac{n\alpha}{\beta^{2}}
=\frac{n\alpha^{2}}{\beta^{2}}
$$

最后计算 Cramér-Rao 下界
$$
D(\varphi)\ge\frac{1}{n\mathcal I(\beta)}=\frac{\beta^2}{n\alpha^2}
$$

原命题得证. 

### 四、计算题（10分）

设 $X_1, X_2, \dots, X_m$ 来自于正态总体 $N(\mu_1, \sigma^2)$，$Y_1, Y_2, \dots, Y_n$ 来自于正态总体 $N(\mu_2, k\sigma^2)$，其中 $\mu_1, \mu_2, \sigma^2$ 均未知，考虑如下的假设检验问题
$$
H_0: a \mu_1 + b \mu_2 \leq c , \quad H_1: a \mu_1 + b \mu_2 > c
$$
其中，$a, b, c, k$ 均为已知非零常数. 

**解：**

根据抽样分布定理，有

$$
\frac{(m-1)S_{X}^{2}}{\sigma^{2}} \sim \chi^{2}(m-1), \quad \frac{(n-1)S_{Y}^{2}}{k\sigma^{2}} \sim \chi^{2}(n-1)
$$

构造公共方差估计（合并均方误差）

$$
S_{p}^{2} = \frac{(m-1)S_{X}^{2} + \frac{1}{k}(n-1)S_{Y}^{2}}{m+n-2}, \quad
\frac{(m+n-2)S_{p}^{2}}{\sigma^{2}} \sim \chi^{2}(m+n-2)
$$

定义线性对比估计量

$$
\hat{\theta} = a\bar{X} + b\bar{Y}, \quad
D(\hat{\theta}) = \sigma^{2}\left(\frac{a^{2}}{m} + \frac{b^{2}}{n}k\right)
$$

用公共方差估计代替 $\sigma^{2}$，得检验统计量

$$
T = \frac{\hat{\theta} - c}{S_{p}\sqrt{\dfrac{a^{2}}{m} + \dfrac{b^{2}}{n}k}}
$$

在 $H_0$ 下，$T \sim t(m+n-2)$，对右侧检验，因此拒绝域为

$$
T > t_{\alpha}(m+n-2)
$$

### 五、计算题（15分）

下表为辽宁某知名 985 大学 2025 届软件工程专业本科生毕业去向与学生本科所选细分专业的人数统计数据，有效人数总计为 338 人，请在显著性水平 $\alpha = 0.05$ 水平下检验该届本科毕业生去向与其所选细分专业是否有显著关系？

| 毕业去向 \ 细分专业 | 软工 | 软英 | 软金 | 软信 | 数媒 |
| ------------------- | ---- | ---- | ---- | ---- | ---- |
| 工作                | 111  | 2    | 9    | 14   | 15   |
| 考研                | 46   | 1    | 3    | 4    | 0    |
| 保研                | 80   | 9    | 10   | 20   | 14   |

**解：**

构建 Pearson 卡方统计量
$$
\begin{align*}
K^2 &= n \left[ \sum_{i=1}^{s} \sum_{j=1}^{t} \dfrac{n_{ij}^2}{n_{i*} n_{*j}} -1 \right]  \\
&= 
338\left[
\frac{111^{2}}{151\times237}+
\frac{2^{2}}{151\times12}+
\frac{9^{2}}{151\times22}+
\frac{14^{2}}{151\times38}+
\frac{15^{2}}{151\times29} 
\right. \\ & \quad +
\frac{46^{2}}{54\times237}+
\frac{1^{2}}{54\times12}+
\frac{3^{2}}{54\times22}+
\frac{4^{2}}{54\times38}+
\frac{0^{2}}{54\times29}
\\
&\quad +\frac{80^{2}}{133\times237}+
\frac{9^{2}}{133\times12}+
\frac{10^{2}}{133\times22}+
\frac{20^{2}}{133\times38}+
\left.\frac{14^{2}}{133\times29}-1\right]
\\
&= 19.13
\end{align*}
$$
拒绝域为
$$
\left\{ K^2 > \chi^2_{\alpha} \left( (s-1)(t-1) \right) \right\}
$$
由于此表有 $s=3$ 行，$t=5$ 列数据，所以拒绝域为 
$$
\left\{ K^2 > \chi^2_{0.05} \left( 8 \right) \right\}
$$
查表得 $\chi^2_{0.05} \left( 8 \right) = 15.507$. 

因为 $K^2=19.13>15.507 = \chi^2_{0.05} $，所以拒绝原假设，认为在显著性水平 $\alpha = 0.05$ 水平下检验该届本科毕业生去向与其所选细分专业是有显著关系. 

### 六、计算题（10分）

某研究人员想比较三种不同金坷垃（A、B、C）对植物生长的影响. 他将植物随机分成三组，每组施加不同的肥料，一段时间后测量 15 株植物的高度（单位：cm），但由于研究人员不小心的误操作，导致 $C$ 组数据缺失了 $2$ 份，记为 $a, b$（均为正整数）. 研究人员还记得，这次单因素方差分析的结果是在显著性水平 $0.1$ 的条件下，认为三种不同金克拉对植物生长有影响. 具体数据如下表：

| 组别 | 株高观测值 (cm)        |
| ---- | ---------------------- |
| A    | 12, 14, 13, 11, 12, 13 |
| B    | 12, 15, 13, 13, 14     |
| C    | 14, 14, $a$, $b$       |

已知，A 组数据：$\bar{x}_1 = 12.5, s_1^2 = 5.5/5$；B 组数据：$\bar{x}_2=13.4, \ s_2^2 = 5.2/4$；总均值 $\bar{x}=13.2$. 

1. 求解组间平方和 $CSS$ 的值. 
2. 根据以上信息，是否可以求解出 $a, \ b$ 的值？如果能，请直接求解；如果不能，请说明原因. 

**解1：**

先计算所有样本的总和
$$
\sum_{i=1}^{r} n_{i} \bar{x}_i = 6 \times 12.5 + 5 \times 13.4 + (14 + 14 + a + b) = n\bar{x} = 13.2 \times 15
$$
可以得出
$$
a+b=28
$$
也就是说我们可以算出 C 组的样本均值
$$
\bar{x}_3 = \dfrac{14+14+a+b}{4} = 14
$$
接着计算 $CSS$
$$
CSS = \sum_{i=1}^{r} n_i (\bar{x}_i - \bar{x})^2 = 6 \times (12.5-13.2)^2 + 5 \times (13.4-13.2)^2 + 4 \times (14-13.2)^2 = 5.7
$$
**解2：**

在显著性水平 $0.1$ 的条件下，认为三种不同金克拉对植物生长有影响，这说明我们拒绝了原假设.

也就是说对于检验统计量
$$
F = \dfrac{n-r}{r-1} \dfrac{CSS}{RSS} \sim F(r-1, n-r)
$$
其落入了拒绝域，即
$$
F \geq F_{\alpha}(r-1,n-r)
$$
查表得 $F_{0.1}(2,12)=2.81$，理论上会有
$$
F = \dfrac{12}{2} \times \dfrac{5.7}{RSS } \geq 2.81 = F_{0.1}(2,12)
$$
整理得
$$
RSS \leq 12.17
$$
也就是说
$$
\begin{align*}
RSS &= \sum_{i=1}^{r} (n_i -1)s_i^2 \\
&= 5\times \dfrac{5.5}{5}+ 4 \times \dfrac{5.2}{4} + \left( (14-14)^2 + (14-14)^2 +(a-14)^2 + (b-14)^2 \right) \\
&= 10.7 + (a-14)^2 + (b-14)^2 \\
&\xlongequal{a+b=28} 10.7+ (a-14)^2 + (28-a-14)^2 \\
&=10.7+ (a-14)^2 + (14-a)^2\\
&=10.7+ 2(a-14)^2\\
&\leq 12.17 
\end{align*}
$$
进一步整理得
$$
(a-14)^2 \leq 0.735
$$
由题意得，$a$ 是正整数，所以 $a$ 只能取 $14$，综上所述，我们得出
$$
a = 14, \quad b = 14
$$

### 七、计算题（10分）

某玩家怀疑某 Minecraft 服务器主世界的钻石矿生成异常，于是随机抽取了 $n = 64$ 个区块中的钻石矿，得到样本最小值 $a = 0$ 个、最大值 $b = 10$ 个、众数 $c = 4$ 个钻石矿/区块. 他按照按离散三角分布将区间 $[a,b]$ 等分成 $ k = 5 $ 组，各组观测频数如下：

| 组号            | 1      | 2      | 3    | 4      | 5       |
| --------------- | ------ | ------ | ---- | ------ | ------- |
| 钻石矿区间 / 个 | [0, 1] | [2, 3] | 4    | [5, 6] | [7, 10] |
| 观测频数        | 6      | 10     | 22   | 14     | 12      |

试问在显著性水平 $\alpha = 0.01$ 的情况下，是否拒绝“钻石矿数量服从三角分布”的原假设？

提示：三角分布的概率质量函数为
$$
P(X=x)=
\begin{cases}
\dfrac{2(x-a)}{(b-a)(c-a)}, & a\le x\le c\\[6px]
\dfrac{2(b-x)}{(b-a)(b-c)}, & c<x\le b
\end{cases}
$$
**解：**

根据题意，写出该三角分布的具体概率质量函数
$$
P(X=x)=
\begin{cases}
\dfrac{2x}{10 \times 4}=\dfrac{x}{20}, & 0\le x\le 4\\[6px]
\dfrac{2(10-x)}{10\times 6}=\dfrac{10-x}{30}, & 4< x\le 10
\end{cases}
$$
写各个分组 $i$ 所占的概率 $p_i$

| 组号 | 区间    | 所含整数 $x$ | 概率和 $p_i$                                                 |
| ---- | ------- | ------------ | ------------------------------------------------------------ |
| 1    | [0, 1]  | 0, 1         | $\dfrac{0}{20}+\dfrac{1}{20}=\dfrac{1}{20}=0.05$             |
| 2    | [2, 3]  | 2,3          | $\dfrac{2}{20}+\dfrac{3}{20}=\dfrac{5}{20}=0.25$             |
| 3    | 4       | 4            | $\dfrac{4}{20}=0.20$                                         |
| 4    | [5, 6]  | 5, 6         | $\dfrac{5}{30}+\dfrac{4}{30}=\dfrac{9}{30}=0.30$             |
| 5    | [7, 10] | 7, 8, 9, 10  | $\dfrac{3}{30}+\dfrac{2}{30}+\dfrac{1}{30}+\dfrac{0}{30}=\dfrac{6}{30}=0.20$ |

检验统计量
$$
\begin{align*}
K^2 &= \dfrac{1}{n} \sum_{i=1}^{k} \dfrac{v_i^2}{p_i} - n \\
 &= \dfrac{1}{64} \left( \dfrac{6^2}{0.05} + \dfrac{10^2}{0.25} + \dfrac{22^2}{0.20} + \dfrac{14^2}{0.30} + \dfrac{12^2}{0.20} \right) - 64
\\
 &= 12.77
\end{align*}
$$
拒绝域为
$$
K^2 > \chi^2(k-r-1)
$$
其中 $k=5$ 是分组数，$r = 0$ 是未知参数的个数，即拒绝域为
$$
K^2 > \chi^2_\alpha(4)
$$
查表可知，$\chi^2_{0.01}(4)=13.227$. 

因为 $K^2 = 12.77 < 13.227 = \chi^2_{0.01}(4)$，所以在显著性水平 $0.01$ 下不能拒绝原假设，认为钻石矿分布遵循该三角分布. 

### 八、计算题（15分）

考虑阿伦尼乌斯型关系

$$
k = A\exp\!\left(-\frac{E}{RT}\right)
$$
其中 $R$ 已知，$A>0$，$E>0$ 为待定参数。对 $n$ 组温度 $T_i$ 测得速率常数 $k_i, (i=1,\dots,n ) $. 
1. 基于一元回归模型的思想，利用最小二乘法求解回归方程 $\hat{k}(T)$. 
2. 对于下列检验问题进行检验并给出置信水平为 $\alpha$ 的拒绝域. 

$$
H_0: \lambda_2 \dfrac{\hat{E}}{R} + \lambda_1 \ln \dfrac{1}{\hat{A}} < 0, \quad H_1: \lambda_2 \dfrac{\hat{E}}{R} + \lambda_1 \ln \dfrac{1}{\hat{A}} > 0
$$

**解1：**
$$
k=A\,e^{-E/(R T)} \quad\Longrightarrow\quad
y=\ln k=\beta_{0}+\beta_{1}x,\quad x=\frac{1}{T},\quad \beta_{0}=\ln A,\quad \beta_{1}=-\frac{E}{R}.
$$

设计矩阵
$$
\underset{n\times 2}{\mathbf{X}}=
\begin{bmatrix}
1 & x_{1}\\[4pt]
1 & x_{2}\\[4pt]
\vdots & \vdots\\[4pt]
1 & x_{n}
\end{bmatrix},\qquad
\underset{n\times 1}{\mathbf{y}}=
\begin{bmatrix}
y_{1}\\ y_{2}\\ \vdots\\ y_{n}
\end{bmatrix}.
$$
最小二乘估计
$$
\hat{\boldsymbol{\beta}}=(\mathbf{X}^{\!\top}\mathbf{X})^{-1}\mathbf{X}^{\!\top}\mathbf{y}.
$$
进一步计算
$$
\mathbf{X}^{\!\top}\mathbf{X}=
\begin{bmatrix}
n & \sum x_i\\[4pt]
\sum x_i & \sum x_i^{2}
\end{bmatrix},\qquad
\mathbf{X}^{\!\top}\mathbf{y}=
\begin{bmatrix}
\sum y_i\\[4pt]
\sum x_i y_i
\end{bmatrix}.
$$
得到
$$
\hat{\beta}_{1}=\frac{\sum_{i=1}^{n} x_iy_i -n \bar{x}\bar{y}}{\sum_{i=1}^{n} x_i^2 -n \bar{x}^2} =\frac{L_{xy}}{L_{xx}} ,\qquad
\hat{\beta}_{0}=\bar{y}- \hat{\beta}_1\bar{x}
$$

代回原始物理量
$$
\ln\hat{A}=\hat{\beta}_{0}\quad\Longrightarrow\quad
\hat{A}=\exp\!\left(\bar{y}-\frac{L_{xy}}{L_{xx}}\bar{x}\right)
\\
\hat{E}=-R\,\hat{\beta}_{1}\quad\Longrightarrow\quad
\hat{E}=-R\,\frac{L_{xy}}{L_{xx}}
$$
最终拟合方程
$$
\hat{k}(T)=\hat{A}\,e^{-\hat{E}/(R T)}=
\exp\!\left[\hat{\beta}_{0}+\frac{L_{xy}}{L_{xx}}\frac{1}{T}\right]
$$

**解2：**

给定
$$
H_0:\lambda_2\frac{\hat{E}}{R}+\lambda_1\ln\frac{1}{\hat A}<0.
$$

用估计量关系
$$
\frac{\hat E}{R}=-\hat\beta_1,\quad \ln\frac{1}{\hat A}=-\ln\hat A=-\hat\beta_0.
$$

代入得
$$
\lambda_2(-\hat\beta_1)+\lambda_1(-\hat\beta_0)=-\lambda_1\hat\beta_0-\lambda_2\hat\beta_1.
$$

因此原假设等价于
$$
H_0:-\lambda_1\hat\beta_0-\lambda_2\hat\beta_1<0
$$
即
$$
H_0:\lambda_1\hat\beta_0 + \lambda_2\hat\beta_1 > 0
$$

令
$$
\theta := \lambda_1\beta_0 + \lambda_2\beta_1 = \mathbf{c}^\top\boldsymbol{\beta}, \quad
\mathbf{c}= \begin{bmatrix}\lambda_1 \\ \lambda_2\end{bmatrix}.
$$

则
$$
\hat\theta = \lambda_1\hat\beta_0 + \lambda_2\hat\beta_1 = \mathbf{c}^\top\hat{\boldsymbol{\beta}}.
$$

由一元正态线性模型
$$
\hat{\boldsymbol{\beta}}\sim N\!\bigl(\boldsymbol{\beta},\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}\bigr),
$$
得（多元正态分布的性质）
$$
\hat\theta\sim N\!\bigl(\theta,\sigma^2\mathbf{c}^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{c}\bigr)
$$

其中方差是 $\mathbf{c}^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{c}= \frac{\lambda_1^2}{n}+\frac{(\lambda_2-\lambda_1\bar{x})^2}{L_{xx}}$. 

中心标准化
$$
Z:=\dfrac{\hat\theta - \theta}{\hat\sigma\sqrt{\frac{\lambda_1^2}{n}+\frac{(\lambda_2-\lambda_1\bar{x})^2}{L_{xx}}}}
\sim N(0,1)
$$
由
$$
K^2 := \frac{(n-2)\hat\sigma^2}{\sigma^2}\sim\chi^2(n-2),
$$
且 $\hat\sigma^2$ 与 $\hat{\boldsymbol\beta}$ 独立，故
$$
T = \frac{Z}{\sqrt{K^2/(n-2)}} = \frac{\hat\theta - \theta}{\hat\sigma\sqrt{\frac{\lambda_1^2}{n}+\frac{(\lambda_2-\lambda_1\bar{x})^2}{L_{xx}}}}\sim t(n-2)
$$

拒绝域为
$$
\left\{ \frac{\lambda_1\hat\beta_0 + \lambda_2\hat\beta_1}{\hat\sigma\sqrt{\frac{\lambda_1^2}{n}+\frac{(\lambda_2-\lambda_1\bar{x})^2}{L_{xx}}}} > t_{\alpha}(n-2) \right\}
$$

## 2024-2025

> [!NOTE]
>
> 这套卷来源不详，但出题风格和历年真题（尤其是近 5 年）有高度相似性，因此极有可能确实是 2024-2025 年的真题. 

### 一、计算题（共 10 分）

假设 $X_1, \dots, X_n$ 独立同分布于 Bernoulli 分布 $B(1,p )$. 

1. （5 分）求出 $T = X_1 + \cdots + X_n$ 的分布；
2. （5 分）证明 $P(X_1=x_1, \dots, X_2=x_n \mid T = t)$ 与参数 $p$ 无关. 

**解 1：**

根据二项分布的可加性

$$
T = X_1 + \cdots + X_n \sim B(n, p)
$$
**解 2：**

根据条件概率的定义，有

$$
P(X_1 = x_1, \dots, X_n = x_n \mid T = t) = \frac{P(X_1 = x_1, \dots, X_n = x_n \text{ 且 } T = t)}{P(T = t)}
$$
注意到当 $T = t$ 时，隐含的意义是有 $t$ 次实验成功了，$n - t$ 次失败了，这意味着在给定的 $x_1, \dots, x_n$ 中一定有 $t$ 个是 $1$，其余是 $0$；如果不满足此条件，那么这个事件发生的概率就是 $0$，也就是说分子满足
$$
P(X_1 = x_1, \dots, X_n = x_n \text{ 且 } T = t ) = P(X_1 = x_1, \dots, X_n = x_n) \cdot \mathbf{1} \left(\sum_{i=1}^{n} x_i=t \right) = p^{t} (1-p)^{n-t}
$$
而对于分母
$$
P(T=t) = \binom{n}{t} p^t (1-p)^{n-t}
$$

综上，有
$$
P(X_1 = x_1, \dots, X_n = x_n \mid T = t) = \dfrac{p^{t} (1-p)^{n-t}}{\binom{n}{t} p^t (1-p)^{n-t}} = \dfrac{1}{\binom{n}{t}}
$$
显然与参数 $p$ 无关. 


> [!TIP]
>
> 为什么分子没有 $\binom{n}{t}$？因为对于分子来说，样本观测值是已经确定了的，即你已经知道哪些样本一定为 1，哪些一定为 0，不需要再排列组合了. 这个结论其实隐含着这样一个思想，即如果你已经知道这么 $n$ 次伯努利实验中成功的次数 $t$，在这个条件下的样本观测值就和参数 $p$ 毫无关系了，它只与 $n$ 和 $t$ 有关.  

### 二、计算题（共 10 分）

假设简单随机样本 $X_1, \dots, X_n$ 来自泊松总体 $P(\lambda)$，参数 $\lambda$ 的先验分布为 $\lambda \sim \Gamma(1, 2)$，在如下损失函数下
$$
L(\theta, d) = \theta (\theta - d )^2
$$
求出 $\lambda$ 的 Bayes 估计. 

提示：$\lambda \sim \Gamma(\alpha, \beta)$ 期望是 $\alpha/\beta$，方差是 $\alpha/\beta^2$. 

**解：**

参数 $\lambda$ 与样本 $X$ 的联合分布是
$$
h(\lambda) \times f(x \mid \theta) = \left( \prod_{i=1}^{n} \dfrac{\lambda^{x_i}}{x_i !} e^{-\lambda} \right) \dfrac{2}{\Gamma(1)} \lambda^{1-1} e^{-2\lambda} 
\propto \lambda^{\sum_{i=1}^{n} x_i} e^{-n\lambda -2\lambda} = \lambda^{n\bar{x}} e^{-\lambda(n+2)}
$$
由于 Gamma 分布共轭于泊松分布，所以
$$
\lambda \mid X \sim \Gamma(n \bar{x} + 1, n+2)
$$
贝叶斯估计是使得风险函数的期望达到最小，那么
$$
E(L(\theta, d)) = E(\theta (\theta - d )^2) = E(\theta^3 - 2 \theta^2 d + \theta d^2) = E(\theta^3) - 2 d E(\theta^2) +  d^2 E(\theta)
$$
这是关于 $d$ 的二次函数（极值点可以口算），当然不失一般性我们仍然通过令偏导数为 $0$，得
$$
\dfrac{\partial}{\partial d} E(L(\theta, d))  = - 2 E(\theta^2) +  2d E(\theta) = 0
$$
所以 $d^{*} = \dfrac{E(\theta^2)}{E(\theta)}$. 

这里我们知道其实 $\theta = \lambda \mid X$，那么根据题意告诉我们的期望和方差来计算
$$
E(\lambda \mid X) = \dfrac{n \bar{x}+1}{n+2} \\
E(\lambda^2 \mid X) =  D(\lambda \mid X) + \left[ E(\lambda \mid X) \right]^2 = \dfrac{n \bar{x}+1}{(n+2)^2} + \dfrac{(n \bar{x}+1)^2}{(n+2)^2} = \dfrac{(n \bar{x}+1)(n \bar{x}+2)}{(n+2)^2}
$$
所以 $\lambda$ 的 Bayes 估计. 
$$
\hat{\lambda}_{B} = \dfrac{E(\lambda^2 \mid X)}{E(\lambda \mid X)} = \dfrac{n\bar{x} + 2}{n+2}
$$

> [!TIP]
>
> 你观察到什么规律了吗？可试算在损失函数 $L(\theta, d) = \theta^k (\theta -d )^2 ,k > 1$ 下的 $\lambda$ 的 Bayes 估计？（答案为：$\frac{n\bar{x}+1+k}{n+2}$）

### 三、计算题（共 15 分）

总体 $X$ 的概率密度函数为
$$
f(x) = \begin{cases}
\dfrac{10x}{\theta} e^{-\frac{5x^2}{\theta}} & x> 0 \\
0 & x \leq 0
\end{cases}
$$
其中 $\theta \ (\theta>0)$ 为未知参数，设 $X_1, \dots, X_n$ 是来自该总体的一个简单随机样本. 

1. （10分）求 $\theta$ 的极大似然估计量 $\hat{\theta}$；
2. （5分）判断 $\hat{\theta}$ 是否是 $\theta$ 的无偏估计？

**解1：**

似然函数为
$$
L(\theta) = \prod_{i=1}^{n} f(x) = \prod_{i=1}^{n} \dfrac{10x_i}{\theta} e^{-\frac{5x_i^2}{\theta}} = 
\left( \theta^{-n} 10^{n} \prod_{i=1}^{n} x_i \right) \exp\left(-\dfrac{5}{\theta} \sum_{i=1}^{n} x_i^2 \right)
$$
取对数
$$
\ln L(\theta) = -n \ln \theta + n \ln 10 + \sum_{i=1}^{n} \ln x_i  -\dfrac{5}{\theta} \sum_{i=1}^{n} x_i^2
$$
对 $\theta$ 求偏导并令偏导数为 $0$，得
$$
\dfrac{\partial \ln L(\theta)}{\partial \theta} = -\dfrac{n}{\theta} + \dfrac{5}{\theta^2} \sum_{i=1}^{n} x_i^2 = 0
$$
解得极大似然估计为
$$
\hat{\theta} = \dfrac{5}{n} \sum_{i=1}^{n} X_i^2
$$
**解2：**

欲证明极大似然估计为无偏估计，需要证明其期望正好等于参数
$$
E(\hat{\theta}) = E\left( \dfrac{5}{n} \sum_{i=1}^{n} X_i^2 \right) = \dfrac{5}{n} \sum_{i=1}^{n} E\left( X_i^2 \right)
$$
进而我们需要求解 $E(X^2)$，即
$$
\begin{align*}
E(X^2) &= \int_0^{\infty} x^2 f(x) \mathrm{d}x \\
&= \int_0^{\infty} x^2 \dfrac{10x}{\theta} e^{-\frac{5x^2}{\theta}} \mathrm{d}x  \\
&\xlongequal{t = \frac{5x^2}{\theta}} \int_0^{\infty} \dfrac{t\theta}{5} \cdot \dfrac{10}{\theta} \sqrt{\frac{\theta}{5}} t^{1/2} e^{-t} \cdot \sqrt{\frac{\theta}{5}} \dfrac{1}{2} t^{-1/2}  \mathrm{d}t \\
&=  \dfrac{\theta}{5}\underbrace{ \int_{0}^{\infty} t e^{-t} \mathrm{d}t}_{\text{Gamma 积分}} \\
&= \dfrac{\theta}{5} \Gamma(2) \\
&= \dfrac{\theta}{5}
\end{align*}
$$
代入极大似然估计的期望中
$$
E(\hat{\theta}) = \dfrac{5}{n} \cdot n \dfrac{\theta}{5} = \theta
$$
所以，$\hat{\theta}$ 是 $\theta$ 的无偏估计. 

### 四、计算分析题（共 15 分）

总体 $X$ 的概率密度函数为
$$
f(x) = \begin{cases}
\dfrac{2x}{\theta^2} & 0 <x<\theta \\
0 & 其他
\end{cases}
$$
其中 $\theta \ (\theta>0)$ 为未知参数，设 $X_1, \dots, X_n$ 是来自该总体的一个简单随机样本. 

1. （10分）令 $X_{(n)} = \max \left\{ X_1, \dots, X_n \right\}$，求 $\dfrac{X_{(n)}}{\theta}$ 的概率密度函数；
2. （5分）构造 $\theta$ 的置信水平为 $1-\alpha$ 的置信区间. 

**解1：**

设 $Y_{(n)} = \dfrac{X_{(n)}}{\theta}$，那么根据极大顺序统计量的概率密度函数公式
$$
f_{Y_{(n)}} (y) = n f_{Y}(y) \left[ F_Y(y) \right]^{n-1}
$$
接下来求概率密度函数，由于 $f(x) = \dfrac{2x}{\theta^2}$，其中 $0 <x<\theta$，不妨令 $y=\dfrac{x}{\theta}$，则根据高等数学中的**变量替换定理**

> [!TIP]
>
> **变量替换定理**
>
> 若随机向量 $X$  具有密度 $f_X(x) \mathrm{d} x$ ，令 $Y=\Phi(X)$ ，则 $Y$ 的密度为
> $$
> f_Y(y) = f_X\!\bigl(\Phi^{-1}(y)\bigr)\cdot \underbrace{\bigl|\det J_{\Phi^{-1}}(y)\bigr|}_{\text{Jacobian}}
> $$

$$
f_Y(y) = \dfrac{2}{\theta^2} \cdot \theta y \cdot \underbrace{\Big\vert \dfrac{\mathrm{d}x}{\mathrm{d}y} \Big \vert}_{\text{Jacobian}}
= \dfrac{2}{\theta^2} \cdot \theta y \cdot \theta = 2y
$$

然后，我们只需要求出累积分布函数即可
$$
F_Y (y) = \int_0^y 2 t \mathrm{d}t = t^2 \Big \vert_0^y = y^2
$$
最后
$$
f_Y(y) = n \cdot 2y \cdot (y^2)^{n-1} = 2ny^{2n-1}
$$
**解2：**

为了构造置信区间，我们按照定义写出
$$
P \left\{ a < \dfrac{X_{(n)}}{\theta} < b \right\} = P \left\{ a < Y < b \right\} = 1 -\alpha
$$
其中
$$
P(Y<a) = \int_0^a 2n y^{2n-1} \mathrm{d}y = y^{2n} \Big \vert_0^a = a^{2n} \\
P(Y>b) = \int_b^1 2n y^{2n-1} \mathrm{d}y = y^{2n} \Big \vert_b^1 = 1 - b^{2n} \\
$$
> [!TIP]
>
> 实际上，这一步我们计算的是在拒绝域上左右两侧的概率，而区间估计恰好是把两边抛去，取中间那一块大的，也就是说我们把 $\alpha$ 劈开两半分给上面的两个式子. 

取等尾区间，把 $\alpha$ 平分给两侧尾部
$$
P(Y<a) = a^{2n} = \dfrac{\alpha}{2}, \quad P(Y>b) = 1 - b^{2n} =\dfrac{\alpha}{2}
$$
解得
$$
a = \left(\dfrac{\alpha}{2}\right)^{\frac{1}{2n}}, \quad b = \left(1 - \dfrac{\alpha}{2}\right)^{\frac{1}{2n}}
$$
所以有
$$
\left(\dfrac{\alpha}{2}\right)^{\frac{1}{2n}} < \dfrac{X_{(n)}}{\theta} <  \left(1 - \dfrac{\alpha}{2}\right)^{\frac{1}{2n}}
$$
最终整理得到 $\theta$ 的置信水平为 $1-\alpha$ 的置信区间
$$
\left(
\dfrac{X_{(n)}}{\left(1 - \dfrac{\alpha}{2}\right)^{\frac{1}{2n}}}, \dfrac{X_{(n)}}{\left(\dfrac{\alpha}{2}\right)^{\frac{1}{2n}}} \right)
$$

### 五、计算题（共 15 分）

设 $X_1, \dots, X_n$ 为来自 Bernoulli 分布 $B(1,p)$ 的简单随机样本. 

1. （5 分）求假设 $H_0: p \geq 0.1$，$H_1: p < 0.1$ 的显著性水平为 $0.05$ 的显著性检验. 
2. （10 分）样本容量 $n$ 至少应为多少，才能保证这个检验在 $p=0.02$ 时犯第二类错误的概率不超过 $0.1$？

**解1：**

设 $p_s = 0.1$，则根据**中心极限定理**有
$$
Z:= \dfrac{p - p_s}{\sqrt{\dfrac{p_s (1-p_s)}{n}}} \sim N(0,1)
$$
题目要求的对立假设等价于 $p-p_s$ 偏小，这是一个单边假设检验. 
$$
P\{ p-p_s < C \} = P\left\{ \dfrac{p - p_s}{\sqrt{\dfrac{p_s (1-p_s)}{n}}} < \dfrac{C}{\sqrt{\dfrac{p_s (1-p_s)}{n}}}  \right\} = P\left\{ Z < \dfrac{C}{\sqrt{\dfrac{p_s (1-p_s)}{n}}}  \right\} = \alpha
$$
我们找到了 $C$，即
$$
\dfrac{C}{\sqrt{\dfrac{p_s (1-p_s)}{n}}}  = -u_{\alpha} \implies C = -u_{\alpha}\sqrt{\dfrac{p_s (1-p_s)}{n}}
$$
因此拒绝域为
$$
\left\{ p-p_s <  -u_{\alpha}\sqrt{\dfrac{p_s (1-p_s)}{n}} \right\}
$$
带入数据 $p_s = 0.1, \ u_{0.05} = 1.64$，即
$$
\left\{ p < \dfrac{-0.492}{\sqrt{n}} +0.1 \right\}
$$
**解2：**

第二类错误是指在对立假设 $H_1$ 为真的情况下却接受了原假设 $H_0$，其概率为
$$
P(接受 H_0| H_1 真) = P\left\{ \hat{p} > \dfrac{-0.492}{\sqrt{n}} +0.1 \mid p=0.02 \right\}

= P\left\{ \dfrac{\hat{p} - p }{\sqrt{p(1-p)/n}}> \dfrac{\dfrac{-0.492}{\sqrt{n}} - p +0.1}{\sqrt{p(1-p)/n}} \mid p=0.02 \right\} \\

= P\left\{ Z > \dfrac{\dfrac{-0.492}{\sqrt{n}} - 0.02 +0.1}{\sqrt{0.02 \times (1-0.02)/n}} \right\} \\
= P\left\{ Z > \dfrac{-0.492+0.08\sqrt{n}}{0.14} \right\} \\
= 1 - P\left\{ Z \leq \dfrac{-0.492+0.08\sqrt{n}}{0.14} \right\} \\
= 1 - \Phi\left(\dfrac{-0.492+0.08\sqrt{n}}{0.14} \right) \leq 0.1 = \beta
$$
根据分位数的定义得
$$
\Phi\left(\dfrac{-0.492+0.08\sqrt{n}}{0.14} \right) > 0.9 \implies
\dfrac{-0.492+0.08\sqrt{n}}{0.14} > 1.28 \implies \sqrt{n} > 8.39
$$
解得（都是正好的数）
$$
n \geq 70.3921
$$
所以样本容量 $n$ 至少应为 $71$.

### 六、计算题（共 10 分）

设总体 $X$ 的概率分布为：

| $X$  | -1   | 0    | 1      | 2    |
| ---- | ---- | ---- | ------ | ---- |
| $P$  | $p$  | $2p$ | $1-5p$ | $2p$ |

其中 $p \ (0 < p < 0.2)$ 为未知参数，现有一样本容量 $n=80$ 的简单随机样本，其中 $-1, \ 0, \ 1 $ 和 $2$ 分别出现了 $6, \ 15, \ 40, \ 19$ 次. 在显著性水平 $0.05$ 下能否可以认为“该组样本来自总体 $X$？

> [!TIP]
>
> 这道题中含有未知参数，我们需要先对参数进行估计，通常选择极大似然估计（教材这么写的）. 

**解：**

写出似然函数
$$
\begin{align*}
L(p) &= \prod_{i=1}^{n} p^{\mathbf{1} (x_i=-1)} (2p)^{\mathbf{1} (x_i=0)} (1-5p)^{\mathbf{1} (x_i=1)} (2p)^{\mathbf{1} (x_i=2)} \\
&= \underbrace{p^{\sum_{i=1}^{n} \mathbf{1} (x_i=-1)} }_{有多少个等于-1的样本} \underbrace{(2p)^{\sum_{i=1}^{n} \mathbf{1} (x_i=0)}}_{有多少个等于0的样本} (1-5p)^{\sum_{i=1}^{n} \mathbf{1} (x_i=1)} (2p)^{\sum_{i=1}^{n} \mathbf{1} (x_i=2)} \\
&= p^{6} (2p)^{15} (1-5p)^{40} (2p)^{19} \\
& = 2^{34} p^{40} (1-5p)^{40} \\
& = 2^{34} \left(\underbrace{ p(1-5p)}_{二次函数} \right)^{40}
\end{align*}
$$
极大似然估计要求我们使得似然函数达到最大，根据开口向下的二次函数必存在且有**唯一的**最大值（最大值点就是两根和的一半），可知
$$
\hat{p} = 0.1
$$

我们可以具体写出概率分布表
| $X$  | -1            | 0              | 1                | 2              |
| ---- | ------------- | -------------- | ---------------- | -------------- |
| $P$  | $\hat{p}=0.1$ | $2\hat{p}=0.2$ | $1-5\hat{p}=0.5$ | $2\hat{p}=0.2$ |

构建 Pearson 统计量
$$
\begin{align*}
K^2 &= \dfrac{1}{n} \sum_{i=1}^{k} \dfrac{v_i^2}{p_i} - n \\
&= \dfrac{1}{80} \left( \dfrac{6^2}{0.1} + \dfrac{15^2}{0.2} + \dfrac{40^2}{0.5} + \dfrac{19^2}{0.2} \right) - 80 \\
&= 1.125
\end{align*}
$$

拒绝域为
$$
\left\{ K^2 > \chi^2 (k-r-1) \right\}
$$
其中 $k=4$ 是分组数，$r=1$ 是未知参数的数量（我们刚才就是用极大似然估计估计了参数 $p$），即 $\left\{ K^2 > \chi^2_{\alpha} (2) \right\}$. 

查表得 $\chi^2_{0.05}(2) = 5.991$. 

因为 $K^2=1.125<5.991=\chi^2_{0.05}(2)$，所以不能拒绝原假设，在显著性水平 $0.05$ 下可以认为“该组样本来自总体 $X$. 

### 七、计算分析题（共 10 分）

某商场准备在商场内安装充电式应急照明灯，通过招标收到 $3$ 家照明灯生产商的投标，该商场对 $3$ 个生产商产品中进行抽样检验，以最终确定供应商。各个样品充电后可持续照明的时间长度(小时)数据如下：

$$
A_1: 9.2, 8.3, 9.0, 9.4, 9.8, 8.5, 8.8, 9.0 \\

A_2: 7.2, 8.3, 7.5, 8.2, 8.0, 8.9, 8.3, 7.6 \\

A_3: 10.5, 9.6, 10.8, 10.8, 11.3
$$

假设上述数据满足单因素方差分析模型的条件，在显著性水平 $\alpha = 0.05$ 下，检验三家生产商的电池的持续照明的平均寿命有无显著差异？

提示：分组均值与样本方差，$A_1: \overline{x}_1 = 9$，$s_1^2 = 1.62/7$；$A_2: \overline{x}_2 = 8$，$s_2^2 = 2.08/7$；$A_3: \overline{x}_3 = 10.6$，$s_3^2 = 1.58/4$；全部数据的均值与样本方差：$\overline{x} = 9$，$s^2 = 26.08/20$.

**解：**
$$
RSS = \sum_{i=1}^{r} (n_i - 1) s_i^2 = 7 \times \dfrac{1.62}{7} + 7 \times \dfrac{2.08}{7} + 4 \times \dfrac{1.58}{4} = 5.28 \\
TSS = (n-1)s^2 = 20 \times \dfrac{26.08}{20} = 26.08 \\
CSS = TSS - RSS = 26.08-5.28=20.8
$$
检验统计量
$$
F = \dfrac{n-r}{r-1} \dfrac{CSS}{RSS} \sim F(r-1, n-r)
$$
即 $F\sim F(2, 18)$. 

拒绝域为
$$
\left\{ F \geq F_{\alpha}(r-1,n-r) \right\}
$$
即 $\left\{ F \geq F_{0.05}(2,18) \right\}$. 

计算得
$$
F = \dfrac{18}{2} \times \dfrac{20.8}{5.28} = 35.45
$$
查表得 $F_{0.05} (2, 18) = 3.54$. 

因为 $F = 35.45 > 3.54 = F_{0.05}(2,18)$，所以拒绝原假设，认为三家生产商的电池的持续照明的平均寿命有显著差异. 

### 八、计算题（共 15 分）

考虑一元回归模型
$$
\begin{cases}
y = \beta_0 + \beta_1 x + \varepsilon \\
\varepsilon \sim N(0, \sigma^2)
\end{cases}
$$
记样本观测数据 $(x_i, y_i), i = 1, 2, \ldots, n$. 设 $\hat{\beta}_0, \hat{\beta}_1$ 分别是参数 $\beta_0, \beta_1$ 的最小二乘估计. 

1. （10分）求 $\hat{\beta}_0 + 2\hat{\beta}_1$ 的分布；

2. （5分）对检验水平 $\alpha$，求假设检验问题的拒绝域：

$$
H_0: \beta_0 + 2\beta_1 = 0, \quad H_1: \beta_0 + 2\beta_1 < 0
$$

**解1：**

首先，根据一元回归模型的性质，可知 $\hat{\beta}_0$ 与 $\hat{\beta}_1$ 都服从于正态分布

$$
\hat{\beta}_0 \sim N\left(\beta_0, \sigma^2 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right)\right) \\
\hat{\beta}_1 \sim N\left(\beta_1, \dfrac{{\sigma}^2}{L_{xx}}\right)
$$
$\hat{\beta}_0 + 2\hat{\beta}_1$ 是 $\hat{\beta}_0 , \ \hat{\beta}_1$ 的线性组合，也就是说 $\hat{\beta}_0 + 2\hat{\beta}_1 = \begin{bmatrix}1 & 2\end{bmatrix}\begin{bmatrix}\hat{\beta}_0 \\ \hat{\beta}_1 \end{bmatrix}$. 

设
$$
\boldsymbol{a}= \begin{bmatrix}1\\2\end{bmatrix}, \quad \hat{\boldsymbol{\beta}}= \begin{bmatrix}\hat{\beta}_0 \\ \hat{\beta}_1 \end{bmatrix}
$$
则 $\hat{\boldsymbol{\beta}}$ 服从于多元正态分布，即
$$
\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$
由于其中均值和协方差矩阵分别为（一定不要忘记了协方差）
$$
\boldsymbol{\mu}= \begin{bmatrix}\beta_0\\[2mm]\beta_1\end{bmatrix}, \quad
\boldsymbol{\Sigma} = \left[\begin{matrix} 
\sigma^2 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & -\sigma^2 \dfrac{\bar{x}}{L_xx} \\
-\sigma^2 \dfrac{\bar{x}}{L_xx} & \dfrac{{\sigma}^2}{L_{xx}}
\end{matrix}\right] = \sigma^2
\left[\begin{matrix} 
 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & - \dfrac{\bar{x}}{L_xx} \\
- \dfrac{\bar{x}}{L_xx} & \dfrac{1}{L_{xx}}
\end{matrix}\right]
$$
而对于列向量 $\boldsymbol{a}$ 和二元正态随机列向量 $\boldsymbol{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，有 $\boldsymbol{a}^T \boldsymbol{X} \sim N(\boldsymbol{a}^T \boldsymbol{\mu}, \boldsymbol{ a}^T \Sigma \boldsymbol{a}) $.
$$
\boldsymbol{a}^T \boldsymbol{\mu} = \left[\begin{matrix} 1 & 2\end{matrix}\right] \left[\begin{matrix} {\beta}_0 \\ {\beta}_1\end{matrix}\right] = \beta_0 + 2 \beta_1 \\
\begin{align*}
\boldsymbol{ a}^T \Sigma \boldsymbol{a} 
& = \left[\begin{matrix} 1 & 2\end{matrix}\right] \sigma^2
\left[\begin{matrix} 
 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & - \dfrac{\bar{x}}{L_xx} \\
- \dfrac{\bar{x}}{L_xx} & \dfrac{1}{L_{xx}}
\end{matrix}\right] \left[\begin{matrix} 1 \\ 2\end{matrix}\right]  \\
& = \sigma^2 \left(  \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) +4 \left(- \dfrac{\bar{x}}{L_xx} \right) + \dfrac{4}{L_{xx}} \right) \\
& = \sigma^2 \left( \dfrac{1}{n} + \dfrac{( \bar{x} - 2)^2}{L_{xx}} \right)
\end{align*}
$$
所以，$\hat{\beta}_0 + 2\hat{\beta}_1$ 的分布是正态分布，即
$$
\hat{\beta}_0 + 2\hat{\beta}_1 \sim N \left(\beta_0 + 2 \beta_1,  \sigma^2 \left( \dfrac{1}{n} + \dfrac{( \bar{x} - 2)^2}{L_{xx}} \right) \right)
$$
**解2：**

首先，找出待估计量的良好点估计，显然 $\hat{\beta}_0, \ \hat{\beta}_1$ 分别是 ${\beta}_0 ,\ {\beta}_1$ 的良好点估计.

对立假设 $H_1$ 要求我们 $\beta_0 + 2\beta_1 < 0$，也就是说 $\hat{\beta}_0 + 2 \hat{\beta}_1 $ 偏小时拒绝原假设 $H_0$（单边检验）.

设一个常数 $C$，当 $\hat{\beta}_0 + 2 \hat{\beta}_1  < C$ 时拒绝原假设 $H_0$，接下来就是寻找 $C$.

首先，在第一问中，我们已经知道了$\hat{\beta}_0 +2\hat{\beta}_1$ 的分布是正态分布，所以可以对其标准化，得到

$$
Z := \dfrac{\hat{\beta}_0 + 2 \hat{\beta}_1 - ({\beta}_0 +2 {\beta}_1)}{\sigma \sqrt{ \left( \dfrac{1}{n} + \dfrac{( \bar{x} - 2)^2}{L_{xx}} \right) }} \sim N(0, 1)
$$

这样里面还带有参数 $\sigma$，我们可以用 $\hat{\sigma}$ 估计它，已知

$$
K^2 := \dfrac{n-2}{\sigma^2} \hat{\sigma}^{2} \sim \chi^2 (n-2)
$$

又因为 $\sigma^2$ 与  $\hat{\beta}_0, \ \hat{\beta}_1$ 都独立，从而我们可以够造出 $t$ 分布，即

$$
T := \dfrac{Z}{\sqrt{K^2 / (n-2)}} =  
\dfrac{\hat{\beta}_0 + 2 \hat{\beta}_1 - ({\beta}_0 +2 {\beta}_1)}{\hat{\sigma} \sqrt{ \left( \dfrac{1}{n} + \dfrac{( \bar{x} - 2)^2}{L_{xx}} \right) }}
\sim t(n-2)
$$

这样我们就可以知道

$$
\begin{align*}
P( \hat{\beta}_0 +2 \hat{\beta}_1 < C) &= 
P \left(  \frac{\hat{\beta}_0 +2\hat{\beta}_1 - ({\beta}_0 +2 {\beta}_1)}{\hat{\sigma} \sqrt{ \left( \frac{1}{n} + \frac{( \bar{x} -2)^2}{L_{xx}} \right) }}  < \frac{C - ({\beta}_0 +2 {\beta}_1)}{\hat{\sigma} \sqrt{ \left( \frac{1}{n} + \frac{(\bar{x} -2)^2}{L_{xx}} \right) }} \right)
\\
&= P_{H_0} \left( T < \frac{C}{\hat{\sigma} \sqrt{ \left( \frac{1}{n} + \frac{(\bar{x} -2)^2}{L_{xx}} \right) }} \Big| \beta_0 + 2\beta_1 =0 \right)
\end{align*}
$$

由此我们找到了 $C$，即 $t$ 分布的分位点
$$
\frac{C}{\hat{\sigma} \sqrt{ \left( \frac{1}{n} + \frac{(\bar{x} -2)^2}{L_{xx}} \right) }} = - t_{\alpha}(n-2) \implies C =  - t_{\alpha}(n-2)\hat{\sigma}\sqrt{ \left( \frac{1}{n} + \frac{(\bar{x} -2)^2}{L_{xx}} \right) }
$$
所以对检验水平 $\alpha$，该检验的拒绝域是

$$
\left\{ \hat{\beta}_0 +2 \hat{\beta}_1 < - t_{\alpha}(n-2)\hat{\sigma}\sqrt{ \left( \frac{1}{n} + \frac{(\bar{x} -2)^2}{L_{xx}} \right) } \right\}
$$

## 2023-2024 A

### 一、计算题（共 10 分）

假设 $X_1, X_2, X_3, X_4$ 独立同分布于正态分布 $N(0, 0.5)$. 

1. （5分）简单推导出 $Y = (X_1 + X_2)^2 + (X_3 + X_4)^2$ 的分布并写出密度函数；

2. （5分）对任意参数 $\lambda > 0$，计算条件概率 $P\{Y > 3\lambda | Y > \lambda\}$. 

**解1：**

由
$$
X_1 + X_2 \sim N(0, 1), \quad X_3 + X_4 \sim N(0, 1)
$$
可得
$$
Y =(X_1 + X_2)^2 + (X_3 + X_4)^2 \sim \chi^2(2)
$$
其概率密度函数为
$$
f_Y (y) = \dfrac{1}{2} e^{-\frac{y}{2}} \quad y>0
$$
**解2：**

> 原答案不小心把 $\cap$ 错打成了 $\cup$，但是答案结果没变，感谢指正. 

$$
\begin{align*}
P(Y>3\lambda | Y>\lambda) &= \dfrac{P(Y>3\lambda \cap Y>\lambda )}{P(Y>\lambda)} \\
&= \dfrac{P(Y>3\lambda )}{P(Y>\lambda)} \\
&= \dfrac{\int_{3\lambda}^{\infty} \frac{1}{2}  e^{-\frac{y}{2}} \mathrm{d}y  }{\int_{\lambda}^{\infty} \frac{1}{2}  e^{-\frac{y}{2}} \mathrm{d}y} \\
&=\dfrac{2e^{-\frac{3}{2}\lambda}}{2e^{-\frac{1}{2}\lambda}} \\
&= e^{-\lambda}
\end{align*}
$$

> [!NOTE]
>
> 由于 $Y\sim χ²(2)=\Gamma(1,\frac{1}{2})=E(\frac{1}{2})$，因此本题也可以用指数分布的无记忆性进行求解. 

### 二、计算题（共 10 分）

假设简单随机样本 $X_1, \ldots, X_n$ 来自正态总体 $N(\theta, 1)$，参数 $\theta$ 的先验分布为 $N(1, 1)$，在如下损失函数下

$$
L(\theta, d) = (\theta - 2d)^2
$$

1. （5分）证明 $\theta$ 的 Bayes 估计是后验分布数学期望的一半；

2. （5分）求出 $\theta$ 的 Bayes 估计. 

**解1：**
$$
E\left(L(\theta, d)\right) = E\left( (\theta - 2d)^2 \right) = E\left( \theta^2  - 4\theta d + 4d^2 \right)
= E(\theta^2) - 4 E(\theta) d+ 4d^2
$$
为了求出的 $E\left(L(\theta, d)\right)$ 极小值，我们对其求导并令其等于 0，即
$$
\dfrac{\mathrm{d} E}{\mathrm{d} d} = -4E(\theta) + 8 d = 0
$$
所以
$$
d = \dfrac{E(\theta)}{2}
$$
即 $\theta$ 的 Bayes 估计是后验分布数学期望的一半. 

**解2：**

样本 $X$ 与参数 $\theta$ 的联合分布是
$$
\begin{align*}
p(X|\theta) \pi(\theta) &= \left( \prod_{i=1}^n \dfrac{1}{\sqrt{2\pi}} e^{-\frac{(x_i-\theta)^2}{2}} \right) \cdot \dfrac{1}{\sqrt{2\pi}} e^{-\frac{(\theta-1)^2}{2}} \\
&\propto \exp \left(-\frac{\sum_{i=1}^n (x_i-\theta)^2}{2} -\frac{(\theta-1)^2}{2}\right) \\
&\propto \exp \left( \dfrac{1}{2} \left( \sum_{i=1}^n x_i^2 -2\theta n \bar{x} + n\theta^2 + \theta^2 -2\theta + 1 \right)  \right)\\
&\propto \exp \left( \dfrac{1}{2} \left( (n+1)\theta^2 - 2(n \bar{x} +1) \theta +C \right)  \right)\\
&\propto \exp \left( -\dfrac{n+1}{2} \left(\theta - \dfrac{n\bar{x}+1}{n+1} \right)^2 \right)
\end{align*}
$$
由于正态分布共轭于正态分布，所以
$$
\theta | X \sim N \left(\dfrac{n \bar{x} + 1}{n+1}, \dfrac{1}{n+1} \right)
$$
由 1 可知，$\theta$ 的 Bayes 估计是后验分布数学期望的一半，所以
$$
\hat{\theta}_B = \dfrac{n \bar{x} + 1}{2(n+1)}
$$

### 三、计算题（共 15 分）

设总体 $X$ 的分布律为 $P(X=0)=2\rho, \ P(X=1)=\rho, \ P(X=2)=1-3\rho$，其中 $\rho \ (0<\rho<1/3)$ 为未知参数，$X_1, \ldots, X_n$ 是总体的一个简单随机样本. 

1. （5分）求 $\rho$ 的矩估计 $\hat{\rho}$；

2. （10分）求 $\hat{\rho}$ 的均方误差. 

**解1：**

首先，我们需要计算总体 $X$ 的期望值（即一阶矩）
$$
\begin{align*}
E(X) &= 0 \cdot P(X=0) + 1 \cdot P(X=1) + 2 \cdot P(X=2) \\
&= 0 \cdot 2\rho + 1 \cdot \rho + 2 \cdot (1 - 3\rho) = \rho + 2 - 6\rho \\
&= 2 - 5\rho
\end{align*}
$$
由于样本均值 $\bar{X}$ 是样本的一阶矩，根据矩估计法，令总体均值等于样本均值
$$
E(X) = \bar{X} \implies 2 - 5\rho = \bar{X}
$$
解得 $\rho$ 的矩估计为
$$
\hat{\rho} = \frac{2 - \bar{X}}{5}
$$
**解2：**

根据均方误差公式
$$
MSE(\hat{\rho}) = D(\hat{\rho}) + \left( E(\hat{\rho}) - \rho \right)^2
$$
首先计算 $E(\hat{\rho})$
$$
E(\hat{\rho}) = E\left(\frac{2 - \bar{X}}{5}\right) = \frac{2 - E(\bar{X})}{5}
$$
由于 $E(\bar{X}) = E(X) = 2 - 5\rho$，代入得
$$
E(\hat{\rho}) = \frac{2 - (2 - 5\rho)}{5} = \frac{5\rho}{5} = \rho
$$
因此，$\hat{\rho}$ 是无偏估计，只需要再计算 $D(\hat{\rho})$
$$
D(\hat{\rho}) = D\left(\frac{2 - \bar{X}}{5}\right) = \frac{D(\bar{X})}{25} \xlongequal {D(\bar{X}) = \frac{D(X)}{n}} \frac{D(X)}{25n}
$$
根据方差的定义 $D(X) = E(X^2) - (E(X) )^2$，其中
$$
E(X^2) = \sum_{x} x^2 \cdot P(X=x) = 0^2 \cdot 2\rho + 1^2 \cdot \rho + 2^2 \cdot (1 - 3\rho) = 4 - 11\rho
$$
所以得到
$$
D(X) = (4-11\rho) - (2-5\rho)^2 = 9\rho-25\rho^2
$$
最后得到 $\hat{\rho}$ 的均方误差
$$
MSE(\hat{\rho}) = D(\hat{\rho}) = \frac{D(X)}{25n} = \dfrac{9\rho-25\rho^2}{25n} = \dfrac{\rho(9-25\rho)}{25n}
$$

### 四、计算分析题（共 15 分）

设 $X \sim N(\mu, \sigma^2)$，$\ln Y \sim N(1, \sigma^2)$，$\mu$ 和 $\sigma^2$ 为未知参数，$X_1, \dots, X_n$ 和 $Y_1, \dots, Y_m$ 分别是来自总体 $X$ 和 $Y$ 的简单随机样本，且相互独立. 求 $\sigma^2$ 的置信度为 0.95 的双侧置信区间. 

**解：**

根据抽样分布定理
$$
K_1^2 : = \dfrac{(n-1) S^2}{\sigma^2} \sim \chi^2(n-1), \quad S^2 = \dfrac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2
$$
对于 $\ln Y$，情况有些特殊，如果我们仿照上面那样构造，会得到一个自由度为 $m-1$ 的卡方分布，但更好的方法是利用已知的均值构造一个自由度为 $m$ 的卡方分布. 

> 实际上，最开始我也第一想法是构造 $(m-1)S_y^2/\sigma^2 \sim \chi^2 (m-1)$，但这没有用上题中已知均值 1，显然这种构造方式得到的卡方分布损失了 1 个自由度. 

$$
\ln Y \sim N(1, \sigma^2) \implies \dfrac{\ln Y - 1}{\sigma} \sim N(0, 1)
$$

如果记 $Z_j = \dfrac{\ln Y_j - 1}{\sigma}$，那么有
$$
K_2^2 := \sum_{j=1}^{m} Z_j^2 = \dfrac{\sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\sigma^2} \sim \chi^2 (m)
$$
根据卡方分布的可加性，有
$$
K^2 := K_1^2 + K_2^2 \sim \chi^2 (n+m-1)
$$
根据概率条件
$$
\begin{align*}
& P\left\{ \chi^2_{1-\alpha/2}(n+m-1) < K^2 < \chi^2_{\alpha/2}(n+m-1)\right\} \\ 
&= 
P\left\{ \chi^2_{1-\alpha/2}(n+m-1) < \dfrac{(n-1) S^2 + \sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\sigma^2} < \chi^2_{\alpha/2}(n+m-1)\right\} \\ 
&= P\left\{ \dfrac{(n-1) S^2 + \sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\chi^2_{\alpha/2}(n+m-1)}  < \sigma^2 < \dfrac{(n-1) S^2 + \sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\chi^2_{1-\alpha/2}(n+m-1)}\right\}
\end{align*}
$$
把 $\alpha=0.05$ 代入可得 $\sigma^2$ 的置信度为 0.95 的双侧置信区间
$$
\left( \dfrac{(n-1) S^2 + \sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\chi^2_{0.025}(n+m-1)} ,\ \dfrac{(n-1) S^2 + \sum_{j=1}^{m} \left(\ln Y_j - 1\right)^2}{\chi^2_{0.975}(n+m-1)} \right)
$$

### 五、计算题（共 10 分）

某灯泡厂用甲乙两种不同配料方案制成灯丝，在生产的灯泡中随机抽取测得使用寿命如下表

| 灯丝配料方案 |       灯泡寿命（小时）       |
| :----------: | :--------------------------: |
|      甲      | 1500, 1510, 1515, 1470, 1490 |
|      乙      |    1480, 1600, 1530, 1550    |

检验灯丝配料方案对灯泡寿命是否有显著影响（$\alpha = 0.05$）.

**解：**

> 题目没有指出灯泡寿命服从于什么分布，且没有提供算好的样本均值和样本方差，显然不是两个正态总体的均值检验，只能选择**秩和检验**，也就是检验甲和乙两种灯泡寿命的分布是否相近. 

| 秩     | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 样本值 | 1470 | 1480 | 1490 | 1500 | 1510 | 1515 | 1530 | 1550 | 1600 |
| 组别   | 甲   | 乙   | 甲   | 甲   | 甲   | 甲   | 乙   | 乙   | 乙   |

秩和检验统计量（选择样本数量最小的那个样本组）
$$
W = 2 + 7 + 8 +9 =26
$$
拒绝域为
$$
\left\{W \leq T_1 \right\} \cup\left\{ W\geq T_2\right\}
$$
当 $\alpha=0.05, \ n_1 = 4, \ n_2 = 5$ 查表得 $T_{1} = 13, \ T_2 = 27$.

> 教材上的秩和检验表里，缺了 $n_1=4$ 这个数字. 

因为 $T_1 < W < T_2$，所以不拒绝原假设，认为灯丝配料方案对灯泡寿命没有显著影响. 

### 六、计算题（共 15 分）

设 $X_1, X_2, \ldots, X_{16}$ 是来自正态总体 $X \sim N(\mu, 4)$ 的简单随机样本，考虑假设检验问题：

$$
H_0: \mu = 3, \quad H_1: \mu = 4
$$
若 $H_0$ 的否定域为

$$
W = \{(X_1, \ldots, X_{16}): \overline{X} > 3.8\}
$$

1. （10分）求犯两类错误的概率 $\alpha$ 与 $\beta$；

2. （5分）在显著性水平 0.01 下，求 $H_0: \mu = 3$，$H_1: \mu = 4$ 的否定域. 

**解1：**

第一类错误是指当原假设成立时，拒绝原假设时的条件概率
$$
\begin{align*}
\alpha &= P\left\{ \bar{X} > 3.8 \Big| \mu = 3 \right\} \\
&= P\left\{ \frac{\bar{X} -3}{\sqrt{\frac{1}{4}}} > \frac{3.8 -3}{\sqrt{\frac{1}{4}}} \Big| \mu = 3 \right\} \\
& \xlongequal{Z_1 = \frac{\bar{X} -3}{\sqrt{\frac{1}{4}}} \sim N(0,1)} P(Z_1 > 1.6) \\
&= 1 - \Phi(1.6) \\
&= 1-0.94520\\
&=0.0548
\end{align*}
$$
第二类错误是指当对立假设成立时，拒绝对立假设时的条件概率
$$
\begin{align*}
\beta &= P\left\{ \bar{X} \leq 3.8 \Big| \mu = 4 \right\} \\
&= P\left\{ \frac{\bar{X} -4}{\sqrt{\frac{1}{4}}} \leq \frac{3.8 -4}{\sqrt{\frac{1}{4}}} \Big| \mu = 4 \right\} \\
& \xlongequal{Z_2 = \frac{\bar{X} -4}{\sqrt{\frac{1}{4}}} \sim N(0,1)} P(Z_2 \leq -0.4) \\
&= 1 - \Phi(0.4) \\
&= 1-0.6554\\
&=0.3446
\end{align*}
$$
**解2：**

由于 $\bar{X}$ 是参数 $\mu$ 的良好点估计，所以当 $\bar{X}$ 偏大时我们拒绝原假设，有
$$
P \left\{ \bar{X} > C\right\} = P \left\{ \frac{\bar{X} -\mu}{\sqrt{\frac{1}{4}}} > \frac{C -\mu}{\sqrt{\frac{1}{4}}} \right\} = P \left\{ Z > \frac{C - 3}{\sqrt{\frac{1}{4}}} \Big| \mu=3 \right\} = \alpha
$$
即
$$
\frac{C - 3}{\sqrt{\frac{1}{4}}} = u_{0.01} \implies C = 3 + \frac{1}{2} u_{0.01} = 3 + \frac{1}{2} \times 2.33 = 4.165
$$
所以否定域为
$$
\left\{ \bar{X} > 4.165 \right\}
$$

### 七、计算分析题（共 10 分）

下表中的数据描述了对 25 名发烧 38°C 或以上的受试者服用五种不同品牌的头痛片所提供的缓解小时数：

$$
\begin{align*}
A_1: & \quad 5.2, \, 4.5, \, 8.1, \, 6.2, \, 3.0 \\
A_2: & \quad 9.1, \, 7.1, \, 8.2, \, 6.0, \, 9.1 \\
A_3: & \quad 3.1, \, 5.7, \, 2.1, \, 3.0, \, 7.1 \\
A_4: & \quad 2.4, \, 3.3, \, 4.1, \, 1.2, \, 4.0 \\
A_5: & \quad 7.1, \, 6.6, \, 9.3, \, 4.4, \, 7.6 \\
\end{align*}
$$

假设上述数据满足单因素方差分析模型的条件，在显著性水平 $\alpha = 0.05$ 下，检验不同品牌的头痛片的疗效有无显著差异. 

提示：分组均值与样本方差，$A_1: \overline{x}_1 = 5.4, s_1^2 = 14.54/4$；$A_2: \overline{x}_2 = 7.9, s_2^2 = 7.22/4$；$A_3: \overline{x}_3 = 4.2, s_3^2 = 17.72/4$；$A_4: \overline{x}_4 = 3.0, s_4^2 = 5.9/4$；$A_5: \overline{x}_5 = 7.0, s_5^2 = 12.58/4$；全部数据的均值与样本方差：$\overline{x} = 5.5, s^2 = 137.76/24$. 

**解：**

原假设与对立假设
$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_5 , \quad H_1: \exist i, \ j, \ \beta_i \neq \beta_j
$$
计算并构造 $F$ 统计量
$$
TSS = \sum_{i=1}^{r} \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2 = (n-1)s^2 = (25-1) \times \frac{137.76}{24} = 137.76 \\
RSS = \sum_{i=1}^{r} (n_i - 1) s_i^2 = 4 \times \frac{14.54}{4} + 4 \times \frac{7.22}{4} + 4 \times \frac{17.72}{4} + 4 \times \frac{5.9}{4}+ 4 \times \frac{12.58}{4} = 57.96 \\
CSS = TSS -RSS = 137.76-57.96 = 79.8
$$
因此统计量
$$
F = \dfrac{n-r}{r-1} \frac{CSS}{RSS} \sim F(r-1, n-r)
$$
即 $F \sim F(4, 20)$. 
$$
F = \frac{20}{4} \times \frac{79.8}{57.96} = 6.88
$$
查表可知 $F_{\alpha}(r-1,n-r) = F_{0.05}(4,20)=2.87$. 

因为 $F=6.88>2.87=F_{0.05}(4,20)$，所以拒绝原假设，认为不同品牌的头痛片的疗效有显著差异. 

### 八、计算题（共 15 分）

考虑一元回归模型 $y = \beta_0 + \beta_1 x + \varepsilon$，$\varepsilon \sim N(0, \sigma^2)$，今对不同的 $x$ 值，对 $y$ 进行观察，得 10 对数据，并根据这些数据对计算得

$$
\overline{x} = 3.9, \overline{y} = 3.9, L_{xx} = 26.4, L_{xy} = 20.9, L_{yy} = 30.9
$$

1. （10分）求回归方程 $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$；

2. （5分）在显著性水平 $\alpha = 0.05$ 下，检验假设：

$$
H_0: \beta_1 \geq 1, \quad H_1: \beta_1 < 1.
$$

**解1：**
$$
\hat{\beta}_1 = \dfrac{L_{xy}}{L_{xx}} = \dfrac{20.9}{26.4} = 0.79 \\
\hat{\beta}_0 = \bar{y} - \hat{\beta_1}\bar{x} = 3.9-0.79\times 3.9 = 0.82 \\
\hat{y} = 0.82 + 0.79 x
$$
**解2：**

由 $\hat{\beta}_1 \sim N \left( \beta_1 , \dfrac{\sigma^2}{L_{xx}} \right)$，可以标准化它
$$
Z := \dfrac{\hat{\beta}_1 - \beta_1}{\sqrt{\frac{\sigma^2}{L_{xx}}}} \sim N(0, 1)
$$
这里还含有未知参数 $\sigma$，我们需要用 $\hat{\sigma}$ 来估计它，由
$$
K^2 : = \dfrac{n-2}{\sigma^2} \hat{\sigma}^2 \sim \chi^2 (n-2)
$$
这样我们凑出 $t$ 分布就能消去未知参数，即
$$
T:= \dfrac{Z}{\sqrt{K^2/(n-2)}} = \dfrac{\hat{\beta}_1 - \beta_1}{\hat{\sigma}/\sqrt{L_{xx}}} \sim t(n-2)
$$
在原假设成立的条件下，当 $\hat{\beta}_1$ 偏小时我们拒绝原假设，所以
$$
P \left\{ \hat{\beta}_1 < C \right\} = P \left\{ \dfrac{\hat{\beta}_1 - \beta_1}{\hat{\sigma}/\sqrt{L_{xx}}} < \dfrac{C - \beta_1}{\hat{\sigma}/\sqrt{L_{xx}}} \right\} = P_{H_0} \left\{ T < \dfrac{C - 1}{\hat{\sigma}/\sqrt{L_{xx}}} \Big| \beta_1 \geq 1 \right\}
$$
这是左边检验，这样我们找到 $C$
$$
\dfrac{C - 1}{\hat{\sigma}/\sqrt{L_{xx}}} = -t_{\alpha} (n-2) \implies C = -\dfrac{\hat{\sigma}}{\sqrt{L_{xx}}} t_{\alpha} (n-2) + 1
$$
所以拒绝域为
$$
\left\{\hat{\beta}_1 < -\dfrac{\hat{\sigma}}{\sqrt{L_{xx}}} t_{\alpha} (n-2) + 1 \right\}
$$
其中，
$$
\hat{\sigma}^2 = \dfrac{1}{n-2} \left( L_{yy} - \hat{\beta}_1 L_{xy} \right) = \dfrac{1}{10-2} \times (30.9-0.79 \times 20.9) = 1.80
$$
当 $\alpha = 0.05$，查表可得 $t_{\alpha}(n-2) = t_{0.05}(8) = 1.8595$. 

所以，代入具体数值，拒绝域具体为
$$
\left\{\hat{\beta}_1 < 0.51\right\}
$$
因为 $\hat{\beta}_1 = 0.79 > 0.51 = C$，所以不能拒绝原假设. 

## 2022-2023 A

### 一、计算题（共 10 分）

设 $X \sim \chi^2(2)$，若常数 $C$ 满足 $P(X > C) = \alpha$，证明 $C = -2\ln \alpha$. 

（提示：利用 Gamma 分布的性质）

**解：**

卡方分布是一种 Gamma 分布，所以由 $X \sim \chi^2(2)$ 可知 $X\sim \Gamma \left(1, \dfrac{1}{2}\right)$. 

如果我们想要更加简化运算，可以把随机变量 $X$ 凑成一个服从指数分布的随机变量 $Y$，这是因为指数分布也是一种 Gamma 分布. 
$$
Y: = \dfrac{1}{2} X \sim \Gamma(1, 1) \implies Y = \dfrac{1}{2} X \sim E(1)
$$
由
$$
P(X > C) = 1 - P(X \leq C) = \alpha \\ 
\implies P(X \leq C) = P\left(\dfrac{1}{2}X \leq \dfrac{1}{2}C\right)= P\left(Y\leq \dfrac{1}{2}C\right) = 1 - \alpha
$$
可以利用指数分布的累计分布函数，也就是
$$
\int_0^{\frac{c}{2}} e^{-x} \mathrm{d}x = 1-e^{-x} \Big|_{0}^{\frac{c}{2}} = 1 - e^{-\frac{c}{2}} = 1 - \alpha
$$
从而解得
$$
C = -2 \ln \alpha
$$

### 二、计算题（共 10 分）

简单随机样本 $X_1, X_2, \cdots, X_{10}$ 来自泊松总体 $P(\lambda)$，参数 $\lambda$ 的先验分布为 Gamma 分布 $\Gamma(2, 10)$，样本均值 $\bar{x} = 1.3$ 利用 $\lambda$ 的后验分布构造出 $\lambda$ 的一个置信度为 0.95 的 Bayes 区间估计. 

**解：**

参数 $\lambda$ 的分布为
$$
\pi(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta \lambda} \quad (\lambda > 0)
$$
样本 $X$ 与参数 $\lambda$ 的联合分布为
$$
p(x, \lambda) = p(x|\lambda)\pi(\lambda) = 
\left( \prod_{i=1}^{n} \dfrac{\lambda^{x_i}}{x_i !} e^{-\lambda} \right) \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta \lambda} 
\propto \lambda^{\sum {x_i} + \alpha - 1} e^{ - n \lambda-\beta \lambda} \\
$$
因此 Gamma 分布共轭于泊松分布
$$
\lambda | X \sim \Gamma (\alpha + n\bar{x}, n+ \beta)
$$
其中，$\alpha = 2$，$\beta = 10$，$n=10$，$\bar{x}=1.3$，所以
$$
\lambda | X \sim \Gamma(15, 20)
$$
我们知道 Gamma 分布可以用来构造出卡方分布，所以
$$
40 \lambda | X \sim \chi^2(30)
$$
所以
$$
P\left\{ c < \lambda < d \right\} = 
P\left\{ \chi^2_{1-a/2}(30) < 40 \lambda < \chi^2_{a/2}(30) \right\} = 
P\left\{ \dfrac{\chi^2_{1-a/2}(30)}{40} <  \lambda < \dfrac{\chi^2_{a/2}(30)}{40} \right\}
$$
即 $\lambda$ 的区间估计为
$$
\left( \dfrac{\chi^2_{1-a/2}(30)}{40}, \dfrac{\chi^2_{a/2}(30)}{40} \right)
$$
具体地，由于 $1-a = 0.95$，所以 $a = 0.05$，查表得 $\chi^2_{0.975}(30)=16.791$，$\chi^2_{0.025}(30)=46.979$，代入数据可得 $\lambda$ 的区间估计为
$$
(0.42, 1.17)
$$

### 三、计算题（共 15 分）

总体 $ X $ 的概率密度函数为

$$
f(x) = 
\begin{cases} 
\dfrac{2x}{\theta} e^{-\frac{x^2}{\theta}}, & x > 0, \\
0, & x \leq 0.
\end{cases}
$$
其中 $\theta \ (\theta > 0)$ 为未知参数，$X_1, X_2, \dots, X_n$ 是总体的一个样本.

1. （10 分）求 $\theta$ 的极大似然估计量 $\hat{\theta}$；
2. （5 分）求 $\hat{\theta}$ 的均方误差.

**解1：**

似然函数
$$
L(\theta) = \prod_{i=1}^n f(x_i) = \prod_{i=1}^n \left( \frac{2x_i}{\theta} e^{-\frac{x_i^2}{\theta}} \right) = \left(  2^n  \prod_{i=1}^n X_i \right) \frac{1}{\theta^n} e^{-\frac{\sum_{i=1}^n x_i^2}{\theta}}
$$
对似然函数取对数
$$
\ln L(\theta) = n \ln 2 + \sum_{i=1}^n \ln X_i - n \ln \theta - \frac{\sum_{i=1}^n X_i^2}{\theta}
$$
对 $\theta$ 求导并令导数为零
$$
\frac{\partial \ln L(\theta)}{\partial \theta} = -\frac{n}{\theta} + \frac{\sum_{i=1}^n X_i^2}{\theta^2} = 0
$$
解得
$$
\hat{\theta} = \frac{\sum_{i=1}^n X_i^2}{n}
$$
**解2：**
$$
MSE(\hat{\theta}) = D(\hat{\theta}) + \left( E(\hat{\theta}) - \theta \right)^2
$$
先计算 $E(\hat{\theta})$
$$
E(\hat{\theta}) = E\left[ \frac{1}{n} \sum_{i=1}^n X_i^2 \right] = \frac{1}{n} \sum_{i=1}^n E(X_i^2) = E(X^2)
$$
然后计算 $E(X^2)$
$$
\begin{align*}
E(X^2) &= \int_0^\infty x^2 \cdot \frac{2x}{\theta} e^{-\frac{x^2}{\theta}} \mathrm{d} x \\
&= \dfrac{2}{\theta} \int_0^\infty x^3 e^{-\frac{x^2}{\theta}} \mathrm{d} x \\
& \xlongequal{t=\frac{x^2}{\theta}} \dfrac{2}{\theta} \int_0^\infty \left(\sqrt{\theta t}\right)^3e^{-t} \mathrm{d}\sqrt{\theta t}\\
&= \dfrac{1}{\theta} \underbrace{ \int_0^\infty t e^{-t} \mathrm{d}t }_{\text{Gamma 积分}} \\
&=  \theta \Gamma{(2)} \\
&= \theta
\end{align*}
$$
所以 $E(\hat{\theta}) = \theta$，也说明了 $\hat{\theta}$ 是无偏估计. 

接下来计算 $D(\hat{\theta})$
$$
D(\hat{\theta}) = D\left( \frac{1}{n} \sum_{i=1}^n X_i^2 \right) = \frac{1}{n^2} \sum_{i=1}^n D(X_i^2) = \frac{D(X^2)}{n} \\
D(X^2) = E(X^4) - \left(E(X^2)\right)^2
$$
因此，我们需要计算 $E(X^4)$
$$
\begin{align*}
E(X^4) &= \int_0^\infty x^4 \cdot \frac{2x}{\theta} e^{-\frac{x^2}{\theta}} \mathrm{d} x \\
& \xlongequal{t=\frac{x^2}{\theta}} \int_0^\infty \theta^2 t^2 \cdot \frac{2\sqrt{\theta t}}{\theta} e^{-t} \cdot \frac{\sqrt{\theta}}{2\sqrt{t}} \mathrm{d}t \\
&= \theta^2 \underbrace{  \int_0^\infty t^2 e^{-t} \mathrm{d}t }_{\text{Gamma 积分}}\\
&= \theta^2 \Gamma(3) \\
&= 2\theta^2
\end{align*}
$$
所以 $D(X^2) = 2\theta^2 - \theta^2 = \theta^2$. 

所以 $\hat{\theta}$ 的均方误差为
$$
MSE(\hat{\theta}) = \dfrac{\theta^2}{n}
$$

### 四、计算分析题（共 10 分）

某人群中不同等级肺活量的人数有如下统计结果：

| 性别 | 一级 | 二级 | 三级 |
| ---- | ---- | ---- | ---- |
| 男性 | 10   | 12   | 18   |
| 女性 | 20   | 28   | 12   |

问：性别和肺活量等级是否有关？并说明理由 $(\alpha = 0.1)$. （要求检验统计量的观测值的有效数字保留到小数点后两位）

**解：**

| 性别 | 一级 | 二级 | 三级 | 总计 |
| ---- | ---- | ---- | ---- | ---- |
| 男性 | 10   | 12   | 18   | 40   |
| 女性 | 20   | 28   | 12   | 60   |
| 总计 | 30   | 40   | 30   | 100  |

原假设与对立假设
$$
H_0 : \forall i,\ j, \ p_{ij}=p_i p_j \quad H_1: \exist i,\ j,\ p_{ij} \neq p_i p_j
$$
统计量
$$
\begin{align*}
K^2 &= n \left[  \sum_{i=1}^{s} \sum_{j=1}^{t} \dfrac{n_{ij}^2}{n_{i*} n _{*j}} -1 \right] \\
&= 100 \left[ \left( \frac{10^2}{40 \times 30} + \frac{12^2}{40 \times 40} + \frac{18^2}{40 \times 30} \right) + \left( \frac{20^2}{60 \times 30} + \frac{28^2}{60 \times 40} + \frac{12^2}{60 \times 30} \right) - 1 \right]
\\
& = 7.22
\end{align*}
$$
拒绝域为
$$
\left\{ K^2 > \chi^2_{\alpha} \left( (s-1) (t-1) \right) \right\}
$$
其中 $\chi^2_{\alpha} \left( (s-1) (t-1) \right) = \chi^2_{0.1} \left(2 \right) = 4.605$. 

因为 $K^2 = 7.22 > 4.605 =  \chi^2_{0.1} \left(2 \right) $，所以拒绝原假设 $H_0$，认为性别与肺活量有关. 

### 五、计算题（共 15 分）

设 $X_1, X_2, \dots, X_m$ 和 $Y_1, Y_2, \dots, Y_n$ 分别为总体 $X \sim N(\mu_1, \sigma^2)$ 和 $Y \sim N(\mu_2, \sigma^2)$ 的相互独立的样本，$\mu_1, \mu_2, \sigma$ 都是未知参数. 利用全部样本求 $\sigma$ 的置信度为 0.95 的单侧置信上限 $\bar{\sigma}$，即 $P\{\sigma < \bar{\sigma}\} = 0.95$. 

**解：**

根据抽样分布定理，有
$$
\dfrac{(m-1)S_1^2}{\sigma^2} \sim \chi^2 (m-1) \quad \dfrac{(n-1)S_2^2}{\sigma^2} \sim \chi^2 (n-1)
$$
题目要求利用全部样本，而卡方分布具有可加性，所以
$$
K^2 := \frac{(m-1)S_1^2 + (n-1)S_2^2}{\sigma^2} \sim \chi^2(m+n-2)
$$
根据题意有
$$
P \left\{ K^2 > \chi^2_{1-\alpha} (m+n-2) \right\} = P \left\{ \frac{(m-1)S_1^2 + (n-1)S_2^2}{\sigma^2} > \chi^2_{1-\alpha} (m+n-2) \right\} \\
= P \left\{ \sigma^2 < \frac{(m-1)S_1^2 + (n-1)S_2^2}{\chi^2_{1-\alpha} (m+n-2)} \right\} = 1-\alpha = 0.95
$$
因此
$$
\sigma < \sqrt{\frac{(m-1)S_1^2 + (n-1)S_2^2}{\chi^2_{0.95} (m+n-2)}}
$$
### 六、计算题（共 15 分）

设 $X_1, X_2, \cdots, X_{16}$ 是来自正态总体 $X \sim N(\mu, 9)$ 的简单随机样本，考虑假设检验问题：

$$
H_0: \mu = 5, \quad H_1: \mu \neq 5
$$

1. （10 分）在显著性水平 $\alpha = 0.05$ 下，求上述检验问题的拒绝域；
2. （5 分）在 $\mu = 6$ 时，推导 1 问中的检验犯第二类错误的概率. （用标准正态分布的分布函数表示）

**解1：**

检验统计量
$$
Z = \frac{\bar{X} - 5}{3/\sqrt{16}} = \frac{\bar{X} - 5}{0.75}
$$
双侧检验的拒绝域
$$
|Z| > u_{0.025}
$$
即
$$
\left\{ \frac{\bar{X} - 5}{0.75} < -u_{0.025}  \right\} \cup \left\{  \frac{\bar{X} - 5}{0.75} > u_{0.025}\right\} \\
\implies \left\{\bar{X} < 5 - 0.75 u_{0.025} \right\} \cup \left\{  \bar{X} > 5 + 0.75 u_{0.025}\right\}
$$
查表得 $u_{0.025} = 1.96$，因此在显著性水平 $\alpha = 0.05$ 下，上述检验问题的拒绝域为
$$
\left\{\bar{X} < 3.53 \right\} \cup \left\{  \bar{X} > 6.47\right\}
$$
**解2：**

第二类错误是指当 $\mu = 6$ 时，未能拒绝 $H_0$，即

$$
p_{\mathrm{II}} = P\left( 5 - 0.75 u_{0.025} \leq \bar{X} \leq 5 + 0.75 u_{0.025} \;\Big|\; \mu = 6 \right)
$$
在 $ \mu = 6 $ 时，$\bar{X} \sim N(6, 0.75^2)$，因此

$$
P\left( \frac{5 - 0.75 u_{0.025} - 6}{0.75} \leq Z \leq \frac{5 + 0.75 u_{0.025} - 6}{0.75} \right)\\
= P\left( \frac{-1 - 0.75 u_{0.025}}{0.75} \leq Z \leq \frac{-1 + 0.75 u_{0.025}}{0.75} \right)\\
= P\left( -\frac{1}{0.75} - u_{0.025} \leq Z \leq -\frac{1}{0.75} + u_{0.025} \right) \\
= P\left( -\frac{4}{3} - u_{0.025} \leq Z \leq -\frac{4}{3} + u_{0.025} \right)
$$

所以犯第二类错误的概率为
$$
p_{\mathrm{II}}= \Phi\left( -\frac{4}{3} + u_{0.025} \right) - \Phi\left( -\frac{4}{3} - u_{0.025} \right) = \Phi(0.6267) - \Phi(-3.293)
$$

### 七、计算分析题（共 10 分）

某公司对一种产品设计了三种新包装，为考察哪种包装最受欢迎，选了地段繁华程度相似、规模相近的商店做试验. 在试验期内，各店货架排放位置、空间都相同，营业员的促销方式也相同，经过一段时间，记录其销量数据：

- 包装类型 $A_1: 12, 18, 15, 13, 12$
- 包装类型 $A_2: 19, 17, 21, 24, 29$
- 包装类型 $A_3: 28, 27, 22, 26, 32$

假设上述数据满足单因素方差分析模型的条件，在显著性水平 $\alpha = 0.01$ 下，检验不同包装类型下的销量有无显著差异？

提示：分组均值与样本方差，$A_1: \bar{x}_1 = 14, s_1^2 = 26/4$；$A_2: \bar{x}_2 = 22, s_2^2 = 88/4$；$A_3: \bar{x}_3 = 27, s_3^2 = 52/4$；全部数据的均值与样本方差：$\bar{x} = 21, s^2 = 596/14$. 

**解：**
$$
TSS = \sum_{i=1}^{r} \sum_{j=1}^{n_i} (y_{ij} - \bar{y})^2 = (n-1)s^2 = 14 \times \dfrac{596}{14}= 596 \\
RSS = \sum_{i=1}^{r} (n_i-1)s^2_i = 4 \times \dfrac{26}{4} + 4 \times \dfrac{88}{4}  + 4 \times \dfrac{52}{4} =166 \\
CSS = TSS - RSS = 596-166=430
$$
原假设与对立假设
$$
H_0:\beta_0 = \beta_1 = \beta_2, \quad H_1: \beta_0, \ \beta_1, \ \beta_2 中至少有一组不相等
$$
计算并构造 $F$ 统计量
$$
F = \dfrac{n-r}{r-1} \cdot \dfrac{CSS}{RSS} \sim F(r-1, n-r)
$$
即 $F \sim F(2, 12)$. 
$$
F = \dfrac{12}{2} \times \dfrac{430}{166} = 15.54
$$
> 教材上查不到的 : (

查表知 $F_{0.01} (2,12)=6.92$，因为 $F=15.54>6.92=F_{0.01} (2,12)$，所以拒绝原假设 $H_0$，认为不同包装类型下的销量有显著差异. 

### 八、计算题（共 15 分）

考虑一元回归模型 $ \begin{cases} y = \beta_0 + \beta_1 x + \varepsilon, \\ \varepsilon \sim N(0, \sigma^2) \end{cases} $，$\beta_0$ 为已知参数，记样本观测数据 $(x_i, y_i), i = 1, 2, \ldots, n$. 

1. （10 分）求未知参数 $\beta_1$ 的最小二乘估计 $\hat{\beta}_1$；
2. （5 分）求 $\hat{\beta}_1$ 的方差. 

> 此处答案可详见教材 205 ~ 207 页，利用线性代数知识和最小二乘方法（求导算极值）. 

## 2021-2022 A

### 一、计算题（共 10 分）

设简单随机样本 $X_{1}, X_{2}, \cdots, X_{n}$ 来自正态总体 $N(\mu, 25)$，若 $(\bar{X} - 2, \bar{X}+2)$ 是 $\mu$ 的置信度为 $0.99$ 的置信区间，样本容量 $n$ 至少应为多少？

**解：**

在总体方差 $\sigma_0^2 = 25$ 已知条件下，总体均值 $\mu$ 的 $1-\alpha = 0.99$ 区间估计为
$$
\left( \bar{X} - u_{\alpha/2} \dfrac{\sigma_0}{\sqrt{n}}, \bar{X} + u_{\alpha/2} \dfrac{\sigma_0}{\sqrt{n}} \right)
$$
已知 $(\bar{X}-2, \bar{X}+2)$ 是 $\mu$ 的置信度为 $1-\alpha = 0.99$ 的置信区间，所以有
$$
u_{0.01/2} \dfrac{\sigma_0}{\sqrt{n}} = 2
$$
其中，$u_{0.01/2} = u_{0.005} = 2.58,\ \sigma_0 = 5$. 

代入解得 $n \approx 41.512$. 

因此，样本容量 $n$ 至少应为 $42$. 

### 二、计算题（共 10 分）

两个总体 $X \sim F(x) $ 和 $Y \sim F(y-\Delta)$ 的两独立样本观测值如下：

$$
X: 29, 33, 30, 32, 27 \\
Y: 38, 28, 35, 31, 36, 34
$$

其中 $F(x)$ 与 $F(y-\Delta)$ 分别为两总体的分布函数，在显著性水平 $\alpha = 0.05$ 下，检验假设

$$
H_0: \Delta = 0, \quad H_1: \Delta \neq 0
$$
**解：**

对混合后的样本从小到大进行排序

| 秩     | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 样本值 | 27   | 28   | 29   | 30   | 31   | 32   | 33   | 34   | 35   | 36   | 38   |
| 分组   | X    | Y    | X    | X    | Y    | X    | X    | Y    | Y    | Y    | Y    |

因为 $X$ 的样本数量有 5 个，$Y$ 的样本数量有 6 个，选择样本数量更少的 $X$ 样本求秩和（把分组为 $X$ 的顺序号求和）
$$
W = 1 +3 +4 +6+ 7=21
$$
原假设 $H_0$ 的拒绝域为
$$
\left\{W \leq T_1 \right\} \cup\left\{ W\geq T_2\right\}
$$
当 $n_1=5, \ n_2 =6, \ \alpha=0.05$ 时，查秩和检验表可得 $T_1 = 20, \ T_2 =40$. 

因为 $T_1 <W < T_2$，所以接受原假设 $H_0$. 

### 三、计算题（共 10 分）

一位经济学家对生产电子设备的企业收集了一年内生产力提高指数（用 0 到 100 内的数表示），并按过去三年在研发上的平均投入分为三个水平：
$$
A_1: 投入少 ,\quad A_2: 投入中等,\quad A_3: 投入多
$$
生产力提高的指数如下所示
$$
A_1: 6,\ 4, \ 5,\ 5,\ 5,\ 5,\  6,\ 4,\  5,\ 5 \\
A_2: 6,\ 6,\ 7,\ 8,\ 6,\ 8,\ 8,\ 8,\ 6,\ 7 \\
A_3: 11,\ 12,\ 12,\ 11,\ 14,\ 12,\ 12,\ 11,\ 12,\ 13
$$
假设上述数据满足方差分析模型的条件，在显著性水平 $\alpha = 0.05$ 下，检验不同投入水平下生产力提高指数有无显著差异？

提示：分组均值与样本方差 $A_1: \bar{x}_1=5, s_1^2=4/9; \ A_2: \bar{x}_2 = 7, s_2^2 = 8/9; \ A_3: \bar{x}_3 = 12, \ s_3^2 = 8/9$. 全部数据的均值与样本方差 $\bar{x}=8,\ s^2 = 280/29$. 

**解：**

原假设与对立假设
$$
H_0: \beta_0 = \beta_1 = \beta_2, \quad H_1: \beta_0, \ \beta_1, \ \beta_2 中至少有一组不相等
$$
计算并构造 $F$ 统计量
$$
TSS = \sum_{i=1}^{r} \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2 = (n-1)s^2 = (30-1) \times \frac{280}{29} = 280 \\
RSS = \sum_{i=1}^{r} (n_i - 1) s_i^2 = 9 \times \frac{4}{9} + 9 \times \frac{8}{9} + 9 \times \frac{8}{9} =20 \\
CSS = TSS -RSS = 280-20 = 260
$$
因此统计量
$$
F = \dfrac{n-r}{r-1} \frac{CSS}{RSS} \sim F(r-1, n-r)
$$
即 $F \sim F(2, 27)$. 
$$
F = \frac{27}{2} \times \frac{260}{20} = 175.5
$$
查表可知 $F_{\alpha}(r-1,n-r) = F_{0.05}(2,27)=3.35$. 

由于 $F=175.5≫F_{0.05}(2,27)=3.35$，我们拒绝原假设，认为不同投入水平下的生产力提高指数存在显著差异. 

### 四、计算题（共 15 分）

在一元回归模型中，收集了一组数据 $(X_i, Y_i), i = 1, 2, \dots, n$，设 $\hat{\beta}_1$ 与 $\hat{\beta}_0$ 分别是 $\beta_1$ 与 $\beta_0$ 的最小二乘估计. 

1. （10分）求 $\hat{\beta}_1 - 3\hat{\beta}_0$ 的分布；
2. （5分）对显著性水平 $\alpha$，求假设检验问题 $H_0: \beta_1 = 3\beta_0 ,\ H_1: \beta_1 \ne 3\beta_0$ 的拒绝域. 

**解1：**

首先，根据一元回归模型的性质，可知 $\hat{\beta}_1$ 与 $\hat{\beta}_0$ 都服从于正态分布
$$
\hat{\beta}_0 \sim N\left(\beta_0, \sigma^2 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right)\right) \\
\hat{\beta}_1 \sim N\left(\beta_1, \dfrac{{\sigma}^2}{L_{xx}}\right)
$$
根据题意，$\hat{\beta}_1 - 3\hat{\beta}_0$ 就是 $\hat{\beta}_0$ 和 $\hat{\beta}_1$ 的线性组合，即
$$
\hat{\beta}_1 - 3\hat{\beta}_0 = \left[\begin{matrix} -3 & 1\end{matrix}\right] \left[\begin{matrix} \hat{\beta}_0 \\ \hat{\beta}_1\end{matrix}\right]
$$
而对于列向量 $\boldsymbol{a}$ 和二元正态随机列向量 $\boldsymbol{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，有 $\boldsymbol{a}^T \boldsymbol{X} \sim N(\boldsymbol{a}^T \boldsymbol{\mu}, \boldsymbol{ a}^T \Sigma \boldsymbol{a}) $. 

如果我们记 $\boldsymbol{\mu} = \left[\beta_0, \beta_1 \right]^T$
$$
\boldsymbol{a}^T \boldsymbol{\mu} = \left[\begin{matrix} -3 & 1\end{matrix}\right] \left[\begin{matrix} {\beta}_0 \\ {\beta}_1\end{matrix}\right] = \beta_1 - 3 \beta_0
$$
这说明 $\hat{\beta}_1 - 3\hat{\beta}_0$ 这个二元正态分布的期望是 $\beta_1 - 3 \beta_0$. 

同理，我们也需要求出其方差，需要注意的是，由于 $\hat{\beta}_0$ 和 $\hat{\beta}_1$ 不独立，且 $\mathrm{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\sigma^2 \dfrac{\bar{x}}{L_xx}$，因此协方差矩阵应该为
$$
\boldsymbol{\Sigma} = \left[\begin{matrix} 
\sigma^2 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & -\sigma^2 \dfrac{\bar{x}}{L_xx} \\
-\sigma^2 \dfrac{\bar{x}}{L_xx} & \dfrac{{\sigma}^2}{L_{xx}}
\end{matrix}\right] = \sigma^2
\left[\begin{matrix} 
 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & - \dfrac{\bar{x}}{L_xx} \\
- \dfrac{\bar{x}}{L_xx} & \dfrac{1}{L_{xx}}
\end{matrix}\right]
$$
从而
$$
\begin{align*}
\boldsymbol{ a}^T \Sigma \boldsymbol{a} 
& = \left[\begin{matrix} -3 & 1\end{matrix}\right] \sigma^2
\left[\begin{matrix} 
 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & - \dfrac{\bar{x}}{L_xx} \\
- \dfrac{\bar{x}}{L_xx} & \dfrac{1}{L_{xx}}
\end{matrix}\right] \left[\begin{matrix} -3 \\ 1\end{matrix}\right]  \\
& = \sigma^2 \left( 9  \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) -6 \left(- \dfrac{\bar{x}}{L_xx} \right) + \dfrac{1}{L_{xx}} \right) \\
& = \sigma^2 \left( \dfrac{9}{n} + \dfrac{(3 \bar{x} + 1)^2}{L_{xx}} \right)
\end{align*}
$$
所以，$\hat{\beta}_1 - 3\hat{\beta}_0$ 的分布是正态分布，即
$$
\hat{\beta}_1 - 3\hat{\beta}_0 \sim N \left( 
\beta_1 - 3 \beta_0
, 
\sigma^2 \left( \dfrac{9}{n} + \dfrac{(3 \bar{x} + 1)^2}{L_{xx}} \right)
\right)
$$

**解2：**

首先，找出待估计量的良好点估计，显然 $\hat{\beta}_0, \ \hat{\beta}_1$ 分别是 ${\beta}_0 ,\ {\beta}_1$ 的良好点估计. 

对立假设 $H_1$ 要求我们 $\beta_1 \neq 3\beta_0$，即 $\beta_1 - 3\beta_0\neq 0$，也就是说 $\vert \hat{\beta}_1 - 3 \hat{\beta}_0 \vert$ 偏大时拒绝原假设 $H_0$（双边检验）. 

设一个常数 $C$，当 $\vert \hat{\beta}_1 - 3 \hat{\beta}_0 \vert > C$ 时拒绝原假设 $H_0$，接下来就是寻找 $C$. 

首先，在第一问中，我们已经知道了$\hat{\beta}_1 - 3\hat{\beta}_0$ 的分布是正态分布，所以可以对其标准化，得到
$$
Z := \dfrac{\hat{\beta}_1 - 3\hat{\beta}_0 - ({\beta}_1 - 3 {\beta}_0)}{\sigma \sqrt{ \left( \dfrac{9}{n} + \dfrac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \sim N(0, 1)
$$
这样里面还带有参数 $\sigma$，我们可以用 $\hat{\sigma}$ 估计它，已知
$$
K^2 := \dfrac{n-2}{\sigma^2} \hat{\sigma}^{2} \sim \chi^2 (n-2)
$$
又因为 $\sigma^2$ 与  $\hat{\beta}_0, \ \hat{\beta}_1$ 都独立，从而我们可以够造出 $t$ 分布，即
$$
T := \dfrac{Z}{\sqrt{K^2 / (n-2)}} =    
\dfrac{\hat{\beta}_1 - 3\hat{\beta}_0 - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \dfrac{9}{n} + \dfrac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }}
\sim t(n-2)
$$
这样我们就可以知道
$$
\begin{align*}
P(\vert \hat{\beta}_1 - 3 \hat{\beta}_0 \vert > C) &= 
P \left( \left| \frac{\hat{\beta}_1 - 3\hat{\beta}_0 - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right| > \frac{C - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right)
\\
&= P_{H_0} \left( \left| T \right| > \frac{C}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \Big| \beta_1 - 3\beta_0 =0 \right)
\end{align*}
$$
由于这是个双侧检验，$C$ 其实有两个值

> 但由于 $t$ 分布是对称的，其实它们只是差了负号，且 $t_{1-\alpha/2}(n) = -t_{\alpha/2}(n)$. 但如果之后有类似题出现不对称分布则需要这样分开讨论. 下面还是给出分开算的方法，最后再合并一起. 

$$
\frac{C_1}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} = t_{\alpha/2} (n-2)
\implies C_1 = \hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) } t_{\alpha/2} (n-2)
\\
\frac{C_2}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} = - t_{ \alpha/2}(n-2)
\implies C_2 = -\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) } t_{\alpha/2} (n-2)
$$

回到刚才的拒绝域中
$$
\left\{ \vert \hat{\beta}_1 - 3 \hat{\beta}_0 \vert > C\right\} \implies  \left\{  \hat{\beta}_1 - 3 \hat{\beta}_0  > C_1\right\} \cup \left\{ \hat{\beta}_1 - 3 \hat{\beta}_0  < C_2\right\}
$$
所以拒绝域是
$$
\left\{  \hat{\beta}_1 - 3 \hat{\beta}_0  > \hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) } t_{\alpha/2} (n-2)\right\} \cup \left\{ \hat{\beta}_1 - 3 \hat{\beta}_0  < - \hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) } t_{\alpha/2} (n-2)\right\}
$$
最后，也可以进一步简化为
$$
\left\{  \left| \hat{\beta}_1 - 3 \hat{\beta}_0 \right| > \hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) } t_{\alpha/2} (n-2)\right\}
$$
### 五、分析题（共 15 分）

图不清晰话不明，不知此题何意味. 

### 六、计算题（共 15 分）

简单随机样本 $X_1, X_2, \dots, X_n$ 来自正态总体 $N(\theta, \sigma_0^2)$，其中 $\sigma_0^2$ 已知. 参数 $\theta$ 的先验分布是指数分布，密度函数为 $p(t) = \lambda_0 e^{-\lambda_0 t},  \ t > 0$，其中 $\lambda_0$ 已知. 在平方损失函数下推导出参数 $\theta$ 的 Bayes 估计，并且计算出这个 Bayes 估计的均方误差. 

**解：**

似然函数为
$$
L(\theta) = \prod_{i=1}^n \dfrac{1}{\sqrt{2\pi \sigma_0^2}} \exp \left(-\dfrac{(x_i-\theta)^2}{2\sigma^2_0} \right) = \left( \frac{1}{2\pi\sigma_0^2} \right)^{n/2} \exp\left( -\frac{1}{2\sigma_0^2} \sum_{i=1}^n (x_i - \theta)^2 \right)
$$

先验分布为 $p(\theta) = \lambda_0 e^{-\lambda_0 \theta}$，因此计算后验分布
$$
\begin{align*}
h(\theta | X)  
& \propto L(\theta) p(\theta)
\\ 
& \propto \exp\left( -\frac{1}{2\sigma_0^2} \sum_{i=1}^n (x_i - \theta)^2 - \lambda_0 \theta \right) & \theta > 0
\\
&\propto \exp\left( -\frac{1}{2\sigma_0^2} \left( \sum_{i=1}^n x_i^2 - 2 \theta \sum_{i=1}^n x_i + n\theta^2 + 2 \lambda_0 \sigma_0^2 \theta \right)  \right) 
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} \left( \theta^2 -2 \left(\dfrac{1}{n} \sum_{i=1}^n x_i - \dfrac{\lambda_0 \sigma_0^2}{n} \right) \theta+ \dfrac{\sum_{i=1}^n x_i^2}{n}  \right)  \right)
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} \left( \theta^2 -2 \left(\bar{x} - \dfrac{\lambda_0 \sigma_0^2}{n} \right) \theta+ \dfrac{\sum_{i=1}^n x_i^2}{n}  \right)  \right)
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} (\theta - \mu )^2  \right) & \mu = \bar{x} - \dfrac{\lambda_0 \sigma_0^2}{n}
\end{align*}
$$
> 因为上述式子是 $\propto$ 连接的而不是 $=$，所以指数部分的常数值可以通过核方法被舍弃. 核心思想是把指数部分凑出一个 $\frac{(x - \mu)^2}{2\sigma^2}$ 的形式. 

由于先验分布要求 $\theta>0$，所以后验分布是带截断的正态分布
$$
\theta | X \sim N \left(\bar{x} - \dfrac{\lambda_0 \sigma_0^2}{n} , \dfrac{\sigma_0^2}{n}\right) \mathbf{1}
_{θ>0}
$$
如果我们忽略截断，那么参数 $ \theta$ 的贝叶斯估计 $\hat{\theta}_B = E(\theta|X) = \bar{X} - \dfrac{\lambda_0 \sigma_0^2}{n}$. 

$\hat{\theta}_B$ 的均方误差是
$$
\begin{align*}
MSE(\hat{\theta}_B) & = D(\hat{\theta}_B) + \left( E(\hat{\theta}_B) - \theta \right)^2 \\
&= D\left(\bar{X} - \frac{\lambda_0 \sigma_0^2}{n}\right) +\left( E\left( \bar{X} - \frac{\lambda_0 \sigma_0^2}{n} \right)  - \theta\right)^2 \\
&=D(\bar{X}) + \left( E( \bar{X}) - \frac{\lambda_0 \sigma_0^2}{n}  - \theta\right)^2 \\
&= \dfrac{\sigma_0^2}{n} + \left( \theta - \frac{\lambda_0 \sigma_0^2}{n}  - \theta\right)^2
\\
&= \dfrac{\sigma_0^2}{n} + \frac{\lambda_0^2 \sigma_0^4}{n^2} \\
&= \dfrac{\sigma_0^2}{n} \left(1 + \dfrac{\lambda_0^2 \sigma_0^2}{n} \right)
\end{align*}
$$

### 七、计算题（共 15 分）

已知总体 $X$ 的概率密度函数为：
$$
f(x; \theta) = 
\begin{cases}
\dfrac{1}{\theta}, & \theta < x < 2\theta, \\
0, & \text{其他}.
\end{cases}
$$
其中 $\theta > 0$ 为未知参数，$X_1, X_2, X_3$ 是来自总体 $X$ 的一个简单随机样本, $\bar{X} = \dfrac{X_1 + X_2 + X_3}{3}$. 

1. （5分）求参数 $\theta$ 的极大似然估计量 $\hat{\theta}$；
2. （5分）求实数 $k_1$ 和 $k_2$，使 $T_1 = k_1 \bar{X}$ 和 $T_2= k_2 \hat{\theta}$ 均为 $\theta$ 的无偏估计量；
3. （5分）比较 $T_1$ 和 $T_2$ 的有效性. 

**解 1：**

首先建立似然函数
$$
L(\theta) = \prod_{i=1}^n f(x;\theta) = \dfrac{1}{\theta^n}, \quad \theta> 0
$$
观察似然函数，若要最大化 $L(\theta) $，只需要最小化 $\theta$. 根据题目的不等式约束条件
$$
0 < \theta < X_{(1)} \quad X_{(n)} \leq 2\theta
$$
可得
$$
\hat{\theta} = \dfrac{X_{(n)}}{2}
$$
**解 2：**
$$
E(T_1) = E(k_1 \bar{X}) = k_1 E(\bar{X}) = \dfrac{k_1}{3} \left(E(X_1) + E(X_2) + E(X_3) \right)
$$
因为 $X_i \sim U(\theta, 2\theta)$，所以 $E(X_i) = \dfrac{\theta + 2\theta}{2} = \dfrac{3}{2} \theta$. 

所以
$$
E(T_1) = \dfrac{k_1}{3}\times 3 \times \dfrac{3}{2} \theta = \dfrac{3}{2} k_1 \theta
$$
为使得 $T_1$ 为无偏估计量，需要令统计量的数学期望等于参数，即
$$
E(T_1) = \theta
$$
解得
$$
k_1 = \dfrac{2}{3}
$$
同理，对于统计量 $T_2$
$$
E(T_2) = E(k_2 \hat{\theta}) = E\left(k_2 \dfrac{X_{(3)}}{2}\right) = \dfrac{k_2}{2}E(X_{(3)})
$$
接下来求 $E\left(X_{(3)}\right)$，首先我们需要构建 $X_{(n)}$ 的概率密度函数
$$
f_{X_{(n)}}(x) = n f_X(x) (F_{X}(x))^{n-1} \\
f_{X_{(3)}}(x) = 3 f_X(x) (F_{X}(x))^{2} = 3 \cdot \dfrac{1}{\theta} \cdot \left( \dfrac{x - \theta}{\theta} \right)^2 = \dfrac{3}{\theta^3} (x-\theta)^2
$$
这样可以计算期望
$$
\begin{align*}
E\left(X_{(3)}\right) &= \int_\theta^{2\theta} f_{X_{(3)}}(x) \cdot x \mathrm{d}x \\
& = \int_\theta^{2\theta} \dfrac{3}{\theta^3} (x-\theta)^2x \mathrm{d}x \\
 &= \dfrac{3}{\theta^3} \left(\dfrac{x^4}{4} - \dfrac{2\theta x^3}{3} + \dfrac{\theta^2 x^2}{2} \right) \Big|^{2\theta}_{\theta}\\
 &= \dfrac{7}{4}\theta
 \end{align*}
$$
所以令
$$
E(T_2) =\dfrac{k_2}{2} \cdot \dfrac{7\theta}{4} = \theta
$$
解得
$$
k_2 = \dfrac{8}{7}
$$
**解 3：**

比较两个统计量的有效性需要计算均方误差，即比较 $MSE(T_1)$ 与 $MSE(T_2)$ 的大小. 

又由于 $T_1$ 与 $T_2$ 都是无偏估计量，所以问题转化为比较 $D(T_1)$ 与 $D(T_2)$ 的大小. 
$$
D(T_1) = D \left( k_1 \bar{X} \right) =  k_1^2 D \left(\bar{X} \right) = \dfrac{k_1^2}{3^2} D \left( \sum_{i=1}^{3} X_i \right)
$$
由于 $X_1, \ X_2, \ X_3$ 相互独立，所以
$$
D(T_1) = \dfrac{k_1^2}{3^2} \times 3 D(X_i)
$$
因为 $X_i \sim U(\theta, 2\theta)$，所以 $D(X_i) = \dfrac{(2\theta  -\theta)^2}{12} = \dfrac{\theta^2}{12}$. 
$$
D(T_1) = \dfrac{k_1^2}{3^2} \times 3 \times \dfrac{\theta^2}{12} = \dfrac{1}{81}\theta^2
$$
接下来计算 $D(T_2)$
$$
D(T_2) = D\left(k_2 \dfrac{X_{(3)}}{2} \right) = \dfrac{k_2^2}{2^2} D( X_{(3)} )
$$
根据方差的定义
$$
D(X_{(3)}) = E(X_{(3)}^2) - [E(X_{(3)}) ]^2
$$
现在只需要计算 $E(X_{(n)}^2) $
$$
\begin{align*}
E\left(X_{(3)}^2\right) &= \int_\theta^{2\theta} f_{X_{(3)}}(x) \cdot x^2 \mathrm{d}x \\
&=  \int_\theta^{2\theta} \dfrac{3}{\theta^3} (x-\theta)^2 x^2 \mathrm{d}x \\
&= \dfrac{3}{\theta^3} \left( \dfrac{x^5}{5} - \dfrac{\theta x^4}{2} + \dfrac{\theta^2 x^3}{3}\right) \Big|^{2\theta}_{\theta} \\
&= \dfrac{31}{10} \theta^2
 \end{align*}
$$
从而
$$
D(X_{(3)}) = \dfrac{31}{10} \theta^2 - \left( \dfrac{7}{4}\theta \right)^2 = \dfrac{3}{80}\theta^2
$$
代入得
$$
D(T_2) =  \dfrac{k_2^2}{2^2} \cdot \dfrac{3}{80}\theta^2 = \dfrac{3}{245}\theta^2
$$
因为
$$
D(T_1) > D(T_2)
$$
即 $T_2$ 的方差更小，所以统计量 $T_2$ 比 $T_1$ 更有效. 

### 八、计算题（共 15 分）

设 $X_1, X_2, \dots, X_n$ 和 $Y_1, Y_2, \dots, Y_m$ 分别来自总体 $X \sim N(\mu_1, \sigma_1^2)$ 和总体 $Y \sim N(\mu_2, \sigma_2^2)$ 的两组独立的简单随机样本，其中 $\sigma_1$ 与 $\sigma_2$ 为大于 $0$ 的未知参数，且 $2\sigma_1 = 3\sigma_2$，对显著性水平 $\alpha$，推导假设检验问题：$H_0: \mu_1 \leq 2\mu_2$，$H_1: \mu_1 > 2\mu_2$ 的拒绝域. 

**解：**

根据假设检验，需要从 $\mu_1$ 和 $\mu_2$ 的良好点估计出发，显然 $\bar{X}$ 与 $\bar{Y}$ 分别是 $\mu_1$ 和 $\mu_2$ 的良好点估计. 

由于 $\bar{X} \sim N\left(\mu_1, \dfrac{\sigma_1^2}{n}\right)$，$\bar{Y} \sim N\left(\mu_2, \dfrac{\sigma_2^2}{m}\right)$，$2\sigma_1 = 3\sigma_2$ 所以有

$$
\bar{Y} \sim N\left(\mu_2, \dfrac{4 \sigma_1^2}{9m}\right) \implies 2\bar{Y} \sim N\left(2\mu_2, \dfrac{16 \sigma_1^2}{9m}\right)\\
\implies \bar{X} - 2\bar{Y} \sim N\left(\mu_1 - 2\mu_2, \sigma_1^2 \left(\dfrac{1}{n} + \dfrac{16}{9m}\right)\right)
$$

显然我们可以通过标准化 $\bar{X} - 2\bar{Y}$ 得到标准正态分布，即

$$
Z := \dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{\sigma_1^2\left( \dfrac{1}{n} +\dfrac{16}{9m} \right)}} \sim N(0,1)
$$

然而我们还有未知参数 $\sigma_1$ 需要估计，显然可以用 $S_1^2$ 来估计. 但由于我们知道 $2\sigma_1 = 3\sigma_2$，因此更合适的方法是将 $S_2^2$ 也充分利用. 

根据抽样分布定理

$$
\dfrac{(n-1)S_1^2}{\sigma_1^2} \sim \chi^2(n-1) , \quad \dfrac{(m-1)S_1^2}{\sigma_2} \sim \chi^2(m-1)
$$

又由于卡方分布具有可加性

$$
\dfrac{(n-1)S_1^2}{\sigma_1^2} + \dfrac{(m-1)S_1^2}{\sigma_2^2} \sim \chi^2(m+n-2)
$$

我们将上式记为 $K^2$，并把 $\sigma_2$ 用 $\sigma_1$ 代换，即

$$
K^2 := \dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 \sigma_1^2} \sim \chi^2 (n+m-2)
$$

从而构造 $t$ 分布以消去其中的未知参数 $\sigma_1$

$$
T := \dfrac{Z}{\sqrt{K^2 /(n+m-2)}} = \dfrac{\dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{\sigma_1^2\left( \dfrac{1}{n} +\dfrac{16}{9m} \right)}}}{\sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 \sigma_1^2 (n+m-2)}}} = 
 \dfrac{\dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} }}}{\sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} 
 = \dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}}
$$

当原假设成立时，我们观察到的 $\bar{X} - 2 {\bar{Y}}$ 应该很小，反之，如果我们要拒绝原假设，就需要 $\bar{X} - 2 {\bar{Y}}$ 偏大，所以拒绝域应为 $\left\{ \bar{X} - 2 {\bar{Y}} > C\right\}$，接下来我们来寻找常数 $C$. 

$$
P \left\{  \bar{X} - 2\bar{Y} > C \right\} 
= P \left\{  \dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} > \dfrac{C - (\mu_1 - 2\mu_2 )}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} \right\} \\
= 
P_{H_0} \left\{  T > \dfrac{C}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} \Big| \mu_1 - 2\mu_2 =0 \right\} < \alpha
$$

从而我们找到了 $C$，即显著性水平 $\alpha$ 下满足

$$
\dfrac{C}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m}} \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} = t_{\alpha}(n+m-2)
$$

所以显著性水平 $\alpha$ 的拒绝域为

$$
\left\{  \bar{X} - 2\bar{Y} > t_{\alpha}(n+m-2) \sqrt{ \dfrac{1}{n} +\dfrac{16}{9m}} \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}} \right\}
$$
