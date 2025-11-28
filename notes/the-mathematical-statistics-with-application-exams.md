# 应用数理统计历年真题

> 作者：赤川鹤鸣_Channel
>
> 不保证全对，请批判地看. 但保证所有题已经过人工和 AI 双重评判. 

我们约定：
1. 样本都是简单随机样本. 样本均值与样本方差分别定义为
$$
\bar{X} = \dfrac{1}{n} \sum_{i=1}^{n} X_k , \quad S^2 = \dfrac{1}{n-1} \sum_{i=1}^{n} (X_k - \bar{X})^2
$$
2. 分位点 $Q_\alpha$ 取为上侧分位点，即：$P(X>Q_\alpha)=\alpha$

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

| 顺序   | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 样本值 | 27   | 28   | 29   | 30   | 31   | 32   | 33   | 34   | 35   | 36   | 38   |
| 分组   | X    | Y    | X    | X    | Y    | X    | X    | Y    | Y    | Y    | Y    |

因为 $X$ 的样本数量有 5 个，$Y$ 的样本数量有 6 个，选择样本数量更少的 $X$ 样本求秩和（把分组为 $X$ 的顺序号求和）
$$
W = 1 +3 +4 +6+ 7=21
$$
原假设 $H_0$ 的拒绝域为
$$
W \leq T_1 \cup W\geq T_2
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
\begin{align}
\boldsymbol{ a}^T \Sigma \boldsymbol{a} 
& = \left[\begin{matrix} -3 & 1\end{matrix}\right] \sigma^2
\left[\begin{matrix} 
 \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) & - \dfrac{\bar{x}}{L_xx} \\
- \dfrac{\bar{x}}{L_xx} & \dfrac{1}{L_{xx}}
\end{matrix}\right] \left[\begin{matrix} -3 \\ 1\end{matrix}\right]  \\
& = \sigma^2 \left( 9  \left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{L_{xx}}\right) -6 \left(- \dfrac{\bar{x}}{L_xx} \right) + \dfrac{1}{L_{xx}} \right) \\
& = \sigma^2 \left( \dfrac{9}{n} + \dfrac{(3 \bar{x} + 1)^2}{L_{xx}} \right)
\end{align}
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
P_{H_0}(\vert \hat{\beta}_1 - 3 \hat{\beta}_0 \vert > C) = 
P_{H_0} \left( \left| \frac{\hat{\beta}_1 - 3\hat{\beta}_0 - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right| > \frac{C - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right)
 \overset{H_0}{=} \\
P \left( \left| \frac{\hat{\beta}_1 - 3\hat{\beta}_0 - ({\beta}_1 - 3 {\beta}_0)}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right| > \frac{C}{\hat{\sigma} \sqrt{ \left( \frac{9}{n} + \frac{(3 \bar{x} + 1)^2}{L_{xx}} \right) }} \right)
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

简单随机样本 $X_1, X_2, \dots, X_n$ 来自正态总体 $N(\theta, \sigma_0^2)$，其中 $\sigma_0^2$ 已知. 参数 $\theta$ 的先验分布是指数分布，密度函数为 $p(t) = \lambda_0 e^{-\lambda_0 t},  \ t > 0$，其中 $\lambda_0$ 已知。在平方损失函数下推导出参数 $\theta$ 的 Bayes 估计，并且计算出这个 Bayes 估计的均方误差. 

**解：**

似然函数为
$$
L(\theta) = \prod_{i=1}^n \dfrac{1}{\sqrt{2\pi \sigma_0^2}} \exp \left(-\dfrac{(x_i-\theta)^2}{2\sigma^2_0} \right) = \left( \frac{1}{2\pi\sigma_0^2} \right)^{n/2} \exp\left( -\frac{1}{2\sigma_0^2} \sum_{i=1}^n (x_i - \theta)^2 \right)
$$

先验分布为 $p(\theta) = \lambda_0 e^{-\lambda_0 \theta}$，因此计算后验分布
$$
\begin{align}
h(\theta | X)  
& \propto L(\theta) p(\theta)
\\ 
& \propto \exp\left( -\frac{1}{2\sigma_0^2} \sum_{i=1}^n (x_i - \theta)^2 - \lambda_0 \theta \right) & \theta > 0
\\
&\propto \exp\left( -\frac{1}{2\sigma_0^2} \left( \sum_{i=1}^n x_i^2 - 2 \theta \sum_{i=1}^n x_i + n\theta^2 - 2 \lambda_0 \sigma_0^2 \theta \right)  \right) 
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} \left( \theta^2 -2 \left(\dfrac{1}{n} \sum_{i=1}^n x_i + \dfrac{\lambda_0 \sigma_0^2}{n} \right) \theta+ \dfrac{\sum_{i=1}^n x_i^2}{n}  \right)  \right)
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} \left( \theta^2 -2 \left(\bar{x} + \dfrac{\lambda_0 \sigma_0^2}{n} \right) \theta+ \dfrac{\sum_{i=1}^n x_i^2}{n}  \right)  \right)
\\
&\propto \exp\left( -\frac{n}{2\sigma_0^2} (\theta - \mu )^2  \right) & \mu = \bar{x} + \dfrac{\lambda_0 \sigma_0^2}{n}
\end{align}
$$
> 因为上述式子是 $\propto$ 连接的而不是 $=$，所以指数部分的常数值可以被舍弃. 核心思想是把指数部分凑出一个 $\frac{(x - \mu)^2}{2\sigma^2}$ 的形式. 

由于先验分布要求 $\theta>0$，所以后验分布是带截断的正态分布
$$
\theta | X \sim N \left(\bar{x} + \dfrac{\lambda_0 \sigma_0^2}{n} , \dfrac{\sigma_0^2}{n}\right) \mathbf{1}
_{θ>0}
$$
如果我们忽略截断，那么参数 $ \theta$ 的贝叶斯估计 $\hat{\theta}_B = E(\theta|X) = \bar{x} + \frac{\lambda_0 \sigma_0^2}{n}$. 

$\hat{\theta}_B$ 的均方误差是
$$
\begin{align}
MSE(\hat{\theta}_B) & = D(\hat{\theta}_B) + \left( E(\hat{\theta}_B) - \theta \right)^2 \\
&= D\left(\bar{X} + \frac{\lambda_0 \sigma_0^2}{n}\right) +\left( E\left( \bar{X} + \frac{\lambda_0 \sigma_0^2}{n} \right)  - \theta\right)^2 \\
&=D(\bar{X}) + \left( E( \bar{X})+ \frac{\lambda_0 \sigma_0^2}{n}  - \theta\right)^2 \\
&= \dfrac{\sigma_0^2}{n} + \left( \theta+ \frac{\lambda_0 \sigma_0^2}{n}  - \theta\right)^2
\\
&= \dfrac{\sigma_0^2}{n} + \frac{\lambda_0^2 \sigma_0^4}{n^2} \\
&= \dfrac{\sigma_0^2}{n} \left(1 + \dfrac{\lambda_0^2 \sigma_0^2}{n} \right)
\end{align}
$$

如果不能忽略截断，我们记
$$
\mu =\bar{x} + \dfrac{\lambda_0 \sigma_0^2}{n}, \quad \sigma^2 = \dfrac{\sigma_0^2}{n}
$$
带截断的正态分布的均值是
$$
E[ \theta|X] = \mu + \sigma \dfrac{\varphi(\mu / \sigma)}{\Phi(\mu / \sigma)}
$$
其中，$\varphi(\cdot)$ 是概率密度函数，$\Phi(\cdot)$ 是累计分布函数. 

所以在平方损失下，Bayes 估计是后验均值，那么参数 $ \theta$ 的贝叶斯估计
$$
\hat{\theta}_B =\bar{X} + \dfrac{\lambda_0 \sigma_0^2}{n} + \dfrac{\sigma_0}{\sqrt{n}} \dfrac{\varphi(\alpha)}{\Phi(\alpha)}, \quad \alpha = \dfrac{\mu}{\sigma} = \dfrac{\sqrt{n} \bar{X}}{\sigma_0} - \dfrac{\lambda_0 \sigma_0}{\sqrt{n}}
$$
这样均方误差损失是
$$
MSE(\theta) = \sigma^2 D_Z \left(Z + \dfrac{\varphi(\alpha)}{\Phi(\alpha)} \right) + \left(\lambda_0 \sigma^2 + \sigma E_Z \left( \dfrac{\varphi(\alpha)}{\Phi(\alpha)} \right) \right)^2
$$
其中 $Z$ 是标准正态分布. 

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
\begin{align}
E\left(X_{(3)}\right) &= \int_\theta^{2\theta} f_{X_{(3)}}(x) \cdot x \mathrm{d}x \\
& = \int_\theta^{2\theta} \dfrac{3}{\theta^3} (x-\theta)^2x \mathrm{d}x \\
 &= \dfrac{3}{\theta^3} \left(\dfrac{x^4}{4} - \dfrac{2\theta x^3}{3} + \dfrac{\theta^2 x^2}{2} \right) \Big|^{2\theta}_{\theta}\\
 &= \dfrac{7}{4}\theta
 \end{align}
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
\begin{align}
E\left(X_{(3)}^2\right) &= \int_\theta^{2\theta} f_{X_{(3)}}(x) \cdot x^2 \mathrm{d}x \\
&=  \int_\theta^{2\theta} \dfrac{3}{\theta^3} (x-\theta)^2 x^2 \mathrm{d}x \\
&= \dfrac{3}{\theta^3} \left( \dfrac{x^5}{5} - \dfrac{\theta x^4}{2} + \dfrac{\theta^2 x^3}{3}\right) \Big|^{2\theta}_{\theta} \\
&= \dfrac{31}{10} \theta^2
 \end{align}
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
\leq
P_{H_0} \left\{  \dfrac{\bar{X} - 2\bar{Y} - (\mu_1 - 2\mu_2 )}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} > \dfrac{C}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m} } \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} \right\} < \alpha
$$

> 最后一步概率式子上的 $≤$ 来自这个基本事实，若 $a ≥ b$，则 $P(T > a) ≤ P(T > b)$，即随着上分位点增大，分位点右侧曲线下的面积减小，所以概率减小. 当原假设成立下，$\mu_1 - 2\mu_2 \leq 0$，因此当我们去掉 $\mu_1 - 2\mu_2$ 相当于让上分位点增大了，因此是 $\leq $ 号. 

从而我们找到了 $C$，即显著性水平 $\alpha$ 下满足
$$
\dfrac{C}{\sqrt{ \dfrac{1}{n} +\dfrac{16}{9m}} \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}}} = t_{\alpha}(n+m-2)
$$
将 $C$ 代入显著性水平 $\alpha$ 的拒绝域
$$
\left\{  \bar{X} - 2\bar{Y} > t_{\alpha}(n+m-2) \sqrt{ \dfrac{1}{n} +\dfrac{16}{9m}} \sqrt{\dfrac{4(n-1) S_1^2 + 9(m-1) S_2^2}{4 (n+m-2)}} \right\}
$$
