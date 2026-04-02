# LeJEPA 中一些证明的补充

## B.3 Proof of lemma.4

这个证明的总体思路是，要求出 $\hat{\eta}(x)$ 与真实 $\eta(x)$ 之间的误差，在概率论中哦我们通常使用某个统计量的期望与该统计量实际值之差作为评判标准，即需要求出
$$
\mathbb{E}[\hat{\eta}(x)] - \eta(x)
$$
只要抓住这一点，就知道为什么我们需要在接下来分别证明球积分以及一些别的东西，这都是要求解该式的前提. 

### 球积分

在 LeJEPA 中的 lemma.4 证明中，出现了这样三个式子令我很困惑
$$
\int_{B(\boldsymbol{0},r)} \boldsymbol{z} \, \mathrm{d}\boldsymbol{z} = 0, \qquad \int_{B(\boldsymbol{0},r)} \boldsymbol{z}\boldsymbol{z}^\top \, \mathrm{d}\boldsymbol{z} = \frac{\mathrm{Vol}^{d+2}}{d+2} I_d, \qquad \int_{B(\boldsymbol{0},r)} \|\boldsymbol{z}\|^2 \, \mathrm{d}\boldsymbol{z} = \frac{d \, \mathrm{Vol}^{d+2}}{d+2}.
$$
> [!NOTE]
>
> 我们将使用正体英文 $\mathrm{d}$ 表示微分符号，而斜体的 $d$ 表示维度. 

这是一个在超球体中的积分问题. 我们用 $B(\boldsymbol{0},r)$ 表示以原点为球心，$r$ 为半径的超球体，假设该超球体具有维度 $d$. 
$$
\int_{B(\boldsymbol{0},r)} \boldsymbol{z} \, \mathrm{d} \boldsymbol{z} = 0 \tag{1}
$$
对于式 (1)，由于整个超球是对称的（即便在高维空间也是如此），给定任何一点都能找到对应相对称的一点使之抵消，这样根据对称性可以得到此积分式的结果为 $0$. 
$$
\int_{B(\boldsymbol{0},r)} \boldsymbol{z}\boldsymbol{z}^\top \, \mathrm{d}\boldsymbol{z} = \frac{\mathrm{Vol}^{d+2}}{d+2} I_d \tag{2}
$$
$$
\int_{B(\boldsymbol{0},r)} \|\boldsymbol{z}\|^2 \, \mathrm{d}\boldsymbol{z} = \frac{d \, \mathrm{Vol}^{d+2}}{d+2} \tag{3}
$$

式 (2) 和式 (3) 开始令人迷惑了，首先观察到里面 $\boldsymbol{z}\boldsymbol{z}^\top$ 是一个 $\mathbb{R}^{d\times1}$ 与 $\mathbb{R}^{1\times d}$ 的向量相乘，这种相乘叫做**向量的外积**，结果是一个矩阵，形状为 $\boldsymbol{z}\boldsymbol{z}^\top \in \mathbb{R}^{d\times d}$，我们记为 $\boldsymbol{Z}$. 

我们学习过标量函数的积分，但是现在里面却是一个矩阵，怎么办？而且，随着 $\boldsymbol{z}$ 在球体中的移动（想象一下微积分），$\boldsymbol{Z}$ 会有不同的变化. 在标量函数的积分，尤其是黎曼积分里，我们通常把自变量的坐标轴切分成足够多的小份，即微元；那么在矩阵的积分中，我们也想象我们也可以按照某种方式讲球也切为微元，由于每个微元都足够小，在这个微元里 $\boldsymbol{Z}$ 几乎都相同，就像标量函数积分那样把对应的函数值加起来，我们这里也把矩阵相加就可以了. 

听起来，原理倒是相似的，不过我们还是不太知道怎么把这么多的微元都加一起，考虑到球是具有对称性的，我们看看这里的矩阵 $\boldsymbol{Z}$ 是不是也有类似的性质. 
$$
\boldsymbol{Z} = \boldsymbol{z}\boldsymbol{z}^\top = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_d \end{bmatrix} \begin{bmatrix} z_1 & z_2 & \cdots & z_d \end{bmatrix} = \begin{bmatrix} z_1^2 & z_1 z_2 & \cdots & z_1 z_d \\ z_2 z_1 & z_2^2 & \cdots & z_2 z_d \\ \vdots & \vdots & \ddots & \vdots \\ z_d z_1 & z_d z_2 & \cdots & z_d^2 \end{bmatrix}
$$
$\boldsymbol{Z}$ 是一个对称矩阵，同时注意到这样的事实. 以 $z_1 z_2$ 为例，如果我们定死 $z_1$，那么给定任何一个 $z_2$ 我们一定能在球内找到一个相反的 $z_2$ 使其抵消，这说明我们可以利用球对称性消去大部分的积分式，即

$$
\int_{{B(\boldsymbol{0},r)}} z_i z_j = 0, \quad i\neq j
$$
由球的各向同性，所有对角元相等（记为 $C$），这是显然的，因为 $z_i$ 的取值范围就是从 $[-r, +r]$，对于任何一个分量也都如此 
$$
\displaystyle\int_{B(\boldsymbol{0},r)} z_1^2 \, \mathrm{d}\boldsymbol{z} = \int_{B(\boldsymbol{0},r)} z_2^2 \, \mathrm{d}\boldsymbol{z} = \cdots = \int_{B(\boldsymbol{0},r)} z_d^2 \, \mathrm{d}\boldsymbol{z} = C
$$

所以我们可以得到这样的式子
$$
\int_{B(\boldsymbol{0},r)} \boldsymbol{z}\boldsymbol{z}^\top \, \mathrm{d}\boldsymbol{z} = \begin{bmatrix} 
\displaystyle\int_{B(\boldsymbol{0},r)} z_1^2 \, \mathrm{d}\boldsymbol{z} & \displaystyle\int_{B(\boldsymbol{0},r)} z_1 z_2 \, \mathrm{d}\boldsymbol{z} & \cdots & \displaystyle\int_{B(\boldsymbol{0},r)} z_1 z_d \, \mathrm{d}\boldsymbol{z} \\[20pt]
\displaystyle\int_{B(\boldsymbol{0},r)} z_2 z_1 \, \mathrm{d}\boldsymbol{z} & \displaystyle\int_{B(\boldsymbol{0},r)} z_2^2 \, \mathrm{d}\boldsymbol{z} & \cdots & \displaystyle\int_{B(\boldsymbol{0},r)} z_2 z_d \, \mathrm{d}\boldsymbol{z} \\[20pt]
\vdots & \vdots & \ddots & \vdots \\[10pt]
\displaystyle\int_{B(\boldsymbol{0},r)} z_d z_1 \, \mathrm{d}\boldsymbol{z} & \displaystyle\int_{B(\boldsymbol{0},r)} z_d z_2 \, \mathrm{d}\boldsymbol{z} & \cdots & \displaystyle\int_{B(\boldsymbol{0},r)} z_d^2 \, \mathrm{d}\boldsymbol{z}
\end{bmatrix}
$$

因此最终简化为
$$
\int_{B(\boldsymbol{0},r)} \boldsymbol{z}\boldsymbol{z}^\top \, \mathrm{d}\boldsymbol{z} = \begin{bmatrix} 
C & 0 & \cdots & 0 \\
0 & C & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & C
\end{bmatrix} = C \cdot \boldsymbol{I}_d
$$

那么问题就转化为求 $C$ 是多少了. 

考虑到矩阵的迹和积分号可以互换，而且可以通过矩阵的迹将矩阵化为对角元的和，我们这样做
$$
\begin{align*}
\mathrm{tr}\left( \int_{B(\boldsymbol{0},r)} \boldsymbol{z}\boldsymbol{z}^\top \, \mathrm{d}\boldsymbol{z} \right)
&= \int_{B(\boldsymbol{0},r)}\mathrm{tr}\left(\boldsymbol{z}\boldsymbol{z}^\top \right)  \mathrm{d}\boldsymbol{z}
\\ &= \int_{B(\boldsymbol{0},r)} \left( z_1^2 + z_2^2 + \cdots + z_d^2 \right) \mathrm{d}\boldsymbol{z} \\
&= \int_{B(\boldsymbol{0},r)} \lVert \boldsymbol{z}\rVert^2_2 \mathrm{d}\boldsymbol{z}
\end{align*}
$$
而 $\lVert \boldsymbol{z}\rVert_2$ 是向量的 2-范数，它是向量里所有元素平方和的开根号. 具体的定义如下
$$
\lVert \boldsymbol{z}\rVert_2 = \sqrt{z_1^2+z_2^2+ \cdots + z_d^2}
$$
因此，不严谨地说，$\lVert \boldsymbol{z}\rVert_2$ 可以视为一种长度，而这刚好就是从球心伸出的不超过其半径 $r$ 的某个线段的长度. 而微元法看起来就像是要遍历所有这样的线段（通过遍历$z_1, z_2, \dots, z_d$），但是在直角坐标系里直接微元非常棘手，而如果我们学习高等数学里的极坐标和球坐标的思想，推广到高维空间里，那就非常哦耶了. 思路就是，想象一个洋葱，我们通过改变这个半径，一层一层的进行微元，每个微元类似一个球壳，而不是像切萝卜丁那样水平竖直地立方微元.  

于是，我们令 $\rho$ 作为一种径向微元，$V_{d}(r)$ 是半径为 $r$ 的 $d$ 维超球的体积，$S_{d-1}(r)$ 是半径为 $r$ 的 $d$ 维超球的表面积，得到
$$
\begin{align*}
\int_{B(\boldsymbol{0},r)} \lVert \boldsymbol{z}\rVert_2^2 \mathrm{d}\boldsymbol{z}  &=
\int_{0}^r \rho^2  \underbrace{\left( S_{d-1}(1) \cdot \rho^{d-1} \right)}_{见下文式(4)} \mathrm{d} \rho \\
&= \int_{0}^r S_{d-1}(1) \cdot \rho^{d+1}  \mathrm{d} \rho \\ 
&= \underbrace{S_{d-1}(1)}_{提出常量}  \int_{0}^r \rho^{d+1}  \mathrm{d} \rho  \\
&= S_{d-1}(1)  \left( \dfrac{\rho^{d+2}}{d+2} \right) \vert_o^r \\
&= S_{d-1}(1) \dfrac{r^{d+2}}{d+2}
\end{align*}
$$
我们知道，球的体积与表面积具有两个非常有意思的关系：

球的表面积是球体积对半径 $r$ 的导数
$$
S_{d-1} (r) = \dfrac{\mathrm{d} }{\mathrm{d} r} V_d(r) \tag{4}
$$
半径为 $r$ 的球体积与单位球体积的关系
$$
V_d(r)= V_d(1)r^d \tag{5}
$$
由此代入，我们得知
$$
\int_{B(\boldsymbol{0},r)} \lVert \boldsymbol{z} \rVert_2^2 \mathrm{d}\boldsymbol{z} = d \cdot V_d(1) \cdot \dfrac{r^{d+2}}{d+2} \\= 
 \dfrac{d}{d+2} \underbrace{\cdot V_{d}(1) r^d}_{式(5)} \cdot r^2
\\ = \dfrac{d}{d+2}V_d(r) r^2
$$
这里的 $\mathrm{Vol}^{d+2}$ 是论文中的记号，实际含义是 $d$ 维球体积乘以半径平方 $V_d(r)\cdot r^2$，而非 $(d+2)$ 维球的体积 $V_{d+2}(r)$. 

需要强调的是，对矩阵的积分结果还是一个矩阵，所以式 (2) 不要忘记最后是一个矩阵；式 (3) 其实就是把式子 (2) 的矩阵 $\boldsymbol{z}\boldsymbol{z}^\top$ 取了 F-范数后平方，在这里等价于对 $\boldsymbol{z}$ 取 2-范数后再平方. 

综上，我们证明了式 (1) (2) (3).

### $\mathcal{N}(\boldsymbol{x})$ 的求解

$\mathcal{N}(\boldsymbol{x}) $ 是式子 $\mathbb{E}[\hat{\eta}(x)]$ 的分子位置

$$
\mathcal{N}(\boldsymbol{x}) \triangleq \int_{B(\boldsymbol{0},r)} \eta(\boldsymbol{x}+\boldsymbol{z})p(\boldsymbol{x}+\boldsymbol{z})\,\mathrm{d}\boldsymbol{z}
$$

参考原论文将其进行泰勒近似，再把两式子相乘展开后，得到

$$
\begin{align*}
\mathcal{N}(\boldsymbol{x}) = \int_{B(\boldsymbol{0},r)} \Bigg[ &\underbrace{\eta(\boldsymbol{x})p(\boldsymbol{x})}_{\text{常数}} + \underbrace{\eta(\boldsymbol{x})\nabla p(\boldsymbol{x})^{\top}\boldsymbol{z}}_{\text{奇对称}} + \frac{1}{2}\eta(\boldsymbol{x})\boldsymbol{z}^{\top}\boldsymbol{H}p(\boldsymbol{x})\boldsymbol{z}  \\ &+ \underbrace{\nabla\eta(\boldsymbol{x})^{\top}\boldsymbol{z}\,p(\boldsymbol{x})}_{\text{奇对称}} + \nabla\eta(\boldsymbol{x})^{\top}\boldsymbol{z}\,\nabla p(\boldsymbol{x})^{\top}\boldsymbol{z} + \underbrace{\frac{1}{2}\nabla\eta(\boldsymbol{x})^{\top}\boldsymbol{z}\,\boldsymbol{z}^{\top}\boldsymbol{H}p(\boldsymbol{x})\boldsymbol{z}}_{\text{奇对称}} \\ &+ \frac{1}{2}\boldsymbol{z}^{\top}\boldsymbol{H}\eta(\boldsymbol{x})\boldsymbol{z}\,p(\boldsymbol{x}) + \underbrace{\frac{1}{2}\boldsymbol{z}^{\top}\boldsymbol{H}\eta(\boldsymbol{x})\boldsymbol{z}\,\nabla p(\boldsymbol{x})^{\top}\boldsymbol{z}}_{\text{奇对称}} + \underbrace{\frac{1}{4}\boldsymbol{z}^{\top}\boldsymbol{H}\eta(\boldsymbol{x})\boldsymbol{z}\,\boldsymbol{z}^{\top}\boldsymbol{H}p(\boldsymbol{x})\boldsymbol{z}}_{\text{可视为余项}} \Bigg]\,\mathrm{d}\boldsymbol{z}
\end{align*}
$$

奇对称的项均为0，然后积分号里面就只有 4 项加 1 个拉格朗日余项. 这个拉格朗日余项的阶可以通过球积分的公式 (3) 得知，如果二次型的余项能够达到 $O(r^{d+2})$，那么三次型（3 阶因奇对称为 0，实际上是 4 阶）以上的余项一定比 $O(r^{d+2})$ 还小，可以用 $O(r^{d+3})$ 表示了. 当然，其实用 $O(r^{d+4})$ 表示更严谨，但由于我们并不关注余项的阶，给出 $O(r^{d+3})$ 已经足够了. 

$$
= \int_{B(\boldsymbol{0},r)} \Bigg[ \underbrace{\eta(\boldsymbol{x})p(\boldsymbol{x})}_{\text{常数}} + \frac{1}{2}\eta(\boldsymbol{x})\boldsymbol{z}^{\top}\boldsymbol{H}p(\boldsymbol{x})\boldsymbol{z} + \underbrace{\nabla\eta(\boldsymbol{x})^{\top}\boldsymbol{z}\,\nabla p(\boldsymbol{x})^{\top}\boldsymbol{z}}_{等价于\boldsymbol{z}^{\top}\nabla \eta(\boldsymbol{x})\nabla p(\boldsymbol{x})^{\top}\boldsymbol{z}} + \frac{1}{2}\boldsymbol{z}^{\top}\boldsymbol{H}\eta(\boldsymbol{x})\boldsymbol{z}\,p(\boldsymbol{x}) + \underbrace{O(r^{d+3})}_{\text{余项}} \Bigg]\,\mathrm{d}\boldsymbol{z}
$$

第一项的常数项直接相当于乘以球的体积，第二三四项和之前一样，用球积分带入处理，得
$$
= \eta(\boldsymbol{x})p(\boldsymbol{x})v_{d}r^{d} + \eta(\boldsymbol{x})\frac{v_{d}r^{d+2}}{2(d+2)}\mathrm{tr}(\boldsymbol{H}p(\boldsymbol{x})) + \frac{v_{d}r^{d+2}}{d+2}\nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x}) + \frac{v_{d}r^{d+2}}{2(d+2)}p(\boldsymbol{x})\,\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) + O(r^{d+3})
$$

然后，我们先计算一下 $\eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x}) $，这是因为
$$
\mathbb{E}[\hat{\eta}(x)] - \eta(x) = \dfrac{\mathcal{N}(\boldsymbol{x})}{\mathcal{D}(\boldsymbol{x})} - \eta(\boldsymbol{x}) = \dfrac{\mathcal{N}(\boldsymbol{x}) - \eta(\boldsymbol{x}) \mathcal{D}(\boldsymbol{x}) }{\mathcal{D}(\boldsymbol{x})}
$$
这里通分后会用到 $\eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x}) $，所以我们提前算好
$$
\eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x}) = \eta(\boldsymbol{x})p(\boldsymbol{x})v_{d}r^{d} + \eta(\boldsymbol{x})\frac{v_{d}r^{d+2}}{2(d+2)}\mathrm{tr}(\boldsymbol{H}p(\boldsymbol{x})) + O(r^{d+3})
$$

$$
\mathcal{N}(\boldsymbol{x}) - \eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x}) = \frac{v_{d}r^{d+2}}{d+2}\left( \nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x}) + \frac{1}{2}p(\boldsymbol{x})\,\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right) + O(r^{d+3})
$$

其中 $\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x}))$ 对应原论文中的 $\Delta\eta(\boldsymbol{x})$，这是为了方便写成了拉普拉斯算子，二者是等价的. 

然后我们整理一下 $\mathcal{D}(\boldsymbol{x})$ 便于后续的计算，为了简洁文中令 $\alpha(\boldsymbol{x}) = \frac{1}{2(d+2)p(\boldsymbol{x})}\mathrm{tr}(\boldsymbol{H}p(\boldsymbol{x}))$，有
$$
\begin{align*}
\mathcal{D}(\boldsymbol{x}) &= v_{d}r^{d}p(\boldsymbol{x}) + \frac{v_{d}r^{d+2}}{2(d+2)}\mathrm{tr}(\boldsymbol{H}p(\boldsymbol{x})) + O(r^{d+3}) \\
&= v_{d}r^{d}p(\boldsymbol{x})\left( 1 + \frac{r^{2}}{2(d+2)p(\boldsymbol{x})}\mathrm{tr}(\boldsymbol{H}p(\boldsymbol{x})) + O(r^{3}) \right) \\
&= v_{d}r^{d}p(\boldsymbol{x})\left( 1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3}) \right)
\end{align*}
$$

回过头来，我们可以代入式子里，接下来几步分别进行了代入、对 $v_d r^d$ 约分、把 $p(\boldsymbol{x})$ 除进去
$$
\begin{align*}
\frac{\mathcal{N}(\boldsymbol{x}) - \eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x})}{\mathcal{D}(\boldsymbol{x})} 
& = \frac{\frac{v_{d}r^{d+2}}{d+2}\left( \nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x}) + \frac{1}{2}p(\boldsymbol{x})\,\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right) + O(r^{d+3})}{v_{d}r^{d}p(\boldsymbol{x})\left( 1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3}) \right)}
\\
&= \frac{r^{2}}{(d+2)p(\boldsymbol{x})} \frac{\nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x}) + \frac{1}{2}p(\boldsymbol{x})\,\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) + O(r)}{1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3})}
\\
&= \frac{r^{2}}{d+2}\left( \frac{\nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x})}{p(\boldsymbol{x})} + \frac{1}{2}\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right) \frac{1}{1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3})} + O(r^{3})
\end{align*}
$$

注意到，$\frac{1}{1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3})}$ 正好是几何级数展开，回想起 $\frac{1}{1+\varepsilon} = 1-\varepsilon + O(\varepsilon^2)$ 这样的近似在很多地方都在用. 
$$
\frac{1}{1 + \alpha(\boldsymbol{x})r^{2} + O(r^{3})} = 1 - \alpha(\boldsymbol{x})r^{2} + O(r^{3}) = 1 + O(r^3)
$$

我们最后完成推导，先把几何级数展开代入，再用皮亚诺余项  $o(r^2)$ 进一步忽略掉之后的拉格朗日余项 $O(r^3)$ 和 $O(r^5)$ ，并在最后一步利用对数的导数的性质进一步化简表达式 $\nabla \log p(\boldsymbol{x}) = \frac{\nabla p(\boldsymbol{x})}{p(\boldsymbol{x})}$，得
$$
\begin{align*}
\mathbb{E}[\hat{\eta}(x)] - \eta(x) &= 
\frac{\mathcal{N}(\boldsymbol{x}) - \eta(\boldsymbol{x})\mathcal{D}(\boldsymbol{x})}{\mathcal{D}(\boldsymbol{x})}  
\\
&= \frac{r^{2}}{d+2}\left( \frac{\nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x})}{p(\boldsymbol{x})} + \frac{1}{2}\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right)\left( 1 + O(r^3)) \right) + O(r^{3})
\\
&= \frac{r^{2}}{d+2}\left( \frac{\nabla\eta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x})}{p(\boldsymbol{x})} + \frac{1}{2}\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right) + O(r^3)
\\
&= \frac{r^{2}}{d+2}\left( \nabla\eta(\boldsymbol{x})^{\top}\nabla \log p(\boldsymbol{x}) + \frac{1}{2}\mathrm{tr}(\boldsymbol{H}\eta(\boldsymbol{x})) \right) + \underbrace{o(r^2)}_{皮亚诺余项}
\end{align*}
$$
至此 Lemma.4 证明完毕. 

因此，误差 $\mathbb{E}[\hat{\eta}(x)] - \eta(x)$ 与下列因素有关：

1. 当采样的超球半径越大，说明更多的采样点会被囊括，意味着采样越粗糙（粗粒度），误差越大；
2. 隐空间的维度越小，说明模型能够容纳的信息越少，误差越大；
3. $\eta{(\boldsymbol{x})}$ 的 1、2 阶导数越大，说明函数越不平滑，越妨碍模型的收敛，误差越大；
4. $p(\boldsymbol{x})$ 越小，说明数据点的分布越稀疏，越难以建立估计，因此误差越大. 
