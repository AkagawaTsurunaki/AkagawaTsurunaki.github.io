# 变分法笔记

> [!NOTE]
>
> 作者：赤川鹤鸣\_Channel | Author: AkagawaTsurunaki | All rights reserved.
>
> 本笔记的内容来自老大中编著的《变分法》第2版. 第一章预备知识与第九章力学中的变分原理及其应用不在本笔记的范畴内.
>
> 本笔记已经修正了大多数原教材中出现的笔误，但仍然可能有错误，请注意自行甄别.

## 2 固定边界的变分问题

### 2.2 变分法的基本概念

#### 泛函

$$
J = J[\underbrace{y(x)}_{\text{宗量}}] = J[y],\quad y(x) \in \underbrace{F}_{\text{类函数}}
$$

- $F$ 是 $J$ 的定义域；
- 在 $F$ 中的 $y(x)$ 为容许函数（可取函数）；
- $J$ 的值是数，而 $J$ 的自变量是函数 $y(x)$；
- $J$ 取决于 $F$ 中 $y(x)$ 与 $x$ 的函数关系.

#### $n$ 阶距离

**$n$ 阶距离**：两函数的 $0$ 至 $n$ 阶导数之差的绝对值中最大的那个数.

$$
d_n[y(x), y_0(x)] = \max_{0 \le i \le n} \max_{a \le x \le b} \vert y^{(i)}(x) - y_0^{(i)}(x) \vert, \quad y(x) \in C^{n} [a, b]
$$

- 两条曲线重合 $\Leftrightarrow$ 两条曲线的零阶距离为 $0$.

**$n$ 阶距离的单调性不等式**

$$
d_0[y, y_0] \le d_1[y, y_0] \le \cdots \le d_n[y, y_0]
$$

#### $n$ 阶 $\delta$ 邻域

**$n$ 阶 $\delta$ 邻域**

$$
N_n[\delta, y_0(x)] = \{y(x) \mid y(x) \in C^n[a,b], d_n[y(x), y_0(x)] < \delta\}
$$

- **强 $\delta$ 邻域**：$y_0(x)$ 的零阶 $\delta$ 邻域；
- **弱 $\delta$ 邻域**：$y_0(x)$ 的一阶 $\delta$ 邻域.

**$n$ 阶的 $\delta$ 接近度**

$$
y(x) \in N_n[\delta, y_0(x)]
$$

<img src="./images/zero-order-vs-first-order-approximation.png" width="400">

零阶的 $\delta$ 接近度使容许函数可以更自由地变化（选择），更有普遍性，所以叫“强”。

#### 连续泛函

对于 $y(x) \in F = C^n[a,b]$，$F$ 是 $J[y(x)]$ 的定义域，对于 $\forall \varepsilon > 0, \ \exists \delta > 0$，若

$$
d_n[y(x), y_0(x)] < \delta \Leftrightarrow N_n[\delta, y_0(x)] \subset F
$$

有

$$
|J[y(x)] - J[y_0(x)]| < \varepsilon
$$

成立，则 $J[y(x)]$ 在 $y_0(x)$ 处具有 $n$ 阶 $\delta$ 接近度的连续泛函.

#### 泛函的极值

| 极值类型               | 定义条件                                                    |
| :--------------------- | :---------------------------------------------------------- |
| **绝对（全局）极小值** | $\forall y(x) \in F,\ \Delta J = J[y(x)] - J[y_0(x)] \ge 0$ |
| **绝对（全局）极大值** | $\forall y(x) \in F,\ \Delta J = J[y(x)] - J[y_0(x)] \le 0$ |
| **强相对极小值**       | $y(x)$ 在 $y_0(x)$ 的零阶 $\delta$ 邻域内，$\Delta J \ge 0$ |
| **强相对极大值**       | $y(x)$ 在 $y_0(x)$ 的零阶 $\delta$ 邻域内，$\Delta J \le 0$ |
| **弱相对极小值**       | $y(x)$ 在 $y_0(x)$ 的一阶 $\delta$ 邻域内，$\Delta J \ge 0$ |
| **弱相对极大值**       | $y(x)$ 在 $y_0(x)$ 的一阶 $\delta$ 邻域内，$\Delta J \le 0$ |

**绝对极值、强极值和弱极值之间的关系**

<img src="./images/abs-strong-weak-extremum-relations.png" width="150">

#### 变分

$$
\forall x \in [x_0, x_1], \ \delta y = \bar{y}(x) - y_0(x) = \varepsilon \eta(x)
$$

- $\delta y$：函数 $y$ 的变分，$\delta$ 是变分符号；
- $\varepsilon$：小参数（不是 $x$ 的函数）；
- $\eta(x)$：任意函数.

<img src="./images/var-vs-diff.jpg" width="300">

#### 固定边界条件

$$
\eta(x_0) = \eta(x_1) = 0 \Leftrightarrow \delta y(x_0) = \delta y(x_1) = 0
$$

**强变分**：$y(x)$ 与 $y_{0}(x)$ 具有零阶接近度.

**弱变分**：$y(x)$ 与 $y_{0}(x)$ 具有一阶接近度.

#### 全变分

$$
\Delta y = \delta y + y'(x)\Delta x
$$

- $\Delta$ 是全变分符号

#### 导数的变分

求变分与求导数可交换

$$
\delta y' = \bar{y}'(x) - y_0'(x) = [\bar{y}(x) - y_0(x)]' = (\delta y)'
\\
\delta \frac{\mathrm{d}y}{\mathrm{d}x} = \frac{\mathrm{d}}{\mathrm{d}x}\delta y
\\
\delta y^{(n)} = (\delta y)^{(n)}
$$

哈密顿算子与变分符号可交换

$$
\delta \nabla \varphi = \nabla \delta \varphi \\
\delta \nabla \cdot \boldsymbol{a} = \nabla \cdot \delta \boldsymbol{a} \\
\delta \nabla \times \boldsymbol{a} = \nabla \times \delta \boldsymbol{a}
$$

拉普拉斯算子与变分符号可交换

$$
\delta \Delta \varphi = \Delta \delta \varphi
$$

### 2.3 最简泛函的变分与极值的必要条件

#### 最简泛函

**最简泛函**

$$
J[y(x)] = \int_{x_0}^{x_1} \underbrace{F(\underbrace{x,y(x),y'(x)}_{\text{三个独立变量}})}_{\text{被积函数（拉格朗日函数）}} \mathrm{d}x
$$

**线性泛函**

$$
J[y_1(x) + y_2(x)] = J[y_1(x)] + J[y_2(x)]
\\
J[cy(x)] = cJ[y(x)]
$$

**对称双线性泛函**

$$
\begin{aligned}
&\hphantom{J[\alpha_1 u_1 + \alpha_2 u_2, v]} \llap{J[u, v] ={}} J[v, u] \quad \text{（对称）} \\
&\left.
\begin{aligned}
J[\alpha_1 u_1 + \alpha_2 u_2, v] &= \alpha_1 J[u_1, v] + \alpha_2 J[u_2, v] \\
J[u, \beta_1 v_1 + \beta_2 v_2] &= \beta_1 J[u, v_1] + \beta_2 J[u, v_2]
\end{aligned}
\right\} \text{ 双线性}
\end{aligned}
$$

**二次泛函**

令 $u=v$，则可得到二次泛函 $$J[u, u]$$.

#### 泛函的变分

第一种定义方法

$$
\Delta J = L[y(x), \delta y] + \underbrace{d[y(x), \delta y]}_{\delta y\text{ 的高阶无穷小量}}
\\
\delta J = L[y, \delta y] = \int_{x_0}^{x_1} [F_y(x, y, y')\delta y + F_{y'}(x, y, y')\delta y'] \mathrm{d}x = \varepsilon \int_{x_0}^{x_1} [F_y \eta + F_{y'} \eta'] \mathrm{d}x
$$

- $\Delta J - \delta J$ 是一个比 $d_{1} (y, y_{1})$ 更高阶的无穷小，$\delta J$ 是 $\Delta J$ 的线性主部；
- $\delta J$ 的被积函数是关于 $\eta$ 和 $\eta'$ 的线性函数；
- 泛函的线性化：$\Delta J = \delta J$.

**法拉格朗日定义的泛函变分**

$$
\Phi(\varepsilon) = J[y+\varepsilon\delta y] = \int_{x_0}^{x_1} F(x,y+\varepsilon\delta y,y'+\varepsilon\delta y')\mathrm{d}x
$$

$$
\delta J = \Phi'(0) = \frac{\partial J[y(x)+\varepsilon\delta y]}{\partial \varepsilon}\bigg|_{\varepsilon=0}
$$

- $\varepsilon=0$ 时，$y_1(x)=y(x)$；
- $\varepsilon=1$ 时，$y_1(x)=y(x)+\delta y$；
- **容许曲线**：与 $y(x)$ 接近的曲线 $y_1(x)$ 称为 $y(x)$ 的容许曲线.

#### 函数的变分

$$
\delta F = F_y \delta y + F_{y'} \delta y'
$$

#### 函数与泛函的区别

|      | 符号记法  | 变量        | 自变量          | 增量       | 微分           |
| ---- | --------- | ----------- | --------------- | ---------- | -------------- |
| 函数 | $f(x)$    | $y=f(x)$    | $x$             | $\Delta x$ | $\mathrm{d} y$ |
| 泛函 | $J[y(x)]$ | $J=J[y(x)]$ | $y(x)$ （函数） | $\delta y$ | $\delta J$     |

#### 变分符号的基本运算性质

- $\delta(F_1+F_2) = \delta F_1 + \delta F_2$
- $\delta(F_1F_2) = F_1\delta F_2 + F_2\delta F_1$（前变后不变，前不变后变）
- $\delta(F^n) = nF^{n-1}\delta F$
- $\delta\left(\dfrac{F_1}{F_2}\right) = \dfrac{F_2\delta F_1 - F_1\delta F_2}{F_2^2}$（上变下不变减下变上不变，分母取平方）
- $\delta(F^{(n)}) = (\delta F)^{(n)}$，$F^{(n)} = \dfrac{\mathrm{d}^n F}{\mathrm{d}x^n}$
- $\delta\displaystyle\int_{x_0}^{x_1} F(x,y,y')\mathrm{d}x = \int_{x_0}^{x_1}\delta F(x,y,y')\mathrm{d}x$（$\delta F$ 是 $\delta y$ 和 $\delta y'$ 的线性函数时）

#### 全变分符号的基本运算性质

- $\Delta(F_1+F_2) = \Delta F_1 + \Delta F_2$
- $\Delta(F_1F_2) = F_1\Delta F_2 + F_2\Delta F_1$
- $\Delta(F^n) = nF^{n-1}\Delta F$
- $\Delta\left(\dfrac{F_1}{F_2}\right) = \dfrac{F_2\Delta F_1 - F_1\Delta F_2}{F_2^2}$
- $(\Delta F^{(n)})' = \Delta(F^{(n+1)}) + F^{(n+1)}(\Delta x)'$（微分与全变分符号不可互换）
- $\Delta\displaystyle\int_{x_0}^{x_1} F\mathrm{d}x = \delta\int_{x_0}^{x_1} F\mathrm{d}x + (F\Delta x)\big|_{x_0}^{x_1} = \int_{x_0}^{x_1}\left(\Delta F + F\dfrac{\mathrm{d}}{\mathrm{d}x}\Delta x\right)\mathrm{d}x$（当 $\dfrac{\mathrm{d}}{\mathrm{d}x}\Delta x \neq 0$ 时，积分与全变分不可互换顺序）
- $\mathrm{d}(\Delta x) = \Delta(\mathrm{d}x)$

#### 泛函极值定理

若 $J[y(x)]$ 在 $y=y(x)$ 上达到极值，则在 $y=y(x)$ 上的变分 $\delta J = 0$，即变分原理.

#### 泛函极值的必要条件/驻值条件

拉格朗日变换

$$
\delta J = \int_{x_0}^{x_1}(F_y\delta y + F_{y'}\delta y')\mathrm{d}x \xlongequal{\text{分部积分}} \int_{x_0}^{x_1}\left(F_y - \frac{\mathrm{d}}{\mathrm{d}x}F_{y'}\right)\delta y\mathrm{d}x
$$

黎曼变换

$$
\delta J = \int_{x_0}^{x_1}(F_{y'} - N)\delta y'\mathrm{d}x, \quad N(x) = \int_{x_0}^{x} F_y\mathrm{d}x
$$

### 2.4 最简泛函的欧拉方程

#### 最简泛函的极值必要条件

使最简泛函取极值且满足固定边界条件 $y(x_0)=y_0, y(x_1)=y_1$ 的极值曲线 $y=y(x)$ 应满足必要条件（对拉格朗日变换运用变分法引理），即欧拉-拉格朗日方程：

$$
F_y - \frac{\mathrm{d}}{\mathrm{d}x}F_{y'} = 0, \quad F(x, y, y') \in C^2 [x_0, x_1]
$$

也可以写成

$$
F_y - F_{xy'} - F_{yy'}y' - F_{y'y'}y'' = 0
$$

- 欧拉方程不一定是微分方程.

**解题步骤**

若欧拉方程中 $F_{y'y'} \neq 0$，则变分问题化为如下的微分方程的边值问题

$$
\begin{cases}
F_y - \dfrac{\mathrm{d}}{\mathrm{d}x}F_{y'} = 0 \\[6pt]
y(x_0) = y_0,\ y(x_1) = y_1
\end{cases}
\Rightarrow \underbrace{y = y(x, c_1, c_2)}_{\text{极值曲线族(簇)}}
$$

### 2.5 欧拉方程的几种特殊类型及其积分

1. $F = F(x, y)$ 或 $F = F(y)$：因为欧拉方程的解为 $F_y(x, y) = 0$ 或 $F_y(y) = 0$ ，除非 $F_y(x, y) = 0$ 或 $F_y(y) = 0$ 的解过边界点，通常无解；
2. $F$ 线性地依赖 $y'$：$F(x, y, y') = M(x, y) + N(x, y)y'$，泛函为定值，欧拉方程不是微分方程，解无意义；
3. $F$ 不依赖 $y$：即 $F = F(x, y')$，$F_{y'}(x, y') = c$，解出 $y' = \varphi(x, c_1)$，$y = \int_{x_0}^{x} \varphi(x, c_1)\mathrm{d}x$；
4. $F$ 只依赖于 $y'$，即 $F = F(y')$，极值曲线必是直线族；
5. $F$ 仅依赖于 $y$ 和 $y'$，即 $F = F(y, y')$，有首次积分 $F - y'F_{y'} = c$，解出 $y' = \varphi(y, c_1)$，$x = \int \frac{\mathrm{d}y}{\varphi(y, c_1)} + c_2$.

### 2.6 依赖于多个一元函数的变分问题

泛函

$$
J[y(x), z(x)] = \int_{x_0}^{x_1} F(x, y, y', z, z')\mathrm{d}x
$$

满足固定边界条件

$$
y(x_0) = y_0,\ y(x_1) = y_1,\ z(x_0) = z_0,\ z(x_1) = z_1
$$

则极值曲线 $y=y(x)$，$z=z(x)$ 必满足欧拉方程组

$$
\begin{cases}
F_y - \dfrac{\mathrm{d}}{\mathrm{d}x}F_{y'} = 0 \\[6pt]
F_z - \dfrac{\mathrm{d}}{\mathrm{d}x}F_{z'} = 0
\end{cases}
$$

### 2.7 依赖于高阶导数的变分问题

#### 依赖于一元函数的二阶导数的变分问题

泛函

$$
J[y(x)] = \int_{x_0}^{x_1} F(x, y, y', y'')\mathrm{d}x, \quad F \in C^3[x_0, x_1]
$$

满足固定边界条件

$$
y(x_0) = y_0,\ y(x_1) = y_1,\ y'(x_0) = y_0',\ y'(x_1) = y_1'
$$

则极值曲线 $y=y(x)$ 必满足欧拉-泊松方程

$$
F_y - \frac{\mathrm{d}}{\mathrm{d}x}F_{y'} + \frac{\mathrm{d}^2}{\mathrm{d}x^2}F_{y''} = 0
$$

#### 依赖于一元函数的高阶导数的变分问题

泛函

$$
J[y] = \int_{x_0}^{x_1} F(x, y, y', \cdots, y^{(n)})\mathrm{d}x
$$

满足固定边界条件

$$
y^{(k)}(x_0) = y_0^{(k)}, \ y^{(k)}(x_1) = y_1^{(k)} \quad (k = 0, 1, \cdots, n-1)
$$

则极值曲线 $y=y(x)$ 必满足欧拉-泊松方程

$$
\sum_{k=0}^{n}(-1)^k \frac{\mathrm{d}^k}{\mathrm{d}x^k}F_{y^{(k)}} = 0
$$

#### 依赖于两个一元函数的不同阶导数的变分问题

泛函

$$
J[y(x), z(x)] = \int_{x_0}^{x_1} F(x, y, y', \cdots, y^{(m)}, z, z', \cdots, z^{(n)})\mathrm{d}x
$$

满足固定边界条件

$$
y^{(k)}(x_0) = y_0^{(k)},\ y^{(k)}(x_1) = y_1^{(k)}\ (k = 0, 1, \cdots, m-1)
$$

$$
z^{(k)}(x_0) = z_0^{(k)},\ z^{(k)}(x_1) = z_1^{(k)}\ (k = 0, 1, \cdots, n-1)
$$

则极值曲线 $y=y(x)$，$z=z(x)$ 必满足欧拉-泊松方程组

$$
\begin{cases}
\displaystyle\sum_{k=0}^{m}(-1)^k \frac{\mathrm{d}^k}{\mathrm{d}x^k}F_{y^{(k)}} = 0 \\[12pt]
\displaystyle\sum_{k=0}^{n}(-1)^k \frac{\mathrm{d}^k}{\mathrm{d}x^k}F_{z^{(k)}} = 0
\end{cases}
$$

#### 依赖于多个一元函数的不同阶导数的变分问题

泛函

$$
J[y_1(x), y_2(x), \cdots, y_m(x)] = \int_{x_0}^{x_1} F(x, y_1, y_1', \cdots, y_1^{(n_1)}, y_2, y_2', \cdots, y_2^{(n_2)}, \cdots, y_m, y_m', \cdots, y_m^{(n_m)}) \, \mathrm{d}x
$$

满足固定边界条件

$$
y_i^{(k)}(x_0) = y_{i0}^{(k)}, \  y_i^{(k)}(x_1) = y_{i1}^{(k)} \quad  (i=1,2,\cdots,m; \ k=0,1,\cdots,n_i-1)
$$

则极值曲线 $y_{i} = y_{i} (x) \quad (i=1, 2, \dots, m)$ 必满足欧拉-泊松方程组

$$
\sum_{k=0}^{n_i} (-1)^k \frac{\mathrm{d}^k}{\mathrm{d}x^k} F_{y_i^{(k)}} = 0
$$

#### 欧拉-泊松方程的4种特殊情况

1. $F$ 不依赖于 $y$，$F_y = 0$，有首次积分

$$
\sum_{k=1}^{n} (-1)^{k-1} \frac{\mathrm{d}^{k-1}}{\mathrm{d}x^{k-1}} F_{y^{(k)}} = C
$$

2. $F$ 不依赖于 $x$，令 $x^{(i)} = \frac{\mathrm{d}^i y}{\mathrm{d}x^i}$，改写被积函数，得到首次积分

$$
F(y, \frac{1}{x'}, \frac{x''}{x'^3}, \frac{3x''^2 - x'x'''}{x'^5}, \cdots) = \varphi(y, x', \cdots, x^{(n)})
\\
\sum_{k=1}^{n} (-1)^{k-1} \frac{\mathrm{d}^{k-1}}{\mathrm{d}x^{k-1}} \varphi_{x^{(k)}} = C
$$

3. $F$ 仅依赖于 $y^{(n)}$，则有

   $$
   y = \underbrace{\iint \cdots \int}_{n} f[\underbrace{P_{(n-1)}(x)}_{n-1 \ \text{次多项式}}] (\mathrm{d}x)^n + \underbrace{Q_{(n-1)}(x)}_{n-1 \ \text{次多项式}}
   $$

4. 若积分号下是某个函数的全微分，则变分问题无意义.

### 2.8 依赖于多元函数的变分问题

#### 依赖于二元函数一阶偏导数的变分问题

设 $D$ 是平面区域，$(x,y) \in D$，$u(x,y) \in C^2(D)$，使泛函

$$
J[u(x,y)] = \iint_D F(x,y,u,u_x,u_y) \, \mathrm{d}x\mathrm{d}y
$$

取极值且在区域 $D$ 的边界线 $L$ 上取已知的极值函数 $u=u(x,y)$ 必满足**奥斯特罗格拉茨基方程（奥氏方程）**

$$
F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} = 0
$$

其中，对自变量的完全偏导数为

$$
\frac{\partial}{\partial x} F_{u_x} = F_{u_x x} + F_{u_x u} u_x + F_{u_x u_x} u_{xx} + F_{u_x u_y} u_{yx}
$$

$$
\frac{\partial}{\partial y} F_{u_y} = F_{u_y y} + F_{u_y u} u_y + F_{u_y u_x} u_{xy} + F_{u_y u_y} u_{yy}
$$

代入可得

$$
F_{u_x u_x} u_{xx} + 2F_{u_x u_y} u_{xy} + F_{u_y u_y} u_{yy} + F_{u_x u} u_x + F_{u_y u} u_y + F_{u_x x} + F_{u_y y} - F_u = 0
$$

#### 依赖于二元函数二阶偏导数的变分问题

设 $D$ 是平面区域，$(x,y) \in D$，$u(x,y) \in C^4(D)$，$F(x,y,u,u_x,u_y,u_{xx},u_{xy},u_{yy}) \in C^3(D)$，泛函

$$
J[u(x,y)] = \iint_D F(x,y,u,u_x,u_y,u_{xx},u_{xy},u_{yy}) \, \mathrm{d}x\mathrm{d}y
$$

的奥氏方程

$$
F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} + \frac{\partial^2}{\partial x^2} F_{u_{xx}} + \frac{\partial^2}{\partial x \partial y} F_{u_{xy}} + \frac{\partial^2}{\partial y^2} F_{u_{yy}} = 0
$$

#### 依赖于二元函数的高阶偏导数的变分问题

设 $D$ 是平面区域，$(x,y) \in R$，$u(x,y) \in C^{2n}(D)$，$F \in C^{n+1}(D)$，泛函

$$
J[u(x,y)] = \iint_D F(x,y,u,u_x,u_y,u_{xx},u_{xy},u_{yy},\cdots,u_{\underbrace{xx\cdots x}_{n}},\cdots,u_{\underbrace{yy\cdots y}_{n}}) \, \mathrm{d}x\mathrm{d}y
$$

奥氏方程

$$
F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} + \frac{\partial^2}{\partial x^2} F_{u_{xx}} + \frac{\partial^2}{\partial x \partial y} F_{u_{xy}} + \frac{\partial^2}{\partial y^2} F_{u_{yy}} + \cdots + (-1)^n \left( \frac{\partial^n}{\partial x^n} F_{u_{\underbrace{xx\cdots x}_{n}}} + \frac{\partial^n}{\partial x^{n-1} \partial y} F_{u_{\underbrace{xx\cdots x}_{n-1}y}} + \cdots + \frac{\partial^n}{\partial y^n} F_{u_{\underbrace{yy\cdots y}_{n}}} \right) = 0
$$

**重调和方程/双调和方程**

$$
\frac{\partial^4 u}{\partial x^4} + 2 \frac{\partial^4 u}{\partial x^2 \partial y^2} + \frac{\partial^4 u}{\partial y^4} = 0
\quad \Leftrightarrow \quad
\Delta \Delta u
\quad \Leftrightarrow \quad
\Delta^2 u
$$

其中 $u$ 称为双调和函数.

#### 依赖于两个二元函数一阶偏导数的变分问题

设 $D$ 是平面区域，$(x,y) \in D$，$u(x,y) \in C^2$，$v(x,y) \in C^2$，泛函

$$
J[u(x,y), v(x,y)] = \iint_D F(x,y,u,v,u_x,v_x,u_y,v_y) \, \mathrm{d}x\mathrm{d}y
$$

奥氏方程组

$$
\begin{cases}
F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} = 0 \\
F_v - \frac{\partial}{\partial x} F_{v_x} - \frac{\partial}{\partial y} F_{v_y} = 0
\end{cases}
$$

#### 依赖于多元函数的一阶偏导数的变分问题

设 $\Omega$ 是 $n$ 维空间区域，$(x_1, x_2, \ldots, x_n) \in \Omega$，$u(x_1, x_2, \ldots, x_n) \in C^{2n}$，泛函

$$
J[u(x_1, x_2, \ldots, x_n)] = \int_\Omega F(x_1, x_2, \ldots, x_n, u, u_{x_1}, u_{x_2}, \ldots, u_{x_n}) \mathrm{d}x_1 \mathrm{d}x_2 \cdots \mathrm{d}x_n
$$

奥式方程

$$
F_u - \sum_{i=1}^n \frac{\partial}{\partial x_i}F_{u_{x_i}} = 0
$$

依赖于平面域与时间域的合成域的变分问题

设 $D + T$ 是平面区域和时间区域的合成域，$t \in T = [t_0, t_1]$，$(x, y) \in D$，$(x, y, t) \in D + T$，$u(x, y, t) \in C^2(D + T)$，$F(x, y, u, u_x, u_y, u_{xx}, u_{xy}, u_{yy}, u_t) \in C^3$，泛函

$$
J[u(x, y, t)] = \int_{t_0}^{t_1} \iint_D F(x, y, u, u_x, u_y, u_{xx}, u_{xy}, u_{yy}, u_t) \mathrm{d}x \mathrm{d}y \mathrm{d}t
$$

奥氏方程

$$
F_u - \frac{\partial}{\partial x}F_{u_x} - \frac{\partial}{\partial y}F_{u_y} - \frac{\partial}{\partial t}F_{u_t} + \frac{\partial^2}{\partial x^2}F_{u_{xx}} + \frac{\partial^2}{\partial x \partial y}F_{u_{xy}} + \frac{\partial^2}{\partial y^2}F_{u_{yy}} = 0
$$

### 2.9 完全泛函的变分问题

偏微分算子

$$
D^{i_s} = \frac{\partial^{i_1+i_2+\cdots+i_m}}{\partial x_1^{i_1} \partial x_2^{i_2} \cdots \partial x_m^{i_m}}, \quad i_s = i_1 + \cdots + i_m
$$

设 $\Omega$ 为 $m$ 维域，$(x_1, x_2, \ldots, x_m) \in \Omega$，$u(x_1, x_2, \ldots, x_m) \in C^{2n}$，泛函

$$
\begin{align*}
J[u] &= \int_\Omega F(x_1, \ldots, x_m, u, u_{x_1}, \ldots, u_{x_m}, u_{x_1 x_1}, \ldots, u_{x_m x_m}, \ldots, u_{\underbrace{x_1 \cdots x_1}_{i_s}}, \ldots, u_{\underbrace{x_1 \cdots x_1}_{n}}) \mathrm{d}x_1 \mathrm{d}x_2 \cdots \mathrm{d}x_m
\\
&= \int_\Omega F(x_1, \ldots, x_m, u, D^{i_1}u, \ldots, D^{i_s}u, \ldots, D^n u) \mathrm{d}x_1 \mathrm{d}x_2 \cdots \mathrm{d}x_m
\end{align*}
$$

方程

$$
F_u + \sum_{s=1}^n (-1)^{i_s} D^{i_s} F_{D^{i_s}u} = 0
$$

#### 完全泛函的极值函数定理

设 $\Omega$ 为 $m$ 维域，$(x_1, x_2, \ldots, x_m) \in \Omega$，$u_k(x_1, x_2, \ldots, x_m) \in C^{2n_k}$，$k = 1,2,\dots,l$，完全泛函

$$
\begin{align*}
J[u_1, u_2, \ldots, u_l] = \int_\Omega F ( & x_1, \ldots, x_m, \\
& u_1, D^{i_{1_{1}}}u_1, \ldots, D^{i_{s_1}}u_1, \ldots,D^{n_1}u_1, \ldots, \\
& u_k, D^{i_{1_{k}}}u_k, \ldots, D^{i_{s_k}}u_k, \ldots, D^{n_k}u_k , \ldots\\
& u_l, D^{i_{1_l}} u_{l}, \ldots, D^{i_{s_l}} u_{l} , \ldots, D^{{n_l}} u_{l} )
\mathrm{d}x_1 \mathrm{d}x_2 \cdots \mathrm{d}x_m
\end{align*}
$$

完全欧拉方程组

$$
F_{u_k} + \sum_{s_k=1}^{S_k} (-1)^{i_{s_k}} D^{i_{s_k}} F_{D^{i_{s_k}}u_k} = 0
$$

### 2.10 欧拉方程的不变性

在已给的变分问题中，对自变量进行某种变换，变换后的泛函的欧拉方程与原来的欧拉方程形式上一样，这种性质称为欧拉方程的不变性.

## 3 泛函极值的充分条件

### 3.1 极值曲线场

<img src="./images/ext-field.png" width="500">

### 3.2 雅可比条件和雅可比方程

#### 雅可比方程

泛函 $J$ 的二阶变分 $\delta^2 J = \frac{\varepsilon^2}{2} J_2$ 中，泛函 $J_2$

$$
J_2 = \int_{x_0}^{x_1} (F_{yy}\eta^2 + 2F_{yy'}\eta\eta' + F_{y'y'}\eta'^2) \mathrm{d}x
$$

取得极值 $\eta = u(x)$ 必满足**雅可比方程**

$$
(F_{yy} - \frac{\mathrm{d}}{\mathrm{d}x}F_{yy'})u - \frac{\mathrm{d}}{\mathrm{d}x}(F_{y'y'}\frac{\mathrm{d}u}{\mathrm{d}x}) = 0
$$

#### 雅可比条件

- 若 $\eta(x_0) = \eta(x_1) = 0$，且 $u = u(x)$ 为雅可比方程的解，在区间 $(x_0, x_1)$ 内 $u \neq 0$，且 $u$ 在 $(x_0, x_1)$ 内不恒为零，则 $J_2$ 可写为

$$
J_2 = \int_{x_0}^{x_1} F_{y'y'} (\eta' - \eta \frac{u'}{u})^2 \mathrm{d}x
$$

- 若 $u(x)$ 为雅可比方程的解，$u(x_0) = 0$，除 $x_0$ 外，设 $u(x) = 0$ 的根为 $x^*$，则 $x^*$ 为 $x_0$ 的**共轭值**，点 $A_c(x^*, y(x^*))$ 称为极值曲线 $y = y(x)$ 上点 $A$ 的**共轭点.**

- **雅可比条件**：设 $y = u(x)$ 是雅可比方程满足边界条件 $u(x_0) = 0$，$u'(x_0) = 1$ 的解，若 $u(x)$ 在 $[x_0, x_1)$ 上除 $x_0$ 外无其他零点，则 $J[y(x)]$ 的极值曲线 $y(x)$ 称为在 $(x_0, x_1)$ 内满足雅可比条件。把 $[x_0, x_1)$ 换成 $[x_0, x_1]$，则称在 $(x_0, x_1]$ 内满足**雅可比强条件**。

### 3.3 魏尔斯特拉斯函数与魏尔斯特拉斯条件

#### 希尔伯特不变积分

泛函的增量为

$$
\Delta J = \int_{\bar{C}} F(x,y,y')\,\mathrm{d}x - \int_C F(x,y,y')\,\mathrm{d}x
$$

其中，$C$ 是极值曲线，$\bar{C}$ 是邻近的可取曲线.

辅助泛函（希尔伯特不变积分）

$$
H[\bar{C}] = \int_{\bar{C}} [F(x,y,p) + (y'-p)F_p(x,y,p)]\,\mathrm{d}x
$$

其中，$p$ 是极值曲线场中所考察点在此点处的斜率.

性质：

- 当 $\bar{C} = C$ 时，有 $y' = p(x,y)$，所以 $H[\bar{C}] = J[y(x)]$；
- $H[\bar{C}]$ 是某一函数的全微分的积分.

#### 魏尔斯特拉斯函数（E 函数）

$$
\Delta J = \int_{\bar{c}} \underbrace{[F(x,y,y') - F(x,y,p) - (y'-p)F_p(x,y,p)]}_{E(x,y,y',p)} \,\mathrm{d}x
$$

性质：

- $\Delta J$ 与 $E$ 同号；
- $E > 0$ 时，$J[y(x)]$ 取极小值；$E < 0$ 时，$J[y(x)]$ 取极大值；
- 当 $y = y(x)$ 是极值曲线时，$y' = p$，此时 $E = 0$.

#### 魏尔斯特拉斯条件

**弱魏尔斯特拉斯条件**：$y = y(x)$ 是最简泛函满足固定边界条件的极值曲线，若对 $y = y(x)$ 近旁所有的点 $(x,y)$ 和近于极值曲线斜率函数 $p(x,y)$ 的 $y'$ 值，有 $E(x,y,y',p) \geq 0$（或 $\leq 0$）；

**强魏尔斯特拉斯条件**：将弱条件中，换成对于任意的 $y'$ 值，都成立.

### 3.4 勒让德条件

用下式替代魏尔斯特拉斯条件（因为魏尔斯特拉斯条件一般难以计算）

$$
F_{y'y'}(x,y,y') \geq 0 \quad (\text{或} \leq 0 )
$$

若上式是严格不等式，则是**勒让德强条件**.

已知最简泛函满足固定边界条件，$F(x,y,y') \in C^2[x_0, x_1]$，若：

1. $u(x)$ 为雅可比方程的解，且 $u(x) \neq 0$，$x \in (x_0, x_1)$；

2. $x$ 在 $[x_0, x_1]$ 内有 $y' - \eta \frac{u'}{u} \neq 0$；

3. $F_{y'y'}$ 在区间 $[x_0, x_1]$ 内不变号.

那么当 $F_{y'y'} > 0$ 时，$J[y]$ 取得弱极小值；当 $F_{y'y'} < 0$ 时，$J[y]$ 取得弱极大值.

### 3.5 泛函极值的充分条件

对于满足边界条件的泛函：

若极值曲线包含在极值曲线场中或满足雅可比条件，且弱魏尔斯特拉斯条件成立，即 $E$ 不变号，则 $E \geq 0$ 时，弱极小值；$E \leq 0$ 时，弱极大值.

若极值曲线包含在极值曲线场中或满足雅可比条件，且强魏尔斯特拉斯条件成立，即 $E$ 不变号，则 $E \geq 0$ 时，强极小值；$E \leq 0$ 时，强极大值.

满足边界条件的泛函，若极值曲线包含在区域为 $D$ 的极值曲线场中，且强魏尔斯特拉斯条件成立，则 $E \geq 0$ 时，绝对极小值；$E \leq 0$ 时，绝对极大值.

若其极值曲线满足雅可比条件，且极值曲线上勒让德条件成立，则 $F_{y'y'} > 0$ 时，弱极小值，$F_{y'y'} < 0$ 时，弱极大值.

若其极值曲线满足雅可比条件，且对极值曲线的某个零阶邻域内的所有点和任意的 $y'$ 上勒让德条件成立（$F_{y'y'} (x,y,q)$ 不变号），且函数 $F(x,y,y')$ 在 $y'=p$ 处的一阶泰勒公式成立，则 $F_{y'y'} (x,y,q) \geq 0$ 是，强极小值；$F_{y'y'} (x,y,q) \geq 0$，强极大值.

### 3.6 泛函的高阶变分

#### 函数的变分

一阶变分

$$
\delta F = F_y \delta y + F_{y'} \delta y'
$$

二阶变分

$$
\delta^2 F = \frac{1}{2} \left[ F_{yy} (\delta y)^2 + 2F_{yy'} \delta y \delta y' + F_{y'y'} (\delta y')^2 \right]
$$

#### 泛函的变分

二阶变分

$$
\delta^2 J = \int_{x_0}^{x_1} \delta^2 F \, \mathrm{d}x.
$$

$$
\delta^2 J = \Phi''(0) = \frac{\partial^2 J[y(x) + \varepsilon \delta y]}{\partial \varepsilon^2} \bigg|_{\varepsilon=0}, \quad  \Phi(\varepsilon) = J[y(x) + \varepsilon \delta y].
$$

$n$ 阶变分

$$
\delta^n J = \int_{x_0}^{x_1} \delta^n F \, \mathrm{d}x = \frac{1}{n!} \int_{x_0}^{x_1} \left[ \left( \delta y \frac{\partial}{\partial y} + \delta y' \frac{\partial}{\partial y'} \right)^n F \right] \mathrm{d}x
$$

$$
= \frac{1}{n!} \int_{x_0}^{x_1} \left[ \sum_{k=0}^{n} \binom{n}{k} \frac{\partial^n F}{\partial y^{n-k} \partial y'^k} (\delta y)^{n-k} (\delta y')^k \right] \mathrm{d}x.
$$

在极值曲线 $y = y(x)$ 上，$\delta^2 J \geq 0$（或 $\leq 0$），则弱极小值（或弱极大值）.

**二次泛函**：$F$ 是未知函数及其导数的二次方的泛函.

## 4 可动边界的变分问题

### 4.1 最简泛函的变分问题

<img src="./images/simplest-varprob.jpg" width="300">

可动边界的最简泛函

$$
J[y(x)] = \int_{x_0}^{x_1} F(x, y, y') \, \mathrm{d}x
$$

其中，可取曲线 $y = y(x) \in C^2$，两个端点 $A(x_0, y_0)$ 与 $B(x_1, y_1)$ 分别在 $y=\varphi(x)$ 与 $y=\psi(x)$ 上移动， $y(x), \varphi(x), \psi(x) \in C^2$.

若 $A$ 点固定，$B$ 点可动，$F_{y'}|_{x=x_1} = 0$ 是自然边界（运动边界）条件.

#### 自然边界条件与横截条件

> 对于下述结论，可以交换左右端点.

可动边界的最简泛函的极值曲线 $y=y(x)$ 左端点固定，右端点在直线 $x=x_1$ 上待定，自然边界条件为

$$
F_{y'}|_{x=x_1} = 0.
$$

可动边界的最简泛函的极值曲线 $y=y(x)$ 左端点固定，右端点在曲线 $y=\psi(x)$ 上待定，则横截条件

$$
\bigl[ F + (\psi' - y') F_{y'} \bigr] \big|_{x=x_1} = 0.
$$

可动边界的最简泛函的极值曲线 $y=y(x)$ 左端点固定，右端点所在的曲线有隐函数 $\Psi(x, y) = 0$，则横截条件

$$
\frac{F - y' F_{y'}}{F_{y'}} = \frac{\Psi_x}{\Psi_y}.
$$

可动边界的最简泛函的极值曲线 $y=y(x)$ 左右端点分别在 $y=\varphi(x)$ 与 $y=\psi(x)$ 上待定，则横截条件

$$
\begin{cases}
\bigl[ F + (\varphi' - y') F_{y'} \bigr] \big|_{x=x_0} = 0, \\
\bigl[ F + (\psi' - y') F_{y'} \bigr] \big|_{x=x_1} = 0.
\end{cases}
$$

---

### 4.2 含有多个函数的泛函的变分问题

边界条件的核心公式

$$
\left[ (F - y' F_{y'} - z' F_{z'}) \, \delta x + F_{y'} \delta y + F_{z'} \delta z \right] \vert_{x=x_1} = 0.
$$

> 哪个变分是任意的，哪个变分前面的系数就为0. 把已知条件的变分算出，并代回到上式. 最后与已知条件连理，解出未知参数.

1. 无关：若 $\delta x_1, \delta y_1, \delta z_1$ 相互无关，则 $\delta x_1, \delta y_1, \delta z_1$ 的系数均为 $0$，得出被积函数为 $0$，变分问题无意义；
2. 曲线：点都在 $\varphi(x_1)$ 上，$\psi(x_1)$ 上，则 $\delta y_1 = \varphi'(x_1) \delta x_1$，$\delta z_1 = \psi'(x_1) \delta x_1$（或 $\delta z_1 = \varphi_{x_1} \delta x_1 + \varphi_{y_1} \delta y_1$），代入后让 $\delta z_1$ 的系数为 $0$，得到横截条件

$$
[F + (\varphi' - y') F_{y'} + (\psi' - z') F_{z'}] \vert_{x=x_1} = 0
$$

3. 曲面：曲面上一点满足 $\Phi_{x_1} \delta x_1 + \Phi_{y_1} \delta y_1 + \Phi_{z_1} \delta z_1 = 0$，代入后让 $\delta x_1, \delta y_1, \delta z_1$ 的系数为 $0$，得到边界条件

$$
\begin{cases}
\left. \left( F - y' F_{y'} -z' F_{z'} - F_{z'} \dfrac{\Phi_x}{\Phi_z} \right)  \right|_{x=x_0} = 0, \\
\left. \left( F_{y'} - F_{z'} \dfrac{\Phi_x}{\Phi_z} \right) \right|_{x=x_1} = 0
\end{cases}
$$

4. 平面：平面 $x=x_1$ 上，$\delta y_1, \delta z_1$ 是任意的，则 $\delta y_1, \delta z_1$ 的系数为0，得到自然边界条件

$$
F_{y'} |_{x=x_1} = 0, \quad F_{z'} |_{x=x_1} = 0
$$

> 上述结论在右端点仍可互换

#### 多个一元函数的泛函变分的横截条件

泛函

$$
J[y_1, y_2, \cdots, y_n] = \int_{x_0}^{x_1} F(x, y_1, y_2, \cdots, y_n, y_1', y_2', \cdots, y_n') \, \mathrm{d}x
$$

的极值曲线的边界点 $B(x_1, y_{11}, y_{21}, \dots, y_{n1})$ 的横截条件是

$$
\left\{
\begin{array}{l}
\displaystyle \left. \left( F - \sum_{i=1}^{n} y_i' F_{y_i'} \right) \right|_{x=x_1} = 0 \\
\displaystyle F_{y_i'}\big|_{x=x_1} = 0 \quad (i=1,2,\cdots,n)
\end{array}
\right.
$$

#### 边界点在已知曲线上的多个一元函数的泛函变分的横截条件

若边界点 $B(x_1, y_{11}, y_{21}, \dots, y_{n1})$ 在已知曲线上 $y_i = \varphi_i(x_1)$，$i=1,2,\cdots,n$，则其横截条件是

$$
\left( F + \sum_{i=1}^{n} (\psi_i' - y_i') F_{y_i'} \right)\big|_{x=x_1} = 0
$$

### 4.3 含有高阶导数的泛函的变分问题

#### 泛函含有一个未知函数二阶导数的情形

核心公式

$$
\delta J = \left. \left[ F - y' \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) - y'' F_{y''} \right] \right|_{x=x_1} \delta x_1 + \left. \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) \right|_{x=x_1} \delta y_1 + F_{y''}\big|_{x=x_1} \delta y_1' = 0
$$

$J[y(x)]$ 在某一端点固定 $y(x_0) = y_0$，$y'(x_0) = y_0'$.

1. 另一可动端点 $(x_1, y_1)$ 在给定已知条件下，则 $\delta x_1, \delta y_1, \delta y_1'$ 的系数为 $0$，得自然边界条件

$$
\begin{cases}
\left. \left[ F - y' \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) - y'' F_{y''} \right] \right|_{x=x_1} = 0
\\
\left. \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) \right|_{x=x_1} = 0
\\
F_{y''}\big|_{x=x_1} = 0
\end{cases}
$$

2. 端点 $(x_1, y_1)$ 在曲线 $y_1 = \varphi(x_1)$ 上，且 $y_1' = \varphi'(x_1)$，则 $\delta y_1 = \varphi' \delta x_1$，$\delta y_1' = \varphi'' \delta x_1$，代入核心公式令 $\delta x_1$ 的系数为 $0$，得自然边界条件

$$
\left. \left[ F + (\varphi' - y') \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) + (\varphi' - y') F_{y''} \right] \right|_{x=x_1} = 0
$$

3. 端点 $(x_1, y_1)$ 满足关系式 $\varphi(x, y, y') = 0$，由 $\varphi_x \delta x + \varphi_y \delta y + \varphi_{y'} \delta y' = 0$ 代入，代入核心公式令变分前的系数为 $0$，得自然边界条件

$$
\left\{ \left[ F - y' \left( F_{y'} - \frac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right) - \left( y'' + \frac{\varphi_x}{\varphi_{y'}} \right) F_{y''} \right] \right\}_{x=x_1} = 0
$$

4. 端点 $(x_1, y_1)$ 在直线 $x = x_1$ 上自由移动，由 $\delta x_1 = 0$，代入核心公式令 $\delta y_1$ 和 $\delta y_1'$ 的系数为 $0$，得自然边界条件

$$
\begin{cases}
\left( F_{y'} - \dfrac{\mathrm{d}}{\mathrm{d}x} F_{y''} \right)\bigg|_{x=x_1} = 0 \\
F_{y''}\bigg|_{x=x_1} = 0
\end{cases}
$$

#### 泛函含有一个未知函数多阶导数的情形

泛函

$$
J[y(x)] = \int_{x_0}^{x_1} F(x, y, y', y'', \dots, y^{(k)}, \dots, y^{(n)}) \mathrm{d}x
$$

的一阶变分为

$$
\begin{align*}
\delta J =& \left\{ F - y' \left[ F_{y'} + \sum_{k=1}^{n-1} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+1)}}}{\mathrm{d}x^k} \right] - y'' \left[ F_{y''} + \sum_{k=1}^{n-2} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+2)}}}{\mathrm{d}x^k} \right] - \cdots \right. \\
& \left. - y^{(j)} \left[ F_{y^{(j)}} + \sum_{k=1}^{n-j} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+j)}}}{\mathrm{d}x^k} \right] - \cdots - y^{(n-1)} \left[ F_{y^{(n-1)}} - \frac{\mathrm{d}F_{y^{(n)}}}{\mathrm{d}x} \right] - y^{(n)} F_{y^{(n)}} \right\} \left. \vphantom{\sum_{k=1}^{n}} \right\vert_{x=x_1} \delta x_1 \\
& + \left[ F_{y'} + \sum_{k=1}^{n-1} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+1)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n}} \right\vert_{x=x_1} \delta y_1 + \left[ F_{y''} + \sum_{k=1}^{n-2} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+2)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n}} \right\vert_{x=x_1} \delta y'_1 + \cdots \\
& + \left[ F_{y^{(j)}} + \sum_{k=1}^{n-j} (-1)^k \frac{\mathrm{d}^k F_{y^{(k+j)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n}} \right\vert_{x=x_1} \delta y_1^{(j-1)} + \cdots \\
& + \left[ F_{y^{(n-1)}} - \frac{\mathrm{d}F_{y^{(n)}}}{\mathrm{d}x} \right] \left. \vphantom{\sum_{k=1}^{n}} \right\vert_{x=x_1} \delta y_1^{(n-2)} + F_{y^{(n)}} \left. \vphantom{F} \right\vert_{x=x_1} \delta y_1^{(n-1)} = 0
\end{align*}
$$

其横截条件即 $\delta x_1, \delta y_1, \delta y'_1, \delta y''_1, \dots, \delta y^{(n-1)}_1$ 前面的系数为 0 组成的方程组（此处不再赘述）.

#### 泛函含有多个未知函数多阶导数的情形

泛函

$$
J[y_1, y_2, \cdots, y_r] = \int_{x_0}^{x_1} F(x, y_1, y'_1, y''_1, \cdots, y_1^{(k)}, \cdots, y_1^{(n_1)}, y_2, y'_2, y''_2, \cdots, y_2^{(k)}, \cdots, y_2^{(n_2)}, \cdots, y_r, y'_r, y''_r, \cdots, y_r^{(k)}, \cdots, y_r^{(n_r)}) \mathrm{d}x
$$

的一阶变分为

$$
\begin{align*}
\delta J =&
\int_{x_0}^{x_1} \left\{
\left\{ F - y'_i \left[ F_{y'_i} + \sum_{k=1}^{n_i-1} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+1)}}}{\mathrm{d}x^k} \right] - y''_i \left[ F_{y''_i} + \sum_{k=1}^{n_i-2} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+2)}}}{\mathrm{d}x^k} \right] - \cdots \right. \right. \\
& \left. - y_i^{(j)} \left[ F_{y_i^{(j)}} + \sum_{k=1}^{n_i-j} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+j)}}}{\mathrm{d}x^k} \right] - \cdots - y_i^{(n_i-1)} \left[ F_{y_i^{(n_i-1)}} - \frac{\mathrm{d}F_{y_i^{(n_i)}}}{\mathrm{d}x} \right] - y_i^{(n_i)} F_{y_i^{(n_i)}} \right\} \left. \vphantom{\sum_{k=1}^{n_i}} \right\vert_{x=x_1} \delta x_1 \\
& + \left[ F_{y'_i} + \sum_{k=1}^{n_i-1} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+1)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n_i}} \right\vert_{x=x_1} \delta y_{i1} + \left[ F_{y''_i} + \sum_{k=1}^{n_i-2} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+2)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n_i}} \right\vert_{x=x_1} \delta y'_{i1} + \cdots \\
& + \left[ F_{y_i^{(j)}} + \sum_{k=1}^{n_i-j} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k+j)}}}{\mathrm{d}x^k} \right] \left. \vphantom{\sum_{k=1}^{n_i}} \right\vert_{x=x_1} \delta y_{i1}^{(j-1)} + \cdots + \left[ F_{y_i^{(n_i-1)}} - \frac{\mathrm{d}F_{y_i^{(n_i)}}}{\mathrm{d}x} \right] \left. \vphantom{\sum_{k=1}^{n_i}} \right\vert_{x=x_1} \delta y_{i1}^{(n_i-2)} \\
& \left. + F_{y_i^{(n_i)}} \left. \vphantom{F} \right\vert_{x=x_1} \delta y_{i1}^{(n_i-1)} + \int_{x_0}^{x_1} \left[ F_{y_i} + \sum_{k=1}^{n_i} (-1)^k \frac{\mathrm{d}^k F_{y_i^{(k)}}}{\mathrm{d}x^k} \right] \delta y_i  \right\} \mathrm{d}x = 0
\end{align*}
$$

### 4.4 含有多元函数的泛函的变分问题

<img src="./images/dom-var-bound.jpg" width="300">

泛函

$$
J[u(x,y)] = \iint_{D} F(x,y,u,u_x, u_y) \mathrm{d}x \mathrm{d}y
$$

奥氏方程

$$
F_u - \frac{\partial}{\partial x} F_{u_x} - \frac{\partial}{\partial y} F_{u_y} = 0, \quad (x,y) \in D
$$

补充边界条件

$$
\left. \left\{ F(x,y,u+\delta u, u_x+\delta u_x, u_y+\delta u_y) - \left[ F_{u_x} \cos(n_2, x) + F_{u_y} \cos(n_2, y) \right] \left( \frac{\partial u}{\partial n_2} - \frac{\partial u}{\partial n_1} \right) \right\} \right|_{\Gamma_2} = 0
$$

### 4.5 具有尖点的极值曲线

<img src="./images/cusp-curve.jpg" width="300">

极值必要条件 $\delta J = \delta J_- + \delta J_+ = 0$

埃德曼第一角点条件

$$
\frac{\partial F_-}{\partial y'}\big|_{x=x_c-0} = \frac{\partial F_+}{\partial y'}\big|_{x=x_c+0}
$$

埃德曼第二角点条件

$$
\left( F_- - y' \frac{\partial F_-}{\partial y'} \right)\big|_{x=x_c-0} = \left( F_+ - y' \frac{\partial F_+}{\partial y'} \right)\big|_{x=x_c+0}
$$

尖点极值曲线存在的必要条件

$$
F_{y'y'}(x_c, y_c, p) = F_{y'y'} = 0
$$

### 4.6 单侧变分问题

极值曲线与 $y=\varphi(x)$ 相切于 $M,N$，$J[y(x)]$ 在不等式 $y(x)\geqslant\varphi(x)$ 约束下，

$$
\text{泛函} \ \delta J = 0 \longrightarrow\text{欧拉方程}\longrightarrow\begin{cases}\text{切点}\\\text{切线}\\\text{边界条件}\end{cases}\longrightarrow\text{解出未知参数}\longrightarrow\text{极值曲线}
$$

## 5 条件极值的变分问题

### 5.1 完整约束的变分问题

泛函

$$
J[y]=\int_{x_0}^{x_1}F(x,y_1,y_2,\cdots,y_n,y_1',y_2',\cdots,y_n')\mathrm{d}x
$$

在约束条件

$$
\varphi_i(x,y_1,y_2,\cdots,y_n)=0\quad(i=1,2,\cdots,m;\,m<n)
$$

边界条件

$$
y_j(x_0)=y_{j0},\quad y_j(x_1)=y_{j1}\quad(j=1,2,\cdots,n)
$$

下的极值问题可以运用**拉格朗日定理**化为如下的泛函

$$
J^*[y]=\int_{x_0}^{x_1}\Big[F+\sum_{i=1}^m\lambda_i(x)\varphi_i\Big]\mathrm{d}x=\int_{x_0}^{x_1}H\mathrm{d}x
$$

其中，$\lambda_i(x)$ 成为拉格朗日乘数（乘子），$H$ 称为辅助泛函.

欧拉方程组

$$
H_{y_j}-\frac{\mathrm{d}}{\mathrm{d}x}H_{y_j'}=0\quad(j=1,2,\cdots,n)
$$

### 5.2 微分约束的变分问题

将 5.1 的约束条件改为

$$
\varphi_i(x,y_1,y_2,\cdots,y_n,\underbrace{y_1',y_2',\cdots,y_n'}_{\text{微分约束}})=0\quad(i=1,2,\cdots,m;\,m<n)
$$

后续步骤与 5.1 相同.

### 5.3 等周问题

将 5.1 的约束条件改为等周约束（积分约束）

$$
\int_{x_0}^{x_1}\varphi_i(x,y_1,y_2,\cdots,y_n,y_1',y_2',\cdots,y_n')\mathrm{d}x=a_i\quad(i=1,2,\cdots,m)
$$

运用拉格朗日定理

$$
J^*[y]=\int_{x_0}^{x_1}\Big(F+\sum_{i=1}^m\underbrace{\lambda_i}_{\text{常数}}\varphi_i\Big)\mathrm{d}x = \int_{x_0}^{x_1}G\mathrm{d}x
$$

欧拉方程组

$$
G_{y_j}-\frac{\mathrm{d}}{\mathrm{d}x}G_{y_j'}=0\quad(j=1,2,\cdots,n)
$$

**互易原理/对偶原理**

$$
H = F + \lambda\varphi \underset{\text{对称}}{\longleftrightarrow} H = \lambda_1 F + \lambda_2 \varphi
$$

例：在底边和面积一定的三角形中，等腰三角形的周长最短；在周长一定的三角形中，等腰三角形的面积最大。

### 5.4 混合型泛函的极值问题

#### 简单混合型泛函的极值问题

波尔查/拉格朗日问题/迈耶问题

> 将约束放到了积分号外面

$$
J=\int_{x_0}^{x_1}F(x,y,y')\mathrm{d}x+\Phi(x_0,y_0,x_1,y_1)
$$

仍然让变分为 $0$，得到

$$
\delta J = \big[F-y'(F_{y'}-\frac{\mathrm{d}}{\mathrm{d}x}F_{y''})-y''F_{y''}+\Phi_{x_1}\big]_{x=x_1}\delta x_1+\big(F_{y'}-\frac{\mathrm{d}}{\mathrm{d}x}F_{y''}+\Phi_{y_1}\big)_{x=x_1}\delta y_1+F_{y''}|_{x=x_1}\delta y_1'=0
$$

令 $\delta x_1,\delta y_1,\delta y_1'$ 前的系数为 $0$ 即可

> 可将 $x_0$ 与 $x_1$ 替换

#### 二维、三维和 $n$ 维问题的欧拉方程

混合型泛函

$$
J[u(x, y)] = \iint_D F(x,y,u,u_x, u_y) \mathrm{d}x \mathrm{d}y + \int_{\Gamma_2} G(x,y,u,u', u'', \dots, u^{(n)}) \mathrm{d} \Gamma
$$

**二维**

$$
F_u - \frac{\partial F_{u_x}}{\partial x} - \frac{\partial F_{u_y}}{\partial y} = 0 \quad (\text{在 } D \text{ 内})
$$

$$
\sum_{k=0}^{n} (-1)^k \frac{\mathrm{d}^k G_u^{(k)}}{\mathrm{d}\Gamma^k} + F_{u_x} n_x + F_{u_y} n_y = 0 \quad (\text{在 } \Gamma_2 \text{ 或 } \Gamma \text{ 上})
$$

> $G_u^{(k)}$ 表示有若干个.

其中，

$$
n_x = \frac{\mathrm{d}y}{\mathrm{d}\Gamma}, \quad n_y = \frac{\mathrm{d}x}{\mathrm{d}\Gamma}
$$

**三维**

$$
F_u - \frac{\partial F_{u_x}}{\partial x} - \frac{\partial F_{u_y}}{\partial y} - \frac{\partial F_{u_z}}{\partial z} = 0 \quad (\text{在 } V \text{ 内})
$$

$$
G_u + F_{u_x} n_x + F_{u_y} n_y + F_{u_z} n_z = 0 \quad (\text{在 } S_2 \text{ 或 } S \text{ 上})
$$

> $G_u$ 表示只有一个.

**$n$ 维**

$$
F_u - \sum_{i=1}^{n} \frac{\partial F_{u_{x_i}}}{\partial x_i} = 0 \quad (\text{在 } \Omega \text{ 内})
$$

$$
G_u + \sum_{i=1}^{n} F_{u_{x_i}} n_{x_i} = 0 \quad (\text{在 } S_2 \text{ 或 } S \text{ 上})
$$

## 6 参数形式的变分问题

### 6.1 曲线的参数形式及其齐次条件

#### $n$ 次齐次函数

$f(x, y, k\dot{x}, k\dot{y}) = k^n f(x, y, \dot{x}, \dot{y})$；其中，若 $k > 0$，则称正 $n$ 次齐次函数.

#### 欧拉齐次函数定理

若 $F(kx_1, kx_2, \cdots, kx_m) = k^n F(x_1, x_2, \cdots, x_m)$ 且 $F \in C^1$，则有

$$
\sum_{i=1}^{m} x_i F_{x_i}(x_1, x_2, \cdots, x_m) = n F(x_1, x_2, \cdots, x_m)
$$

#### 参数形式泛函的参数无关性定理

若 $J[x(t), y(t)] = \int_{t_0}^{t_1} F(x(t), y(t), \dot{x}(t), \dot{y}(t)) \mathrm{d}t$ 的被积函数中不显含 $t$，且对于 $\dot{x}(t), \dot{y}(t)$ 是一次齐次函数，则泛函的形式与参数的选择无关.

### 6.2 参数形式的等周问题和测地线

#### 欧拉方程的魏尔斯特拉斯形式

极值曲线 $x = x(t), y = y(t)$，满足欧拉方程（这一对方程不是相互独立的）

$$
\begin{cases}
F_x - \dfrac{\mathrm{d}}{\mathrm{d}t} F_{\dot{x}} = 0 \\[6pt]
F_y - \dfrac{\mathrm{d}}{\mathrm{d}t} F_{\dot{y}} = 0
\end{cases}
\Leftrightarrow \underbrace{F_{x\dot{y}} - F_{\dot{x}y} + (\dot{x}\ddot{y} - \ddot{x}\dot{y}) F_1 = 0}_{\text{欧拉方程的魏尔斯特拉斯形式}}
\Leftrightarrow
\underbrace{-\frac{1}{R} = \frac{F_{x\dot{y}} - F_{\dot{x}y}}{(\dot{x}^2 + \dot{y}^2)^{\frac{3}{2}} F_1} = \frac{y''}{(1+y'^2)^{\frac{3}{2}}} = \frac{\dot{x}\ddot{y} - \ddot{x}\dot{y}}{(\dot{x}^2 + \dot{y}^2)^{\frac{3}{2}}}}_{\text{欧拉方程的魏尔斯特拉斯形式}}
$$

其中，$F_1(x, y, \dot{x}, \dot{y}) = \dfrac{F_{\dot{x}\dot{x}}}{\dot{y}^2} = -\dfrac{F_{\dot{x}\dot{y}}}{\dot{x}\dot{y}} = \dfrac{F_{\dot{y}\dot{y}}}{\dot{x}^2}$，$R$ 是曲率半径，泛函取得极小值的必要条件是 $F_1 \geq 0$.

#### 曲面方程的情形

曲面 $\Sigma: \boldsymbol{r} = \boldsymbol{r}(u, v)$，$\boldsymbol{r}_u, \boldsymbol{r}_v \in C^1$ 且 $\boldsymbol{r}_u \times \boldsymbol{r}_v \neq 0$.

$$
\varphi_1 = (\mathrm{d}s)^2 = (\mathrm{d}\boldsymbol{r})^2 = E(\mathrm{d}u)^2 + 2F\,\mathrm{d}u\mathrm{d}v + G(\mathrm{d}v)^2
$$

其中，$E, F, G$ 是 $u, v$ 的函数，它们称为 $\Sigma$ 的第一基本量或度量张量.

$\Gamma$ 的弧长 $J[u, v] = \int_{t_0}^{t_1} \dfrac{\mathrm{d}s}{\mathrm{d}t} \mathrm{d}t = \int_{t_0}^{t_1} \sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2} \mathrm{d}t$.

该泛函的欧拉方程组

$$
\begin{cases}
\displaystyle \frac{E_u \dot{u}^2 + 2F_u \dot{u}\dot{v} + G_u \dot{v}^2}{\sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}} - \frac{\mathrm{d}}{\mathrm{d}t} \frac{2(E\dot{u} + F\dot{v})}{\sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}} = 0 \\[1.2em]
\displaystyle \frac{E_v \dot{u}^2 + 2F_v \dot{u}\dot{v} + G_v \dot{v}^2}{\sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}} - \frac{\mathrm{d}}{\mathrm{d}t} \frac{2(E\dot{u} + G\dot{v})}{\sqrt{E\dot{u}^2 + 2F\dot{u}\dot{v} + G\dot{v}^2}} = 0
\end{cases}
$$

### 6.3 可动边界参数形式泛函的极值

#### 端点具有一个曲线方程的情形

设可取曲线类 $C: x = x(t), y = y(t)$，它们有连续旋转的切线，端点分别在 $C_1: \varphi(x, y) = 0$ 和 $C_2 : \psi(x, y) = 0$.

泛函 $J[x(t), y(t)] = \int_{t_0}^{t_1} F(x(t), y(t), \dot{x}(t), \dot{y}(t)) \mathrm{d}t$ 的横截条件为

$$
\frac{F_{\dot{x}}}{\varphi_x} = \frac{F_{\dot{y}}}{\varphi_y} \quad (在位于 C_1 的端点上)
\\
\frac{F_{\dot{x}}}{\psi_x} = \frac{F_{\dot{y}}}{\psi_y} \quad (在位于 C_2 的端点上)
$$

#### 端点具有多个曲线方程的情形

设可取曲线类 $C: x = x(t), y_1 = y_1(t), y_2 = y_2(t), \cdots, y_n = y_n(t)$，它们有连续旋转的切线，端点分别在 $C_1: \varphi_i(x, y_i) = 0$ 和 $C_2: \psi_i(x, y_i) = 0$ $(i = 1, 2, \cdots, n)$.

泛函 $J(x, y_1, y_2, \dots, y_n) = \int_{t_0}^{t_1} F(x, y_1, y_2, \dots, y_n, \dot{x}, \dot{y}_1, \dot{y}_2, \dots, \dot{y}_n) \mathrm{d}t$ 的横截条件为

$$
F_{\dot{x}} = \sum_{i=1}^{n} F_{\dot{y}_i} \frac{\varphi_{ix}}{\varphi_{iy_i}} \quad (在位于 C_1 的端点上)
\\
F_{\dot{x}} = \sum_{i=1}^{n} F_{\dot{y}_i} \frac{\psi_{ix}}{\psi_{iy_i}} \quad (在位于 C_2 的端点上)
$$

## 第7章 变分原理

### 7.1 集合与映射

**上确界** $\sup X$：数集 $X$ 的最小上界.

**下确界** $\inf X$：数集 $X$ 的最大下界.

**等势**：$X$ 和 $Y$ 之间存在一个双射 $f$，$X \sim Y$.

**可数集**：一个集合与自然数集等势.

**数域**：$P$ 是某些复数的集合，$P$ 中任两数进行 $+,-,\times,\div$（除数不为零），其结果仍是 $P$ 中的数.

<img src="./images/surj-inj-bij.jpg" width="400">

### 7.2 集合与空间

**度量空间**：$X \neq \varnothing$，$\rho: X \times X \to \mathbb{R}$，$\forall x, y \in X$，若满足

1. 正定性 & 恒等性：$\rho(x, y) \geq 0$，且 $\rho(x, y) = 0 \Leftrightarrow x = y$
2. 对称性：$\rho(x, y) = \rho(y, x)$
3. 三角不等式：$\rho(x, y) + \rho(x, z) \geq \rho(y, z)$

则 $(X, \rho)$ 称为度量空间，$X$ 引入了拓扑结构，$(X, \rho)$ 为拓扑空间

**线性空间**

1. 加法交换律：$x + y = y + x$
2. 加法结合律：$(x + y) + z = x + (y + z)$
3. 零元素：$x + \theta = x$
4. 逆元：$x + (-x) = \theta$
5. 单位标量：$1 \cdot x = x$
6. 标量结合律：$(\alpha\beta)x = \alpha(\beta x)$
7. 乘法分配律：$(\alpha + \beta)x = \alpha x + \beta x$
8. 标量分配律：$\alpha(x + y) = \alpha x + \alpha y$

**$p$ 方和序列空间**：数域 $M$ 上满足 $\sum_{n=1}^{\infty} |x_n|^p < \infty$ 的数列 $\{x_n\}_{n=1}^{\infty}$ 的全体所组成的集合

$$
l^p = \left\{ x = \{x_n\}_{n=1}^{\infty} \mid x_n \in M, n \in \mathbb{N} \text{ 且 } \sum_{n=1}^{\infty} |x_n|^p < \infty \right\}
$$

**凸集**：$X$ 是数域 $P$ 上的线性空间，$E \subset X$，若 $\forall x, y \in E$，$\lambda \in [0, 1]$，都有 $\lambda x + (1 - \lambda)y \in E$，则 $E$ 称为凸集. 其几何意义是凸集中任意两点的连线仍在集合中.

**赋范线性空间**：$\forall x, y \in X, \alpha \in P$，若满足（范数公理）

1. 正定性：$\|x\| \geq 0$，且 $\|x\| = 0 \Leftrightarrow x = 0$
2. 齐次性：$\|\alpha x\| = |\alpha| \cdot \|x\|$
3. 三角不等式：$\|x + y\| \leq \|x\| + \|y\|$

则 $(X, \|\cdot\|)$ 称为赋范线性空间.

**基本数列/柯西数列：**设 $\{ x_n\}$ 是度量空间 $(X, \rho)$ 中的数列，如果 $\forall \varepsilon >0, \exists N = N(\varepsilon) \in \mathbb{N}$，当 $m \geq N, n \geq N$ 时，恒有 $|x_m - x_n| < \varepsilon$ 成立，则 $\{ x_n\}$ 称为 $(X, \rho)$ 中的柯西数列/基本数列.

**巴拿赫空间**：赋范线性空间 $X$ 中的任何基本数列都是收敛数列，则 $X$ 为巴拿赫空间.

**内积空间**

1. 对称性：$(x, y) = (y, x)$ 或 $(x, y) = \overline{(y, x)}$（复数域）
2. 齐次性：$(\alpha x, y) = \alpha(x, y)$
3. 线性或可加性：$(x + y, z) = (x, z) + (y, z)$
4. 正定性：$(x, x) \geq 0$，且 $(x, x) = 0 \Leftrightarrow x = 0$

**多种积分内积定义**

1. 希尔伯特内积：$(u, v) = \int_D uv \, \mathrm{d}D$
2. 权函数：$(u, v) = \int_D wuv \, \mathrm{d}D$
3. $(u, v) = \int_D (uv + u'v') \, \mathrm{d}D$
4. $(u, v) = \int_D (uv + u'v' + u''v'') \, \mathrm{d}D$
5. 狄利克雷内积：$(u, v) = \int_D \nabla u \cdot \nabla v \, \mathrm{d}D$

**希尔伯特空间**：内积空间 $X$ 中的任何基本数列都是收敛数列，$X$ 为希尔伯特空间.

### 7.3 标准正交系与傅里叶级数

#### 标准正交系

$M$ 是内积空间 $X$ 中的标准正交系，在 $M$ 中任取 $n$ 个向量 $e_1, e_2, \cdots, e_n$，$\alpha_1, \alpha_2, \cdots, \alpha_n$，则

$$
\left\| x - \sum_{i=1}^{n} (x, e_i)e_i \right\|^2 = \|x\|^2 - \sum_{i=1}^{n} |(x, e_i)|^2 \geq 0, \quad \left\| x - \sum_{i=1}^{n} \alpha_i e_i \right\| \geq \left\| x - \sum_{i=1}^{n} (x, e_i)e_i \right\|
$$

**贝塞尔不等式**：$A = \{e_k \mid k \in \mathbb{N}\}$ 是内积空间 $X$ 的一个标准正交系，$N\in \mathbb{N}$，$\forall x \in X$，

$$
\sum_{i=1}^{\infty} |(x, e_i)|^2 \leq \|x\|^2
$$

> $x$ 在各基的分量长度的平方和小于 $x$ 的范数.

#### 傅里叶级数

$A=\{e_k\mid k\in N\}$ 是内积空间 $X$ 的一个标准正交系，$N\in \mathbb{N}$，$\forall x\in X$，数列 $\{(x,e_k)\}$ 称为 $x$ 关于标准正交系 $A$ 的傅里叶系数集，内积 $(x,e_k)$ 称为傅里叶系数，级数

$$
\sum_{k=1}^{\infty}(x,e_k)e_k
$$

称为 $x$ 关于标准正交系 $A$ 的傅里叶级数。

设 $A=\{e_k\mid k\in N\}$ 为希尔伯特空间 $H$ 中的规范正交系，那么：

1. 级数 $\displaystyle\sum_{i=1}^{\infty}\alpha_i e_i$ 收敛的充要条件是级数 $\displaystyle\sum_{i=1}^{\infty}\vert\alpha_i\vert^2$ 收敛；
2. 若 $x=\displaystyle\sum_{i=1}^{\infty}\alpha_i e_i$，则 $\alpha_i=(x,e_i)$，且 $x=\displaystyle\sum_{i=1}^{\infty}(x,e_i)e_i$；
3. 对于任何 $x\in H$，级数 $\displaystyle\sum_{i=1}^{\infty}(x,e_i)e_i$ 收敛.

### 7.4 算子与泛函

**定义** 设 $X,Y$ 为数域 $P$ 上的赋范线性空间，$D \subseteq X$. 若对应法则 $T$ 使每个 $x \in D$ 唯一确定 $y = Tx \in Y$，则称 $T$ 为 $D$ 到 $Y$ 的**算子**（映射）. 称 $D$ 为**定义域** $D(T)$，象集 $\{ y \mid y = Tx, x \in D \}$ 为**值域** $T(D)$.

|     $X$      |     $Y$      | $T$ 的名称               |
| :----------: | :----------: | :----------------------- |
|    数空间    |    数空间    | 函数                     |
| 赋范线性空间 | 赋范线性空间 | 算子                     |
| 赋范线性空间 |    数空间    | 泛函（值域为数集的算子） |
|    数空间    | 赋范线性空间 | 抽象函数                 |

#### 算子方程

设 $u \in D, f \in T(D)$. 等式 $Tu = f$ 称为**算子方程**，其中 $u$ 为未知函数，$f$ 为自由项（源或汇）. 当 $f = 0$ 时称为**齐次方程**；若边界条件 $u = \varphi$ 中 $\varphi = 0$，则称为**齐次边界条件**.

**算子的基本性质**

设 $X,Y$ 为赋范线性空间，$D$ 为 $X$ 的子空间，$T: D \to Y$，$\alpha \in P$：

1. 可加算子（加法算子）：$T(x+y) = Tx + Ty,\quad x,y \in D$
2. 齐次算子：$T(\alpha x) = \alpha Tx,\quad x \in D$
3. 连续算子：$x_n \to x \ (\text{即 } \Vert x_n - x \Vert \to 0) \Rightarrow Tx_n \to Tx \ (\text{即 } \Vert Tx_n - Tx \Vert \to 0)$
4. 有界算子：$\exists M > 0,\ \Vert Tx \Vert \leq M \Vert x \Vert,\quad \forall x \in D$. 其中那个最小正数 $M$ 称为 $T$ 的**范数**，记作 $\Vert T \Vert$，显然 $\Vert Tx \Vert \leq \Vert T \Vert \Vert x \Vert$.
5. 相似算子：$Tx = \alpha x$
   - 当 $\alpha = 1$ 时为单位算子（恒等算子），记作 $I$；
   - 当 $\alpha = 0$ 时为零算子，记作 $\theta$；
6. 线性算子（齐次可加算子）：$\forall x_1, x_2 \in D, \alpha, \beta \in P, T(\alpha x_1 + \beta x_2) = \alpha T x_1 + \beta T x_2$. 若线性算子 $T$ 的值域为数集，则称**线性泛函**，常记作 $f,g$ 等.
7. 连续线性算子
8. 有界线性算子：$\exists K > 0, \Vert Tx \Vert_Y \leq K \Vert x \Vert_X$（这里 $\Vert \cdot \Vert_D,\Vert \cdot \Vert_Y$ 分别为 $D$ 与 $Y$ 中的范数）. 值域为数集时称**有界线性泛函**；
9. 无界线性算子

#### 线性算子的连续性与有界性

一点连续 $\Rightarrow$ 全局连续：设 $T:D \to Y$ 为线性算子. 若 $T$ 在某点 $x_0 \in D$ 连续，则 $T$ 在 $D$ 上处处连续.

连续与有界的充要条件： 线性算子 $T$ 连续当且仅当 $T$ 有界.

**特殊算子**

| 名称             | 条件                                                          |
| :--------------- | :------------------------------------------------------------ |
| 下界算子         | $\exists m > 0,\ \Vert Tx \Vert \geq m \Vert x \Vert$         |
| 可逆算子         | $\exists S: Y \to X,\ ST = I_X,\ TS = I_Y$（记 $S = T^{-1}$） |
| 保范（等距）算子 | $\Vert Tx \Vert_Y = \Vert x \Vert_X$                          |
| 同构映射         | 保范且一一对应，记 $X \cong Y$                                |

#### 算子范数与算子空间

算子范数：$\Vert T \Vert = \sup_{\substack{x \in D \\ x \neq 0}} \frac{\Vert Tx \Vert}{\Vert x \Vert}$

几何意义：$\dfrac{\Vert Tx \Vert}{\Vert x \Vert}$ 为 $T$ 在 $x$ 方向的**伸缩系数**，$\Vert T \Vert$ 是所有方向的上确界.

有界线性算子空间：$X$ 到 $Y$ 的有界线性算子全体记 $\mathcal{B}(X,Y)$，按算子范数构成赋范空间. 当 $X = Y$ 时简记 $\mathcal{B}(X)$.

对偶空间（共轭空间）：$X$ 上连续线性泛函全体记为 $X^*$.

#### Hilbert 空间上的共轭算子

设 $X,Y$ 为 Hilbert 空间，$T \in \mathcal{B}(X,Y)$. 若存在 $T^* \in \mathcal{B}(Y,X)$ 使

$$
(Tx, y) = (x, T^* y) + B(x,y),\quad \forall x \in X,\ y \in Y
$$

则 $T^*$ 为 $T$ 的**共轭算子**（伴随算子）. 若边界项 $B(x,y) \equiv 0$，则 $(Tx,y) = (x,T^*y)$.

有界算子的共轭算子有界，且 $\Vert T^* \Vert = \Vert T \Vert$.

| 类型                   | 定义                                       |
| :--------------------- | :----------------------------------------- |
| 正规算子               | $TT^* = T^*T$                              |
| 酉算子                 | $T^*T = I_X,\ TT^* = I_Y$（常用 $U$ 表示） |
| 自共轭（埃尔米特）算子 | $T = T^*$（要求 $X = Y$）                  |

#### 对称算子与自共轭算子

设 $H$ 为 Hilbert 空间，$X \subseteq H$ 为子空间，$T: X \to H$ 线性.

**对称算子**： $(Tx, y) = (x, Ty)\ (\forall x,y \in X)$.

**对称算子的判定**： $T$ 对称 $\iff (Tx, x) \in \mathbb{R}\ (\forall x \in X)$.

**自共轭的判定**： $T$ 自共轭 $\iff (Tx,x) \in \mathbb{R}$.

**自共轭算子的乘积**：设 $T_1,T_2$ 均为自共轭算子，则 $T_1T_2$ 自共轭 $\iff T_1T_2 = T_2T_1$.

#### 正算子与正定算子

| 名称     | 条件                                                                                             |
| :------- | :----------------------------------------------------------------------------------------------- |
| 正算子   | $(Tx,y) \geq 0$                                                                                  |
| 正定算子 | $(Tx,y) > 0\ (\forall x,y \neq 0)$；等价地，$\exists r > 0$ 使 $(Ty,y) \geq r^2 \Vert y \Vert^2$ |

若 $T$ 正定，则方程 $Ty = f$ 至多只有一个解.

#### 微分算子

微分算子：若算子 $T: F_1 \to F_2$ 的象 $f = Tu$ 在每点 $x$ 处的值由原象 $u$ 及其有限个导数在 $x$ 处的值决定.

**$n$ 阶线性微分算子**：设 $X$ 为 $[a,b]$ 上多项式组成的线性空间，若

$$
Tx(t) = \sum_{k=1}^n \frac{\mathrm{d}^k}{\mathrm{d}t^k} x(t)
$$

**常用例子**

| 案例                                                                                                                                                                            | 算子                                                                                                          | 名称                                   |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ | :------------------------------------- |
| $Ty = y'' + y$                                                                                                                                                                  | $T = \dfrac{\mathrm{d}^2}{\mathrm{d}x^2} + 1$                                                                 | 二阶微分算子                           |
| $Ty = [p(x)y']' + q(x)y$                                                                                                                                                        | $T = \dfrac{\mathrm{d}}{\mathrm{d}x}\!\left[p(x)\dfrac{\mathrm{d}}{\mathrm{d}x}\right] + q(x)$                | 一般二阶微分算子                       |
| $\Delta u = \dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} + \dfrac{\partial^2 u}{\partial z^2}$                                                       | $\Delta = \sum \dfrac{\partial^2}{\partial x_i^2}$                                                            | 三维 Laplace 算子（缺 $z$ 项则为二维） |
| $\nabla \varphi = \dfrac{\partial \varphi}{\partial x}\boldsymbol{i} + \dfrac{\partial \varphi}{\partial y}\boldsymbol{j} + \dfrac{\partial \varphi}{\partial z}\boldsymbol{k}$ | $\nabla = \sum \dfrac{\partial}{\partial x_i}\boldsymbol{e}_i$                                                | Hamilton 算子（梯度/位势算子）         |
| $TF = \displaystyle\sum_{k=0}^n (-1)^k \dfrac{\mathrm{d}^k}{\mathrm{d}x^k} \dfrac{\partial F}{\partial y^{(k)}}$                                                                | $T = \displaystyle\sum_{k=0}^n (-1)^k \dfrac{\mathrm{d}^k}{\mathrm{d}x^k} \dfrac{\partial}{\partial y^{(k)}}$ | 微分算子                               |
| $J = F_{y'} - \dfrac{\mathrm{d}}{\mathrm{d}x}F_{y''} - \dfrac{\mathrm{d}}{\mathrm{d}x}\!\left(F_{y''}\dfrac{\mathrm{d}}{\mathrm{d}x}\right)$                                    | —                                                                                                             | Jacobi 算子                            |

### 7.5 泛函的导数

设有泛函 $J[u]$，如果

$$
\Delta J = J[u + \varepsilon\eta] - J[u] = \delta J + \delta^2 J + \cdots + \delta^n J + \mathrm{o}(\varepsilon^n)
$$

且 $n$ 次变分

$$
\delta^n J = \frac{\varepsilon^n}{n!}[T(u)\eta^{n-1}, \eta]
$$

式中 $\eta$ 是满足齐次边界条件的任意可取函数，则 $J[u]$ 称为在 $u$ 上可导，且 $T(u)$ 称为泛函 $J[u]$ 在 $u$ 上的 $n$ 阶导数，记作 $J^n[u]$.

- 当 $n=1$ 时，也称为泛函 $J[u]$ 在 $u$ 上的梯度，记作 $\mathrm{grad} J[u]$.
- 当 $n=2$ 时，也可记作 $J''[u]$.

下面讨论最简泛函 $J[y] = \int_{x_0}^{x_1} F(x,y,y')\mathrm{d}x$（固定边界 $y(x_0)=y_0,\, y(x_1)=y_1$）的各阶导数.

#### 一阶导数

$$
J'[y] = \mathrm{grad} J[y] = F_y - \frac{\mathrm{d}}{\mathrm{d}x}F_{y'}
$$

#### 二阶导数

$$
J''[y] = T(y) = S - \frac{\mathrm{d}}{\mathrm{d}x}\left(R\frac{\mathrm{d}}{\mathrm{d}x}\right)
$$

其中 $S = F_{yy} - \dfrac{\mathrm{d}}{\mathrm{d}x}F_{yy'}$，$R = F_{y'y'}$

#### 三阶导数

$$
J'''[y] = F_{yyy} + \left(\frac{3}{2}F_{yyy'}\frac{\mathrm{d}}{\mathrm{d}x}\right) - \frac{3}{2}\frac{\mathrm{d}}{\mathrm{d}x}\left(F_{yy'y'}\frac{\mathrm{d}}{\mathrm{d}x}\right) + \frac{1}{2}\frac{\mathrm{d}}{\mathrm{d}x}\left(\frac{\mathrm{d}}{\mathrm{d}x}F_{y'y'y'}\frac{\mathrm{d}}{\mathrm{d}x}\right) - \frac{\mathrm{d}^2}{\mathrm{d}x^2}\left(F_{y'y'y'}\frac{\mathrm{d}}{\mathrm{d}x}\right)
$$

**结论**：最简泛函的二阶及二阶以上导数都是算子.

### 7.6 算子方程的变分原理

#### 对称正定算子的情形

设 $T$ 是对称正定算子，其定义域为 $D$，值域为 $T(D)$，$u \in D$，$f \in T(D)$，若算子方程 $Tu = f$ 存在解 $u = u_0$，则 $u_0$ 所满足的充要条件是泛函

$$
J[u] = (Tu,u) - 2(u,f)
$$

取得极小值. 式中自由项 $f$ 为自变量的已知函数，称为驱动函数或强制函数. 具有对称正定算子的微分方程的边值问题可以转化为等价的变分问题.

#### 对称负定算子的情形

若 $T$ 是对称的、负定的，即 $(Tu,u) \leqslant 0$，则使泛函 $J[u]$ 取得极大值的 $u$ 就是方程 $Tu=f$ 的解.

#### 非对称正定算子的情形

若方程 $Tu=f$ 的 $T$ 不是对称正定算子，也可作内积 $(Tu-f,\delta u)$，若能化成

$$
\int_D (Tu-f)\delta u\mathrm{d}D = \delta\int_D F\mathrm{d}D + \text{边界项} = 0
$$

的形式（$F$ 是 $u$ 和 $D$ 的函数），则有可能找到相对应的泛函

$$
J[u(D)] = \int_D F\mathrm{d}D
$$

边界条件为边界项等于零或边界固定.

### 7.7 与自共轭常微分方程边值问题等价的变分问题

#### 自共轭常微分方程（施图姆-刘维尔型 $2n$ 阶方程）

$$
\sum_{k=0}^n (-1)^k \frac{\mathrm{d}^k}{\mathrm{d}x^k}[p_k(x)y^{(k)}] = f(x)
$$

其中，$p(x) \geqslant 0$，$q(x) \geqslant 0$ 且均为连续函数，$p(x)$ 至多只有有限个零点.

当 $n=1$ 时，为施图姆-刘维尔型 2 阶方程：

$$
-\frac{\mathrm{d}}{\mathrm{d}x}\left[p(x)\frac{\mathrm{d}y}{\mathrm{d}x}\right] + q(x)y = f(x) \tag{7-7-2}
$$

边界条件为

$$
\begin{cases}
\alpha y'(x_0) - \beta y(x_0) = A \\
\gamma y'(x_1) + \sigma y(x_1) = B
\end{cases}
$$

式中 $\alpha, \beta, \gamma, \sigma$ 为非负常数，$A, B$ 为常数，$\alpha^2+\beta^2 \neq 0$，$\gamma^2+\sigma^2 \neq 0$，且当 $\alpha \neq 0,\, \gamma \neq 0$ 时 $\beta^2+\sigma^2 \neq 0$.

**解的充要条件**

设 $y \in C^2[x_0,x_1]$，$y=y(x)$ 是施图姆-刘维尔型 2 阶方程在其边界条件下的解的充要条件为它对应的泛函

$$
J_1[y] = J[y] + J_B = \int_{x_0}^{x_1} (py'^2 + qy^2 - 2yf) \mathrm{d}x + p(x_0) \dfrac{2A y(x_0) + \beta y^2(x_0)}{\alpha} - p(x_1) \dfrac{2B y(x_1) + \sigma y^2(x_1)}{\gamma}
$$

在 $y=y(x)$ 处取得绝对极小值，且边界条件中的常数相关项与泛函中的同名常数相关项相一致.

#### 自共轭微分算子

$$
T = -\frac{\mathrm{d}}{\mathrm{d}x}\left[p(x)\frac{\mathrm{d}}{\mathrm{d}x}\right] + q(x) \tag{7-7-4}
$$

**算子性质**

- 当 $A=B=0$ 时或 $\alpha=\gamma=0$ 时，此自共轭微分算子 $T$ 是对称正定算子.
- 在 $p \geqslant 0,\, q \geqslant 0,\, \alpha,\beta,\gamma,\sigma \geqslant 0$ 的假定下，$(Ty,y) \geqslant 0$，即 $T$ 是正定算子.

**非自共轭方程的化法**

对于非自共轭微分方程 $p_0(x)y'' + p_1(x)y' + p_2(x)y = 0$（其中 $p_0(x) \neq 0,\, p_1(x) \neq p_0'(x)$），可用待定因子 $\mu(x)$ 乘以该方程使其化为自共轭形式，积分因子为

$$
\mu = \frac{1}{p_0}\mathrm{e}^{\int\frac{p_1}{p_0}\mathrm{d}x}
$$

### 7.8 与自共轭偏微分方程边值问题等价的变分问题

#### 一般椭圆型（微分）方程

$$
-\nabla\cdot(p\nabla u) + qu = f\quad (x,y,z \in V) \tag{7-8-1}
$$

式中 $S$ 为空间封闭曲面，$V$ 为 $S$ 所包围的空间开区域，在 $V$ 上 $p=p(x,y,z)>0$，$q=q(x,y,z)\geqslant 0$.

- 当 $p \equiv 1,\, q \equiv 0$ 时，一般椭圆型方程化为泊松方程 $-\Delta u = f$.
- 若 $f=0$ 则为拉普拉斯方程.

**三种边界条件**

| 名称                               | 边界条件                                                                        |
| :--------------------------------- | :------------------------------------------------------------------------------ |
| 椭圆型方程第一边值问题（狄利克雷） | $u\big\vert_S = g$                                                              |
| 椭圆型方程第二边值问题（诺伊曼）   | $\left.\frac{\partial u}{\partial n}\right\vert_S = h$                          |
| 椭圆型方程第三边值问题（罗宾）     | $\left.\left(p\frac{\partial u}{\partial n} + \sigma u\right)\right\vert_S = k$ |

若 $\sigma=0$，第三边值问题退化为第二边值问题.

#### 方程的解与泛函极值的充要条件

椭圆方程

$$
-\nabla\cdot(p\nabla u) + qu = f\quad (x,y \in D) \tag{7-8-6}
$$

有边界 $\Gamma = \Gamma_1 + \Gamma_2$，其边界条件为

$$
u\big|_{\Gamma_1} = g,\quad \left.\left(p\frac{\partial u}{\partial n} + \sigma u\right)\right|_{\Gamma_2} = k
$$

且假设在 $D+\Gamma$ 上 $p>0$，$q \geqslant 0$，$f,\sigma$ 满足适当光滑条件，则方程的解 $u=u_0$ 所满足的充要条件是泛函

$$
J[u] = \int_D[p(\nabla u)^2 + qu^2 - 2uf]\mathrm{d}D + \int_{\Gamma_2}(\sigma u^2 - 2ku)\mathrm{d}\Gamma \tag{7-8-10}
$$

在 $u\big|_{\Gamma_1} = g$ 条件下于 $u=u_0$ 时取极小值.

### 7.9 弗里德里希斯不等式和庞加莱不等式

**利普希茨边界**：设 $G$ 是 $n$ 维欧氏空间中的有界区域，若其边界 $\Gamma$ 充分光滑或分段光滑，则称为利普希茨边界.

**利普希茨条件**：函数 $f(x)$ 在 $[a,b]$ 上满足 $\alpha$ 次利普希茨条件，存在 $M>0,\, \alpha>0$，使得

$$
\vert f(x^*) - f(x) \vert \leqslant M\vert x^* - x \vert^\alpha
$$

当 $\alpha=1$ 时，称满足利普希茨条件. 多变量情形类似. 利普希茨条件是保证微分方程解的存在性和唯一性的重要条件.

#### 弗里德里希斯不等式

**$n$ 重积分的简写**

$$
I = \int\cdots\int_G u(x_1,\cdots,x_n)\mathrm{d}x_1\cdots\mathrm{d}x_n = \int_G u(\boldsymbol{x})\mathrm{d} \boldsymbol{x}
$$

**弗里德里希斯不等式**

设 $G$ 是具有利普希茨边界的区域，$M$ 是函数 $u(\boldsymbol{x})$ 的线性集合，这些函数在 $\bar{G}$ 内具有一阶连续偏导数，那么 $\exist c_1 \geq 0, c_2 \geq 0$，$c_1, c_2$ 依赖于所考虑的区域，与 $M$ 中的函数 $u(\boldsymbol{x})$ 无关，则有下列不等式

$$
\Vert u \Vert^2 = (u,u) = \int_G u^2(\boldsymbol{x})\mathrm{d}\boldsymbol{x} \leqslant c_1\sum_{k=1}^n\int_G\left(\frac{\partial u}{\partial x_k}\right)^2\mathrm{d}\boldsymbol{x} + c_2\int_\Gamma u^2(s)\mathrm{d}s
$$

**弗里德里希斯第一不等式（一维情形）**

$$
\int_a^b u^2\mathrm{d}x \leqslant c_1\int_a^b u'^2\mathrm{d}x + c_2u^2(a)
\\
\int_a^b u^2\mathrm{d}x \leqslant c_1\int_a^b u'^2\mathrm{d}x + c_2u^2(b)
\\
\int_a^b u^2\mathrm{d}x \leqslant c_1\int_a^b u'^2\mathrm{d}x + c_2[u^2(a)+u^2(b)]
$$

**弗里德里希斯第二不等式（二维情形）**

$$
\iint_G u^2(x,y)\mathrm{d}x\mathrm{d}y \leqslant c_1\iint_G(u_x^2+u_y^2)\mathrm{d}x\mathrm{d}y + c_2\oint_\Gamma u^2(s)\mathrm{d}s
$$

#### 庞加莱不等式

在与弗里德里希斯不等式同样的条件下，存在非负常数 $c_3,c_4$ 使得：

$$
\int_G u^2(x)\mathrm{d}x \leqslant c_3\sum_{k=1}^n\int_G\left(\frac{\partial u}{\partial x_k}\right)^2\mathrm{d}x + c_4\left[\int_G u(x)\mathrm{d}x\right]^2
$$

当 $n=2$ 时

$$
\iint_G u^2(x,y)\mathrm{d}x\mathrm{d}y \leqslant c_3\iint_G(u_x^2+u_y^2)\mathrm{d}x\mathrm{d}y + c_4\left[\iint_G u(x,y)\mathrm{d}x\mathrm{d}y\right]^2
$$

当 $n=1$ 时

$$
\int_a^b u^2(x)\mathrm{d}x \leqslant c_3\int_a^b u'^2(x)\mathrm{d}x + c_4\left[\int_a^b u(x)\mathrm{d}x\right]^2
$$

## 8 变分问题的直接方法

> 直接求取泛函的极大值或极小值的解析式比较困难，就应该采用数值方法进行逼近、迭代等方法求解。

### 8.1 极小（极大）化序列

设有一系列满足边界条件且满足一定的连续条件的函数 $\varphi_0(x), \varphi_1(x), \dots, \varphi_n(x), \dots$，用这些函数构成一系列容许函数作为极值函数的各级近似函数

$$
\begin{cases}
u_0(x) = a_{00} \varphi_0(x) \\
u_1(x) = a_{10} \varphi_0(x) + a_{11} \varphi_1(x)\\
u_2(x) = a_{20} \varphi_0(x) + a_{21} \varphi_1(x) + a_{22} \varphi_2(x) \\
\cdots \\
u_n(x) = a_{n0} \varphi_0(x) + a_{n1} \varphi_1(x) + a_{nn} \varphi_n(x) \\
\cdots
\end{cases}
$$

极小化序列（极大化序列）

$$
J_0 \leq J_1 \leq  J_2 \leq \cdots \leq J_n \leq \cdots \leq J \\
J_0 \geq J_1 \geq  J_2 \geq \cdots \geq J_n \geq \cdots \geq J
$$

只有在一定条件下，极小化（极大化）序列才能收敛到泛函的极值函数. 当 $J[y]$ 式一个与正定算子等价的泛函时，在一定的条件下，其每个序列都收敛到机制函数.

$$
\lim_{n \rightarrow \infty} J_n = J \\
\lim_{n \rightarrow \infty} u_n = \underbrace{y(x)}_{极值函数}
$$

### 8.2 欧拉有限差分法

> 离散法，类比于微积分，最终要求解的是极值曲线的离散值

对于固定边界条件下的最简泛函（不涉及高阶导数），将区间 $[x_0, x_1]$ 划分为 $n$ 各小段（每个有限单元的长度可等可不等），则可用一条折线来替代解析解中的可取曲线.

$$
y(x) = \dfrac{x_i - x}{x_i - x_{i-1}} y_{i-1} + \dfrac{x - x_{i-1}}{x_i - x_{i-1}} y_{i} \quad (x_{i-1} \leq x \leq x_i)
$$

将上式代入最简泛函的式子可得 $J[y(x)] = \varphi(y_1, y_2, \dots, y_{n-1})$，用下列方程组解出 $y_1, y_2, \dots, y_{n-1}$

$$
\dfrac{\partial \varphi}{\partial y_i} = 0 \quad (i=1,2,\dots,n-1)
$$

最后令 $n \rightarrow \infty$ 取极限，只要对函数 $F$ 加些限制，便得到变分问题的解.

也可以用泛函的近似值，用积分和代替积分计算更简便

$$
\int_{x_0}^{x_1} F(x,y,y') \mathrm{d}x = \sum_{k=0}^{n-1} \int_{x_0 + k \Delta x}^{x_1+ (k+1) \Delta x}  F \left(x,y,\dfrac{y_{k+1}-y_k}{\Delta x} \right) \mathrm{d}x
$$

### 8.3 里茨法

> 基函数法，类比于泰勒展开，最终要求解的是基函数的系数

取自完备函数系的 $n$ 个基函数 $\varphi_1(x), \varphi_2(x), \dots, \varphi_n(x)$ ，这些基函数线性无关且满足线性泛函 $J[y] = (Ty,y)-2(f,y)$ 的边界条件.

设 $\varphi_{k}(x) \ (k=1,2,\dots,n)$ 的线性组合

$$
y_n= \sum_{k=1}^{n} a_k \varphi_k (x)
$$

代入泛函的式子得到（注意 $Ty$ 是线性的）

$$
J[y_n] = \sum_{i,j=1}^{n} a_i a_j (T \varphi_i, \varphi_j) - 2 \sum_{i=1}^{n} a_i (\varphi_i, f)
$$

令泛函取得极值，用下列的方程组解出 $a_1, a_2, \dots, a_n$

$$
\dfrac{\partial J}{\partial a_1} = \dfrac{\partial J}{\partial a_2} = \cdots = \dfrac{\partial J}{\partial a_n} = 0
$$

从而最后带回到 $y_n= \sum_{k=1}^{n} a_k \varphi_k (x)$，从而得到泛函变分问题的第 $n$ 近似解.

令 $n \rightarrow \infty$，可得到变分问题的精确解 $y = \lim_{n \rightarrow \infty} y_n$.

### 8.4 坎托罗维奇法

> 基函数Plus法，最终要求解的是系数，但这个系数是自变量的函数

泛函

$$
J[u(x_1, x_2, \cdots, x_n)] = \underbrace{\iint \cdots \int_{\Omega}}_{n} F(x_1, x_2, \dots, x_n, u, u_{x_1}, u_{x_2}, \dots, u_{x_n}) \mathrm{d}x_1 \mathrm{d}x_2 \cdots \mathrm{d}x_n
$$

在边界条件 $u(S) = f(S)$，其中边界 $S \in \Omega$.

选取坐标函数系

$$
\varphi_1(x_1, x_2, \dots, x_n), \varphi_2(x_1, x_2, \dots, x_n), \dots, \varphi_m(x_1, x_2, \dots, x_n), \dots
$$

作函数

$$
u_m= \sum_{k=1}^m a_k (x_i) \varphi_k (x_1, x_2, \dots, x_n)
$$

注意这里 $a_k(x_i)$ 是某一自变量的函数.

将 $u_m$ 代入泛函 $J$，使泛函 $J$ 转换为 $a_1(x_i), a_2(x_i), \dots, a_m(x_i)$，者 $m$ 个函数的泛函为 $\bar{J}$，而$a_1(x_i), a_2(x_i), \dots, a_m(x_i)$ 的选取应使泛函 $\bar{J}$ 达到极值，从而得到原泛函 $J$ 的 $m$ 次近似解.

令 $m \rightarrow \infty$，若极限

$$
u(x_1, x_2, \dots, x_n) = \sum_{k=1}^{\infty} a_k(x_i) \varphi_k(x_1, x_2, \dots, x_n)
$$

### 8.5 伽辽金法

> 三个步骤：选取试函数；代入算子方程求出剩余表达式；选取权函数，作剩余表达式与权函数的内积，并令其正交以消除剩余.

设算子方程及其边界条件分别为

$$
Tu-f=0 \quad (u\in V) \\
Bu-g=0 \quad (u\in S)
$$

其中，$u$ 为待求的未知函数，$T$ 和 $B$ 分别为域内 $V$ 上和边界 $S$ 上的算子；$f$ 和 $g$ 分别为定义在域内和边界上不含 $u$ 的已知函数.

设算子方程的近似解为

$$
u_n = \sum_{i=1}^{n} \underbrace{a_i}_{待定系数} \underbrace{\varphi_i(x_1, x_2, \dots, x_m)}_{基函数}
$$

由于近似解代入原算子方程中一般会有残差，就有域内剩余（边界剩余、残差、剩余、余量）

$$
R_v= T u_n - f \\
R_s = B u_n -g
$$

适当地选择域内加权函数 $W_{vi}$ 和边界加权函数 $W_{si}$ ，使得剩余 $R_v$ 和 $R_s$ 与其相应的权函数的乘积在某种意义意义上等于 0，即可令余量与加权的内积满足正交条件

$$
\underbrace{(R_v, W_{vi})_v}_{域内内积} = \int_V R_v W_{vi} \mathrm{d} V = \int_V (Tu_n-f)W_{vi} \mathrm{d} V = 0
\\
\underbrace{(R_s, W_{si})_s}_{边界内积} = \int_V R_s W_{si} \mathrm{d} S = \int_V (Bu_n-g)W_{si} \mathrm{d} S  = 0
$$

然后取决于具体情况有三种方法

|        | 满足条件                   | 怎么做？                                                 |
| ------ | -------------------------- | -------------------------------------------------------- |
| 内部法 | 边界条件 $B u -g = 0$      | 利用 $(R_v, W_{vi})_v$                                   |
| 边界法 | 算子方程 $Tu-f=0$          | 利用 $(R_s, W_{si})_s$                                   |
| 混合法 | 边界条件和算子方程都不满足 | 同时利用 $(R_v, W_{vi})_v$ 和 $(R_s, W_{si})_s$ 消除剩余 |

将近似解代入域内内积中，并选择 $n$ 个权函数 $W_{vi} \ (i=1,2,\dots,n)$，可建立方程组并解出 $a_i \ (i=1,2,\dots,n)$

$$
\int_V (T u_n -f )W_{vi} \mathrm{d}V = 0 \quad (i=1,2,\dots,n)
$$

因为不同的权函数会有不同的近似方法，若取 $W_{vi} = \varphi_i$ 则得到**伽辽金方程组**

$$
\int_V (T u_n -f ) \varphi_{i} \mathrm{d}V = 0 \quad (i=1,2,\dots,n)
$$

由此解出的 $a_i \ (i=1,2,\dots,n)$ 的系数称为伽辽金系数.

> [!WARNING]
>
> 伽辽金法只适用于齐次边界条件. 否则，可通过适当的变量代换换为其次边界条件.

### 8.6 最小二乘法

对伽辽金法中的域内内积式子中，令 $W_{vi} = \dfrac{\partial R_v}{\partial a_i}$，则有最小二乘法

$$
\int_V R_v \dfrac{\partial R_v}{\partial a_i} \mathrm{d}V
= \int_V (T u_n -f )\dfrac{\partial T u_n}{\partial a_i}  \mathrm{d}V
= 0 \quad (i=1,2,\dots,n)
$$

同样利用此方程组解出待定系数 $a_i \ (i=1,2,\dots,n)$

### 8.7 算子方程的特征值和特征函数

本征方程

$$
Tu - \lambda u = 0
$$

**特征函数正交性** 设 $T$ 是希尔伯特空间 $H$ 的自共轭算子，则 $T$ 的所有特征值都是实数，且不同的特征值相应的特征函数相互正交.

**最小特征值** 设 $T$ 是下有界对称算子，若存在 $u_0 \neq 0$，使 $\lambda_m = \dfrac{(Tu_0,u_0)}{\lVert u_0 \rVert^2}$，则 $\lambda_m$ 就是 $T$ 的最小特征值，$u_0$ 就是与 $\lambda_m$ 对应的特征函数.

**找到次一特征值** 设 $\lambda_1 \leqslant \lambda_2 \leqslant \cdots \leqslant \lambda_{n-1}$ 是下有界对称算子 $T$ 的前 $n-1$ 个特征值，$u_1, u_2, \cdots, u_{n-1}$ 是与之对应的标准正交特征函数. 若存在函数 $u_n \neq 0$，使得 $(u_i, u_n) = 0\ (i = 1, 2, \cdots, n-1)$，

$$
\lambda_n = \frac{(Tu_n, u_n)}{\|u_n\|^2}
$$

取得极小值，则 $\lambda_n$ 就是 $\lambda_{n-1}$ 的次一特征值，$u_n$ 就是与 $\lambda_n$ 对应的特征函数.

**本征方程与伽辽金法和里茨法的等价性** 设 $T$ 是线性微分算子，对于方程 $Tu - \lambda u = 0$，若伽辽金法和里茨法采用相同的基函数，则这两种方法等价，可以求得相同的特征值及相应的特征函数.
