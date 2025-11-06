# 人工神经网络之竞争型学习

> Ciallo～(∠・ω< )⌒★ 我是赤川鹤鸣. 这是我的第一篇关于人工智能技术的博客. 内容大多数为本人的思考和学习笔记，希望对你有所帮助. 

现今，以反向传播为主的神经网络，在处理诸如分类任务时，通常由事先已经规定的分类来决定标签的分类数，进而确定网络架构. 

例如，如果我们对 MNIST 数据集进行分类，那么神经网络的训练标签分类数通常为 10，对应着 10 个阿拉伯数字. 我们通常会创造一个神经网络，使其输出神经元的个数为 10，然后利用图片对应的标签对模型进行训练，最终得到一个手写数字的分类器. 

如果我们要对一些现实生活中拍摄的照片进行分类，例如我们拍摄了一批照片，其中所有照片都可以被划分为“小猫”、“小狗”两类，你据此训练了一个神经网络，并且运行得很好，它以较高的准确率分类了小猫和小狗. 然而，某一天，你拍摄了一批“小鹿”的照片，并希望，在不改变原有网络架构的情况下，继续训练原来的网络模型，使其变成可以分类“小猫”、“小狗”、“小鹿”的模型. 

通常，一个已经训练完毕的反向传播网络不会自主地根据新的输入来更新参数，如果此时你只是用小鹿图片进行模型推理，效果一定是奇差无比. 而且，模型并不会因为你给它看过了小鹿图片就自主地学习了这个分类. 而与人类的大脑相比，人类可以对没有见过的图片进行推理和学习，如果你给从来没有见过“小鹿”的学生们展示一张小鹿照片，那么他们很快就可以学习到“小鹿”这一新的分类，并且在之后看到类似的照片甚至是实物时也可以很快地反应出正确的结果. 

除了上文提及的监督学习，现实世界中存在大量无预先标签的样本，如果想让模型也学习到模式特征，通常使用的是无监督学习的方法，您可能已经知道了其中的一些，例如聚类算法（K-means、DB Scan）等. 这里要介绍的竞争学习也是无监督学习的一种. 

竞争学习和近邻学习是大脑高效地利用有限的皮层神经元资源进化的自然选择进化法则的具体表现形式. 类似人类大脑，如果我们的模型也能够根据新的输入自主动态地调整参数，并且能够增加对新的模式的识别能力（在线学习），同时不显著损失之前的分类能力（避免神经网络灾难性遗忘），那么这样的模型无疑是更健壮的. 

本期，我们就来学习人工神经网络中的**竞争型学习**. 

## 1. 竞争学习原理

### 1.1 网络结构

输入层 → 竞争层
本质上是一个线性层. 

### 1.2 前提条件

$$
\sum_{j} w_{ji} = 1, \ 0 \leq w_{ij} \leq 1
$$

$$
x_i \in \left\{0, \ 1\right\}
$$

### 1.3 计算输出

$$
s_{j} = \sum_{i} w_{ij} x_i
$$

### 1.4 竞争方法

WTA (Winner Takes All)：信号输出最大的那个神经元获胜.

$$
a_k = 
\left\{
\begin{array}{l}
1 &  s_{k} > s_{j}, \ \forall j, \ k \neq j \\
0 & 其他 \\
\end{array}
\right.
$$

### 1.5 参数修正方法
$$
\Delta w_{ij} = \alpha \left( \frac{x_i}{m} - w_{ij} \right)
$$

$$
m = \sum_{i} x_i
$$

### 1.6 推论

$$
\sum_{i} \Delta w_{ij} = 0
$$

## 2. 竞争学习网络特征

1. 竞争层中的神经元总是趋向于**响应它所代表的某个特殊的样本模式**，输出神经元则是检测不同模式的检测器. 
2. 网络通过**极小化同一模式类里样本间距离**（Hamming 距离），**极大化不同模式类间的距离**来寻找模式类. 
3. 网络学习有时依赖于**初始的权值**和**输入样本的次序**. 
4. 无法预先得知模式分类个数，仅在学习后确定. 
5. 使用明显不同的新模式进行分类时，可能能力下降或无法分类，因为使用了非推理方式调节权值. 一般作为其他网络的子网络结构使用. 


## 3. 竞争学习网络的实现
接下来我们来进行竞争学习网络的实现

```python
# 导入必要的包
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import ParamsT
from tqdm import tqdm

```

定义用于竞争型学习网络中的线性层

```python
class NormalizedWeightLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        竞争学习线性层
        :param in_features: 输入特征数 
        :param out_features: 输出特征数
        """
        super(NormalizedWeightLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重，使得每个神经元的权重之和为1
        with torch.no_grad():
            # 遵循公式
            # $$
            # \sum_{j} w_{ji} = 1, \ 0 \leq w_{ij} \leq 1
            # $$
            nn.init.uniform_(self.weight, 0, 1)  # 随机初始化权重
            self.weight /= self.weight.sum(axis=0, keepdims=True)  # 归一化权重

    def forward(self, x):
        return F.linear(x, self.weight)
```

定义竞争型学习的网络结构 

```python
class CompetitiveLearningNet(nn.Module):
    def __init__(self, input_features: int, comp_features: int) -> None:
        """
        单层竞争学习线性层网络
        :param input_features: 输入神经元的个数
        :param comp_features: 竞争神经元的个数 注意这里并不是模式分类个数，只是输出神经元的个数
        """
        super().__init__()
        self.input_features = input_features
        self.comp_features = comp_features

        self.input_2_comp = NormalizedWeightLinear(input_features, comp_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈计算
        :param x: x 具有形状 [batch_size, input_features]
        :return:
        """
        assert x.shape[1] == self.input_features
        x: torch.Tensor = self.input_2_comp(x)

        # 竞争
        print(f"输出层\n{x}")
        y = x.argmax(dim=1)

        return y
```

编写竞争型学习的优化器

```python
class CompetitiveLearningOptimizer(optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float,
                 defaults: Dict[str, Any] = None):
        """
        单层竞争学习线性层网络优化器
        :param params: 模型的参数
        :param lr: 学习率
        :param defaults: 将忽略此参数 
        """
        super().__init__(params, defaults if defaults else {})
        self.lr = lr

    def step(self, closure=None):
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                assert closure and callable(closure)
                # 这里的闭包是为了获取输入 x
                x: torch.Tensor = closure()
                assert isinstance(x, torch.Tensor)
                m = x.sum(dim=1, keepdim=True)
                # delta_w = self.lr * (x / m - param.data)
                batches = x.shape[0]
                for b in range(batches):
                    # 小提示：如果直接操作张量有困难，那么先写循环，然后一步一步拆解
                    # 1. 先写三层循环 批大小, 输入维度, 输出维度
                    # for i in range(self.input_features):
                    #     for j in range(self.comp_features):
                    #         delta_w_bij = self.lr * (x[b, i] / m[b] - param.data[i, j])
                    # 2. 改写为两层循环 批大小, 输入维度
                    # for i in range(self.input_features):
                    #     delta_w_bi = self.lr * (x[b, i] / m[b] - param.data[i, :])
                    # 3. 改写为一层循环 批大小
                    delta_w = self.lr * (x[b] / m[b] - param.data)
                    param.data += delta_w

```

开始训练和推理

```python
batch_size = 2
in_features = 10
out_features = 20

net = CompetitiveLearningNet(input_features=in_features, comp_features=out_features)


def gen():
    """
    随机生成样本
    :return: 形状为 [batch_size, in_features] 的张量
    """
    x = torch.rand(size=(batch_size, in_features))
    x = torch.where(x >= 0.5, torch.ones_like(x), torch.zeros_like(x))
    return x

# 开始训练和推理
for epoch in tqdm(range(10)):
    print(f"Epoch {epoch}")
    # 生成样本
    x = gen()
    print(f"输入层\n{x}")
    
    # 更新参数
    optimizer = CompetitiveLearningOptimizer(params=net.parameters(), lr=0.02)
    optimizer.step(closure=lambda: x)
    
    # 输出结果
    y = net(x)
    print(f"结果\n{y}")
```


参考文献：

> [1] [《人工神经网络》第6章 自组织神经网络](https://www.cnblogs.com/dingruijin/p/12956078.html)
> 
> [2] [【猿知识】汉明距离(Hamming Distance)](https://blog.csdn.net/weixin_44231544/article/details/123351969)
> 
> [3] [THE MNIST DATABASE of handwritten digits](https://yann.lecun.com/exdb/mnist)
> 
> [4] [大脑皮层的神经编码理论与类脑计算方法（二）](https://blog.sciencenet.cn/blog-38584-1027905.html)