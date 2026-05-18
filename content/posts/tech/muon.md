---
title: "Muon 优化器详解：另辟蹊径"
date: 2026-05-02
author: "Yijun Long"
tags: ["Optimizer", "LLM", "Muon"]
categories: ["Machine Learning", "LLM", "Optimizer"]
description: "从数学原理出发，解析 Muon 优化器的算法逻辑与工程细节。"
math: true
summary: "从数学原理出发，解析 Muon 优化器的算法逻辑与工程细节。"
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: "202605-muon"
draft: false # 是否为草稿
comments: true
showToc: true # 显示目录
TocOpen: true # 自动展开目录
autonumbering: true # 目录自动编号
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
searchHidden: false # 该页面可以被搜索到
showbreadcrumbs: true #顶部显示当前路径
mermaid: true
cover:
  image: ""
  caption: ""
  alt: ""
  relative: false
---

# Muon 优化器

## 一、数学原理

### 1.1 动机：从向量到矩阵的本质跨越

在传统优化器（SGD、Adam 等）中，模型的参数被展平为一维向量 $\theta \in \mathbb{R}^d$ 进行优化，完全忽略了参数的矩阵结构。然而在实际模型中，大量参数天然以矩阵形式存在——MLP 的权重矩阵 $W \in \mathbb{R}^{n \times m}$、Transformer 的 QKV 投影矩阵等。将矩阵展平为向量进行优化，会丢失矩阵所特有的结构信息。

矩阵与向量的一个本质区别在于：矩阵描述的是输入空间到输出空间的**线性变换**，评估一个矩阵的优劣应该关注"它对输出产生了什么影响"，而非"它的元素数值有多大"。具体而言，矩阵的迹（trace）、奇异值、谱范数等概念都编码了变换的结构特性，这些信息在向量视角下是无法被利用的。Muon 的核心思想正是：将参数视为矩阵，利用矩阵特有的数学结构来指导优化过程。

### 1.2 矩阵范数基础

为了度量矩阵参数"对网络输出有多大影响"，我们需要引入矩阵范数的概念。本节介绍推导 Muon 所必需的范数理论。

#### 1.2.1 向量范数

对于一个向量 $\mathbf{x} \in \mathbb{R}^{d}$，最常用的是 **L2 范数**（欧几里得范数）：

$$
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_d^2}
$$

**RMS 范数**（Root Mean Square）在 L2 范数基础上除以 $\sqrt{d}$：

$$
\|\mathbf{x}\|_{\text{RMS}} = \frac{\|\mathbf{x}\|_2}{\sqrt{d}} = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

RMS 范数去除了维度对范数值的影响。当维度 $d$ 很大时，L2 范数自然也大（分量多了）；RMS 范数让不同维度的向量可以被公平比较。

#### 1.2.2 矩阵的诱导范数

对于一个线性变换 $\mathbf{y} = W\mathbf{x}$，矩阵 $W$ 的**诱导范数**（induced norm）定义为：

$$
\|W\|_{\text{induced}} = \sup_{\mathbf{x} \neq 0} \frac{\|W\mathbf{x}\|}{\|\mathbf{x}\|}
$$

直觉上，诱导范数衡量的是"这个矩阵作为变换器，最多能把输入放大多少倍"。它在所有可能的输入 $\mathbf{x}$ 中，找那个被 $W$ 放大得最多的，那个放大倍数就是诱导范数。

#### 1.2.3 谱范数

当使用 L2 向量范数时，对应的诱导范数称为**谱范数**（spectral norm）：

$$
\|W\|_{sp} = \sup_{\mathbf{x} \neq 0} \frac{\|W\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}(W)
$$

其中 $\sigma_{\max}(W)$ 是矩阵 $W$ 的最大奇异值。谱范数衡量的是"这个矩阵在最'厉害'的方向上能把向量拉长多少倍"。

类似地，基于 RMS 向量范数的诱导范数为：

$$
\|W\|_{\text{RMS}} = \sup_{\mathbf{x} \neq 0} \frac{\|W\mathbf{x}\|_{\text{RMS}}}{\|\mathbf{x}\|_{\text{RMS}}} = \sqrt{\frac{d_x}{d_y}} \cdot \|W\|_{sp}
$$

其中 $d_x$ 和 $d_y$ 分别是输入和输出的维度。

#### 1.2.4 范数相容性

范数相容性（norm compatibility）是矩阵范数的一个基本性质：

$$
\|W\mathbf{x}\| \leq \|W\|_{\text{induced}} \cdot \|\mathbf{x}\|
$$

这意味着：输出的范数不超过矩阵范数与输入范数的乘积。这是一个上界（不等式），但它给了我们一个重要保证——**只要控制住矩阵 $W$ 的范数，就能控制住输出的上限**。这个性质将在下一节中发挥关键作用。

### 1.3 从范数约束出发推导 Muon

有了范数理论基础，我们可以从第一性原理出发推导 Muon 的更新公式。

#### 1.3.1 控制输出变化量

考虑网络的某一层 $\mathbf{y} = W\mathbf{x}$。当权重更新 $\Delta W$ 时，输出的变化量为 $\Delta\mathbf{y} = \Delta W \cdot \mathbf{x}$。由范数相容性：

$$
\|\Delta\mathbf{y}\|_{\text{RMS}} = \|\Delta W \cdot \mathbf{x}\|_{\text{RMS}} \leq \|\Delta W\|_{\text{RMS}} \cdot \|\mathbf{x}\|_{\text{RMS}}
$$

这意味着：**只要控制住 $\Delta W$ 的诱导范数，就能直接控制网络输出的变化量上界**，从而实现稳定的训练过程。梯度下降法（SGD）的更新量 $\Delta W = -\eta G$ 的范数直接取决于梯度的范数 $\|G\|$，当梯度很大时对网络输出的扰动也大；Adam 虽然用二阶矩做了逐元素归一化，但那是元素级别的操作，没有从"变换对输出的影响"这个矩阵视角进行控制。

因此，一个自然的问题是：**能否在限制矩阵范数的前提下，寻找最优的参数更新方向？**

#### 1.3.2 带约束的优化问题

将上述直觉形式化为如下约束优化问题：

$$
\begin{aligned}
&\min_{\Delta W} \quad \langle \nabla_W \mathcal{J}, \Delta W \rangle \\
&\text{s.t.} \quad \|\Delta W\|_{\text{RMS}} \leq \beta
\end{aligned}
$$

其中矩阵内积定义为 $\langle A, B \rangle = \text{Tr}(A^T B)$，即两个矩阵对应元素相乘再求和（与向量点积完全类似）。

- **目标函数** $\langle \nabla_W \mathcal{J}, \Delta W \rangle$：梯度下降的目标——让更新方向与梯度方向尽可能对齐（取负号后即尽可能减小损失）
- **约束条件** $\|\Delta W\|_{\text{RMS}} \leq \beta$：限制更新量对网络输出的最大扰动

这等价于"在**不把网络搞崩**的前提下（范数约束），**尽可能多地减小损失**（目标函数）"。

#### 1.3.3 最优解指向正交化

可以证明，上述约束优化问题的最优解为：

$$
\Delta W \propto -\text{msign}(\nabla_W \mathcal{J})
$$

其中 $\text{msign}(\cdot)$ 是**矩阵符号函数**（matrix sign function），其效果是将矩阵投影到最近的正交矩阵（详见 1.4 节）。由于正交矩阵的谱范数恒为 1，$\|\text{msign}(\nabla_W \mathcal{J})\|_{sp} = 1$，因此这个最优解恰好满足 $\|\Delta W\|_{\text{RMS}} \leq \beta$（取 $\beta = \eta$ 时）。

这个结论揭示了 Muon 与梯度下降的深层联系。与向量化视角的对比更能说明问题：

| | 向量优化（SGD） | 矩阵优化（Muon） |
|---|---|---|
| 步长约束 | $\|\Delta W\|_2 \leq \beta$ | $\|\Delta W\|_{\text{RMS}} \leq \beta$ |
| "单位化"操作 | 除以 L2 范数：$G / \|G\|$ | msign：投影到正交矩阵 |
| 最优解 | $\Delta W \propto -G / \|G\|$ | $\Delta W \propto -\text{msign}(G)$ |
| 度量的是 | 参数元素数值变化 | 对网络输出的实际影响 |

当把参数当作向量看待时，"步长限制"用向量范数度量，"单位化"是除以标量；当把参数当作矩阵看待时，"步长限制"用矩阵诱导范数度量，"单位化"是 msign。这正是"从向量到矩阵的本质跨越"。

### 1.4 矩阵符号函数 msign

上一节推导出 Muon 的最优更新方向涉及矩阵符号函数 $\text{msign}(M)$。本节详细介绍其定义和性质。

#### 1.4.1 SVD 定义

$\text{msign}$ 是标量符号函数 $\text{sign}(x)$ 在矩阵上的推广。其定义基于奇异值分解（SVD）：

$$
\begin{aligned}
U, \Sigma, V^T &= \text{SVD}(M) \\
\text{msign}(M) &= U_{[:,:r]} V_{[:,:r]}^T
\end{aligned}
$$

其中 $U \in \mathbb{R}^{n \times n}$ 和 $V \in \mathbb{R}^{m \times m}$ 为正交矩阵，$\Sigma \in \mathbb{R}^{n \times m}$ 为对角奇异值矩阵，$r$ 为矩阵 $M$ 的秩。msign 的效果是：**丢弃所有奇异值，仅保留奇异向量结构**，将任意矩阵投影到最近的正交矩阵。

#### 1.4.2 等价形式与标量的类比

利用 SVD，可以证明 msign 具有如下等价形式：

$$
\begin{aligned}
\text{msign}(M) &= (MM^T)^{-1/2} M \\
&= M(M^T M)^{-1/2}
\end{aligned}
$$

这个形式与标量情况高度相似——对于实数 $x$，有 $\text{sign}(x) = x(x^2)^{-1/2}$。矩阵版本不过是把 $x^2$ 替换为 $M^T M$（或 $MM^T$），把 $x^{-1/2}$ 替换为矩阵平方根的逆。

当 $M$ 是对角矩阵时，msign 退化为逐元素的 sign 操作；当 $M$ 是向量时，msign 退化为 L2 归一化（$m / \|m\|_2$）。对于满秩方阵，msign 还可以看作最优正交近似：

$$
\text{msign}(M) = \arg\min_{OO^T=I} \|M - O\|_F^2
$$

#### 1.4.3 关键性质：谱范数恒为 1

由于 $\text{msign}(M)$ 是正交矩阵（或正交矩阵的子矩阵），其谱范数恒为 1：

$$
\|\text{msign}(M)\|_{sp} = 1
$$

这一性质是 Muon 能够控制更新量的关键——无论输入矩阵 $M$ 的梯度有多大，msign 输出的"最大放大倍数"始终为 1，更新量对网络输出的扰动完全由学习率 $\eta$ 决定，不会随梯度大小暴涨。

### 1.5 Muon 的完整更新公式

将 1.3 节的理论最优解与 1.4 节的 msign 结合，并引入动量机制和解耦式权重衰减，得到 Muon 的完整更新公式：

$$
\begin{aligned}
G_t &= \nabla_W \mathcal{J}(W_{t-1}) \\
M_t &= \beta M_{t-1} + G_t \\
W_t &= W_{t-1} \cdot (1 - \eta_t \lambda) - \eta_t \cdot \text{msign}(M_t)
\end{aligned}
$$

其中：
- $G_t$：当前步对权重矩阵 $W$ 的梯度
- $M_t$：动量矩阵（梯度的一阶指数移动平均），$\beta$ 为动量系数
- $\eta_t$：学习率
- $\lambda$：权重衰减系数

公式第三行的两个组成部分各司其职：

- **$\text{msign}(M_t)$**：对纯梯度动量做正交化，得到在 RMS 范数约束下的最优下降方向。由于 $\|\text{msign}(M_t)\|_{sp} = 1$，这一项对网络输出的扰动完全由学习率控制
- **$W_{t-1} \cdot (1 - \eta_t \lambda)$**：解耦式权重衰减（decoupled weight decay），直接对参数进行 shrinkage，保证参数范数有界。这是从 1.3 节的理论推导中额外加入的工程手段（详见第二章）

与 Adam 相比，Muon 仅需维护一份动量估计（Adam 需要同时维护一阶矩和二阶矩），在大规模训练中具有显著的内存优势。

### 1.6 自适应学习率特性

Muon 具有与 Adam 类似的自适应学习率特性，这可以从 msign 的数学性质中直接得出。

**常数缩放不变性**：当损失函数乘以常数因子 $c$ 时，动量矩阵 $M$ 也乘以 $c$，但经过 SVD 分解后的 $UV^T$ 不变（奇异值被丢弃），因此 $\text{msign}(M)$ 不变。这意味着损失函数的常数缩放不会改变优化轨迹，与 Adam 对梯度缩放的不变性类似。

**各向同性的更新幅度**：由于 $\text{msign}(M)$ 是对 $M$ 的正交化结果，其谱范数恒为 1，因此更新量在不同方向上的"最大影响"是一致的。这类似于 Adam 通过二阶矩归一化让不同参数的更新幅度趋于一致，但 Muon 是在矩阵结构层面实现这一点的，而非逐元素操作。

### 1.7 不同参数类型的处理

Muon 对不同形状的参数使用不同的"单位化"操作，这与 msign 在不同维度下的退化形式一致：

| 参数类型 | 示例 | 操作 | 直觉 |
|----------|------|------|------|
| 一般矩阵 $n \times m$ | MLP 权重、QKV 投影 | msign（正交化） | 保留变换方向，丢弃缩放 |
| 对角矩阵 | LayerNorm 的 $\gamma$ | 逐元素 sign | 退化为一维情况 |
| 向量 | 偏置、词表 embedding | L2 归一化 | 退化为一维情况 |

值得注意，虽然词表 embedding 在形式上也是矩阵，但由于其使用方式是稀疏的（每次只取部分行），对其进行逐行 L2 归一化比 msign 更合理。

### 1.8 Newton-Schulz 迭代近似

SVD 的计算开销较大（$O(\min(n^2m, nm^2))$），不适合在每一步优化中执行。因此 Muon 使用 Newton-Schulz 迭代来近似计算 msign。

迭代的出发点是 msign 的等价形式 $\text{msign}(M) = M(M^T M)^{-1/2}$。考虑矩阵函数 $X^{-1/2}$ 在 $X = I$ 处的二阶泰勒展开：

$$
X^{-1/2} \approx I - \frac{1}{2}(X - I) + \frac{3}{8}(X - I)^2 = \frac{15}{8} - \frac{5}{4}X + \frac{3}{8}X^2
$$

将 $X = M^T M$ 代入，得到 msign 的二阶近似：

$$
\text{msign}(M) = M(M^T M)^{-1/2} \approx \frac{15}{8}M - \frac{5}{4}M(M^T M) + \frac{3}{8}M(M^T M)^2
$$

由此得到 Newton-Schulz 迭代公式。令 $X_0 = M / \|M\|_{sp}$（先对 $M$ 做谱范数归一化以保证收敛），然后迭代：

$$
X_{t+1} = \frac{15}{8}X_t - \frac{5}{4}X_t(X_t^T X_t) + \frac{3}{8}X_t(X_t^T X_t)^2
$$

每次迭代的计算量为 $O(nm^2)$（假设 $n \geq m$），通常 4~6 次迭代即可收敛到足够精度，远快于直接计算 SVD。值得注意的是，Muon 的官方实现中迭代公式的常数项与上述理论推导略有不同，其目的是加速收敛。

---

## 二、工程细节

### 2.1 权重衰减的必要性

在 1.3 节的理论推导中，最优解 $\Delta W \propto -\text{msign}(\nabla_W \mathcal{J})$ 并不包含权重衰减项。但在实际的大规模训练中，如果不加权重衰减，Muon 会出现以下问题：

- 训练前期收敛很快，但中后期被 Adam 追上
- 参数范数逐渐失控，出现训练崩溃的苗头
- 在更大的模型上问题更加严重

因此，工程上采用了**解耦式权重衰减**（decoupled weight decay），即权重衰减项独立于 msign 正交化操作，直接作用于参数本身。这与 AdamW 的解耦权重衰减思路一致。更新公式展开为：

$$
W_t = W_{t-1} \cdot (1 - \eta_t \lambda) - \eta_t \cdot \text{msign}(M_t)
$$

其中 $M_t = \beta M_{t-1} + \nabla_W \mathcal{J}(W_{t-1})$ 是纯粹的梯度动量，**不包含权重衰减**。权重衰减以 $W_{t-1} \cdot (1 - \eta_t \lambda)$ 的形式单独执行。

这种解耦设计已被所有官方实现确认：Keller Jordan 的原始实现、Moonshot AI 的 Moonlight 框架、以及 PyTorch 官方的 `torch.optim.Muon` 均采用此方式。其代码形式为：

```python
p.mul_(1 - lr * wd)      # 解耦权重衰减
p.add_(update, alpha=-lr) # 正交化更新
```

**权重衰减保证参数范数有界**。对于任意矩阵范数，可以证明：

$$
\|W_t\| = \|(1 - \eta_t \lambda)W_{t-1} - \eta_t \Phi_t\| \leq (1 - \eta_t \lambda)\|W_{t-1}\| + \eta_t \lambda \|\Phi_t / \lambda\| \leq \max(\|W_{t-1}\|, \|\Phi_t / \lambda\|)
$$

其中 $\Phi_t = \text{msign}(M_t)$ 是 Muon 的正交化更新方向。由于 $\|\text{msign}(\cdot)\|_{sp} = 1$，取谱范数时有：

$$
\|W_t\|_{sp} \leq \max(\|W_{t-1}\|_{sp}, 1/\lambda) \leq \cdots \leq \max(\|W_0\|_{sp}, 1/\lambda)
$$

这意味着权重矩阵的谱范数被 $1/\lambda$ 严格上界约束。由于 $\|\mathbf{y}\| = \|W\mathbf{x}\| \leq \|W\|_{sp} \cdot \|\mathbf{x}\|$，模型的输出也被控制住，不会有爆炸的风险。这对 Attention Logits 爆炸等问题尤为关键。

> **注**：将权重衰减写为 $\text{msign}(\nabla_W \mathcal{J} + \lambda W)$（耦合式，将 $\lambda W$ 混入 msign 内部）是一种常见的误解。这种写法不存在于任何官方实现中。耦合式与解耦式的关键区别在于：耦合式中 $\lambda W$ 仅影响正交化方向（msign 输出谱范数始终为 1），不提供额外的参数 shrinkage；而解耦式中权重衰减直接对参数进行缩放，是保证范数有界的直接手段。

### 2.2 Update RMS 对齐：超参数迁移策略

使用新优化器时，一个实际挑战是如何快速找到好的超参数。Muon 至少有两个重要超参数：学习率 $\eta_t$ 和权重衰减系数 $\lambda$。

**核心观察**：Adam 更新量的 RMS（Root Mean Square）基本稳定在 0.2 ~ 0.4 之间。基于此，可以将 Muon 的更新量 RMS 也对齐到 0.2，从而直接复用 Adam 的超参数。

对于一个矩阵 $W \in \mathbb{R}^{n \times m}$，定义其 RMS 为：

$$
\text{RMS}(W) = \frac{\|W\|_F}{\sqrt{nm}} = \sqrt{\frac{1}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} W_{ij}^2}
$$

注意这里的 RMS 与 1.2 节中定义的 RMS 诱导范数 $\|W\|_{\text{RMS}}$ 不同——前者是矩阵元素平方均值的开方，后者是矩阵作为线性变换的放大倍数上界。

**msign 输出的 RMS 可以解析计算**。由于 $\text{msign}(M) = U_{[:,:r]} V_{[:,:r]}^T$，利用正交矩阵每列的 L2 范数为 1 的性质：

$$
\text{RMS}(\text{msign}(M)) = \sqrt{\frac{1}{nm}\sum_{k=1}^{r}\left(\sum_{i=1}^{n}U_{ik}^2\right)\left(\sum_{j=1}^{m}V_{kj}^2\right)} = \sqrt{\frac{r}{nm}}
$$

在实践中矩阵严格低秩的概率很小，可近似认为 $r \approx \min(n, m)$，因此：

$$
\text{RMS}(\text{msign}(M)) \approx \sqrt{\frac{1}{\max(n, m)}}
$$

为了将 Muon 的更新量 RMS 对齐到 0.2（与 Adam 一致），需要在 msign 输出前乘以缩放因子 $0.2 / \text{RMS}(\text{msign}(M)) = 0.2\sqrt{\max(n, m)}$。

这个结果同时说明了一个重要结论：**Muon 不宜对所有参数矩阵使用同一个学习率**。不同形状的矩阵（如 $4096 \times 4096$ 的 QKV 投影和 $4096 \times 11008$ 的 MLP 权重）具有不同的 $\sqrt{\max(n,m)}$ 因子，如果使用统一学习率，必然导致部分参数更新过快或过慢。

### 2.3 QK-Clip：解决 MaxLogits 爆炸

当模型规模超过 100B 时，即使加入了权重衰减，仍然会出现 **MaxLogits 爆炸**问题：

$$
S_{max} = \max_{i,j} \mathbf{q}_i \cdot \mathbf{k}_j
$$

$S_{max}$ 随训练步数线性甚至超线性增长，长期不收敛。虽然 Softmax 会将 logits 归一化到 $(0, 1)$，但极端大的 MaxLogit 仍可能导致梯度爆炸和训练崩溃。

**已有方案的局限**：

- **softcap**：$O = \text{softmax}(\tau \tanh(QK^T / \tau)) V$，通过 $\tanh$ 截断 logits。但截断前的 logits 仍可能很大，只是把问题转移了，并未根除。
- **QK-Norm**：对 Q、K 分别做 RMSNorm 后再计算注意力。但这需要显式构造完整的 Q、K 矩阵，**与 MLA 不兼容**——MLA 的训练阶段和推理阶段的 Q、K 矩阵是不同的（推理使用吸收模式），无法在推理时复现 QK-Norm。

**QK-Clip 的方案**：

核心思想是用 MaxLogit 本身作为触发信号，自适应地对 Q、K 进行缩放：

1. 计算当前层的 MaxLogit：$S_{max} = \max_{i,j} \mathbf{q}_i \cdot \mathbf{k}_j$
2. 如果 $S_{max} > \tau$（预设阈值），计算缩放比例 $\gamma = \tau / S_{max}$
3. 对 Q 和 K 矩阵分别乘以 $\sqrt{\gamma}$（对称缩放），使新的 MaxLogit 恰好不超过 $\tau$

由于 QK-Clip 直接操作 Q、K 矩阵的参数（而非 attention logits），因此不影响推理阶段的前向计算路径，天然兼容 MLA。QK-Clip 是 **Kimi K2（1000B 参数）** 预训练的关键技术之一。

### 2.4 学习率调度

Muon 的学习率调度与常见的 cosine annealing 策略兼容。在实践中，通常采用如下调度：

- **预热阶段**：从 0 线性增加到峰值学习率，持续前 $T_{warmup}$ 步
- **余弦退火阶段**：从峰值学习率按余弦曲线衰减到最小学习率

需要注意的是，Muon 对学习率的敏感度与 Adam 不同。由于 msign 的输出谱范数恒为 1，学习率直接控制了每一步对网络输出的最大扰动幅度。因此 Muon 的最优学习率通常比 Adam 大（因为 Adam 的有效步长被二阶矩归一化缩小了）。

### 2.5 Newton-Schulz 迭代的实践考量

在实际实现中，Newton-Schulz 迭代需要注意以下细节：

**初始化归一化**：迭代开始前需要对输入矩阵做谱范数归一化 $X_0 = M / \|M\|_{sp}$。实践中常用 Frobenius 范数近似：$X_0 = M / \|M\|_F$（计算更简单），或通过矩阵乘法的幂迭代近似最大奇异值。

**迭代次数**：通常 4~6 次 Newton-Schulz 迭代即可达到足够的近似精度。迭代次数越多，$\text{msign}(M)$ 的近似越精确，但计算开销也相应增加。

**收敛条件**：当 $\|X_t^T X_t - I\|_F < \epsilon$ 时可以提前终止迭代，因为这意味着 $X_t$ 已经足够接近正交矩阵。

**数值稳定性**：在低精度（bf16/fp8）训练中，Newton-Schulz 迭代可能引入数值误差。实现时需要在关键步骤（如 $X_t^T X_t$ 的计算）使用高精度累加。

---

## 三、伪代码

```python
# ============ Muon 优化器 ============

# --- 全局超参数 ---
beta = 0.95          # 动量系数
lr = 1e-2            # 学习率（峰值）
wd = 0.1             # 权重衰减系数
n_schulz_iters = 5   # Newton-Schulz 迭代次数
max_logit_tau = 30.0 # QK-Clip 阈值


def muon_update(param, grad, state, param_type):
    """
    对单个参数执行 Muon 更新。

    参数:
        param: (n, m) 参数矩阵
        grad: (n, m) 梯度矩阵
        state: 优化器状态字典，包含 'momentum'
        param_type: 'matrix' | 'diagonal' | 'vector'

    返回:
        updated_param: 更新后的参数
        updated_state: 更新后的优化器状态
    """
    n, m = param.shape

    # ========== 1. 动量更新 ==========
    if 'momentum' not in state:
        state['momentum'] = zeros_like(param)   # 初始化动量为零矩阵
    momentum = beta * state['momentum'] + grad  # 指数移动平均
    state['momentum'] = momentum

    # ========== 2. 根据参数类型选择单位化操作 ==========
    # 注意：msign/NS 的输入是纯梯度动量，不包含权重衰减
    # 权重衰减在下一步单独、解耦地作用于参数

    if param_type == 'matrix':
        # ---------- 一般矩阵: Newton-Schulz 迭代近似 msign ----------
        M = momentum  # 纯梯度动量，不含权重衰减

        # 谱范数归一化（用 Frobenius 范数近似）
        M_norm = M / frobenius_norm(M)

        # Newton-Schulz 迭代
        X = M_norm.clone()
        for _ in range(n_schulz_iters):
            # 计算 X^T X（在 fp32 精度下以保数值稳定）
            XtX = X.T @ X
            # 迭代更新
            X = (15.0 / 8.0) * X \
                - (5.0 / 4.0) * (X @ XtX) \
                + (3.0 / 8.0) * (X @ (XtX @ XtX))

        # X 现在近似 msign(M)
        update_direction = X

        # Update RMS 对齐: 将更新量 RMS 对齐到 0.2
        # msign 输出的 RMS ≈ sqrt(1 / max(n, m))
        rms_msign = sqrt(1.0 / max(n, m))
        adjusted_lr = lr * (0.2 / rms_msign)  # = lr * 0.2 * sqrt(max(n,m))

    elif param_type == 'diagonal':
        # ---------- 对角矩阵: 逐元素 sign ----------
        update_direction = sign(momentum)  # 纯梯度动量
        # 对角矩阵的 RMS ≈ 1/sqrt(n)，对齐到 0.2
        rms_sign = sqrt(1.0 / n)
        adjusted_lr = lr * (0.2 / rms_sign)

    elif param_type == 'vector':
        # ---------- 向量: L2 归一化 ----------
        M = momentum  # 纯梯度动量
        norm_m = l2_norm(M)
        update_direction = M / max(norm_m, 1e-12)  # 防除零
        # 向量 RMS ≈ 1/sqrt(d)，对齐到 0.2
        rms_l2 = sqrt(1.0 / len(param))
        adjusted_lr = lr * (0.2 / rms_l2)

    # ========== 3. 解耦式权重衰减 + 正交化更新 ==========
    # 步骤 A: 先对参数做权重衰减（解耦，AdamW 风格）
    param = param * (1 - lr * wd)
    # 步骤 B: 再减去正交化后的更新量
    updated_param = param - adjusted_lr * update_direction

    return updated_param, state


def qk_clip(Q, K, tau=30.0):
    """
    QK-Clip: 当 MaxLogit 超过阈值时，对 Q 和 K 进行对称缩放。

    参数:
        Q: (n_heads, seq_len, head_dim) Query 矩阵
        K: (n_heads, seq_len, head_dim) Key 矩阵
        tau: MaxLogit 阈值

    返回:
        clipped_Q, clipped_K: 缩放后的 Q 和 K
    """
    # 计算 QK^T 的最大值（MaxLogit）
    # 实际实现中通常用分块计算以节省显存
    S_max = max_over_heads_and_positions(Q @ K.transpose(-1, -2))

    if S_max > tau:
        # 缩放比例: 使新的 MaxLogit 恰好等于 tau
        gamma = tau / S_max
        # 对 Q 和 K 对称缩放: (sqrt(gamma)*Q) @ (sqrt(gamma)*K)^T = gamma * QK^T
        scale = sqrt(gamma)
        Q = Q * scale
        K = K * scale

    return Q, K


def muon_training_step(model, batch, optimizer_states, lr_schedule, step):
    """
    完整的 Muon 训练步骤。

    参数:
        model: 包含各种参数的模型
        batch: 输入数据
        optimizer_states: 各参数的优化器状态字典
        lr_schedule: 学习率调度函数
        step: 当前训练步数
    """
    # ========== 1. 前向传播 ==========
    hidden = model.forward(batch)

    # ========== 2. QK-Clip（在前向传播的 Attention 计算中）==========
    # 在每个 Attention 层中:
    #   Q, K = compute_qk(hidden)
    #   Q, K = qk_clip(Q, K, tau=max_logit_tau)
    #   attn_weights = softmax(Q @ K.T / sqrt(d)) @ V

    # ========== 3. 计算损失 ==========
    loss = compute_loss(hidden, batch.labels)

    # ========== 4. 反向传播 ==========
    grads = backward(loss, model.parameters())

    # ========== 5. 获取当前学习率 ==========
    current_lr = lr_schedule(step)

    # ========== 6. 逐参数更新 ==========
    for name, param in model.named_parameters():
        grad = grads[name]

        # 判断参数类型
        if is_diagonal_parameter(name):
            # LayerNorm 的 gamma 等对角参数
            param_type = 'diagonal'
        elif is_vector_parameter(name):
            # 偏置、词表 embedding 等向量参数
            param_type = 'vector'
        else:
            # MLP 权重、QKV 投影等一般矩阵参数
            param_type = 'matrix'

        # 执行 Muon 更新
        updated_param, updated_state = muon_update(
            param=param,
            grad=grad,
            state=optimizer_states.get(name, {}),
            param_type=param_type
        )
        # 写回参数和状态
        model.set_parameter(name, updated_param)
        optimizer_states[name] = updated_state

    return loss.item()
```

---

## Citation

- [1] Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon).
- [2] Kenney, C. S. & Laub, A. J. (1992). On Scaling Newton's Method for Polar Decomposition and the Matrix Sign Function. *SIAM Journal on Matrix Analysis and Applications*, 13(3), 688–706.
- [3] Loshchilov, I. & Hutter, F. (2019). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). *ICLR 2019*.
- [4] Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM. DOI: [10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778).
- [5] DeepSeek-AI. (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). *arXiv preprint arXiv:2412.19437*.
- [6] Kimi Team. (2025). [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534). *arXiv preprint arXiv:2507.20534*.
- [7] PyTorch. (2025). [Add Muon optimizer (PR #153048)](https://github.com/pytorch/pytorch/pull/153048). *PyTorch*.

## Citation

如果你在研究或工作中引用了本文，请以以下格式引用：

**BibTeX:**

```bibtex
@misc{long2026rl_llm,
  author       = {Long, Yijun},
  title        = {Muon 优化器详解：另辟蹊径},
  year         = {2026},
  howpublished = {\url{https://procrastinatorrrr.github.io/posts/tech/202605-muon/}},
  note         = {Accessed: 2026-05-02}
}
```

**APA Style:**

```txt
Long, Y. (2026). Muon 优化器详解：另辟蹊径. https://procrastinatorrrr.github.io/posts/tech/202605-muon/
```
