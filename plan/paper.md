以下是 Jacobi 中三个关键优化技术（动态 K、N-gram 投机、Intra 投机）的原始论文及技术要点，按与你的项目相关程度排序。

---

## 0. 基石论文：Lookahead Decoding（ICML 2024）— Jacobi 投机解码的理论基础

你项目中的 Jacobi 解码并非凭空而来，其理论基础来自这篇论文。**理解 Jacobi 的根源才能理解其他三项技术。**

| 项目          | 内容                                                                      |
| ------------- | ------------------------------------------------------------------------- |
| **论文标题**  | Break the Sequential Dependency of LLM Inference Using Lookahead Decoding |
| **作者/机构** | Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang / UC Berkeley              |
| **发表**      | ICML 2024 (PMLR 235:14060-14079)                                          |
| **代码**      | github.com/hao-ai-lab/LookaheadDecoding                                   |
| **论文链接**  | https://proceedings.mlr.press/v235/fu24a.html                             |

**核心思路**：将自回归解码等价地建模为求解非线性方程组，然后用 **Jacobi 迭代法**并行求解——每次迭代中，所有 token 位置同时更新，逐步收敛到正确序列，从而打破逐 token 生成的串行依赖。

**两个核心机制**：

1. **Jacobi 迭代**：并行生成 K 个候选 token，每一轮用当前上下文同时更新所有位置
2. **验证与接受**：用目标模型验证候选，接受匹配的 token 前缀，类似投机解码的 verify 步骤

**结论**：无需辅助模型或外部数据，在 MT-bench 上加速 1.8×，多 GPU 代码补全场景加速 4×。

> ⚠️ **重要**：Lookahead Decoding 是「理论框架」，你项目中的 Jacobi 解码是它在 Mamba 混合模型上的工程实现。以下是基于此框架的三个优化技术。

---

## 1. 动态 K（Dynamic Speculation Length）

### 1a. SpecDec++（2024）

| 项目         | 内容                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| **论文标题** | SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths |
| **作者**     | Kaixuan Huang et al.                                                    |
| **发表**     | arXiv:2405.19715 (2024.05)                                              |
| **论文链接** | https://arxiv.org/abs/2405.19715                                        |

**核心思路**：将候选长度 K 的选择建模为 **马尔可夫决策过程（MDP）**，理论上证明最优策略是**阈值策略**——当任一 token 被拒绝的概率超过阈值时，应立即停止投机。

**实现方式**：在草稿模型上附加一个训练好的「接受预测头」，预测当前候选 token 被接受的条件概率。当概率低于阈值时停止投机，动态调整 K。

**性能**：在 Alpaca 上加速 2.04×，比基准投机解码额外提升 7.2%。

### 1b. DSDE（2025）

| 项目         | 内容                                                                         |
| ------------ | ---------------------------------------------------------------------------- |
| **论文标题** | DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving |
| **作者**     | Mingyu Yang et al.                                                           |
| **发表**     | arXiv:2509.01083 (2025.09)                                                   |
| **论文链接** | https://arxiv.org/abs/2509.01083                                             |

**核心思路**：利用 **KL 散度的方差**作为诊断信号——低方差区域表示模型输出分布稳定（高接受率，适合增大 K），高方差区域表示不稳定（低接受率，应减小 K）。**无需训练**，直接利用模型输出的概率分布作为信号。

**与你项目的关联**：你项目中的动态 K 也是训练无关的，通过 ARL 的 EMA（指数移动平均）来调整 K（ARL < 40%K → 降 K，ARL ≥ 85%K → 升 K），这与 DSDE 基于分布稳定性的思路一致，只是信号来源不同（你用的是实际接受长度，DSDE 用的是 KLD 方差）。

---

## 2. N-Gram 投机（Learning-Free Draft Strategy）

### The N-Grammys（NeurIPS 2024 Workshop ENLSP-IV）

| 项目         | 内容                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------- |
| **论文标题** | The N-Grammys: Accelerating Autoregressive Inference with Learning-Free Batched Speculation |
| **作者**     | Lawrence Stewart, Matthew Trager, Sujan Gonugondla, Stefano Soatto                          |
| **发表**     | arXiv:2411.03786 (2024.11), 发表于 NeurIPS 2024 ENLSP-IV Workshop                           |
| **论文链接** | https://arxiv.org/abs/2411.03786                                                            |

**核心发现**：用 N-gram 模式（从当前上下文中提取的最长匹配后缀 + 接续）作为草稿 token，虽然 N-gram 预测的 top-1 通常不是目标模型的 argmax，但目标模型的真实 token **经常在 N-gram 的 top-k 预测中**（k 较小）。

**两个 N-gram 源**：

1. **模型权重中的 N-gram**：从训练数据统计中获得
2. **上下文中的 N-gram**：从当前输入/输出序列中实时提取（你项目中用的就是这个）

**优势**：

- **零成本草稿**：无需训练、无需额外模型
- **即插即用**：不修改目标模型，无缝集成
- 性能可与更复杂的投机方法相当

**与你项目的关联**：你的 n-gram cache 正是基于上下文 N-gram 的草稿策略，n=4 表示使用 4-gram 模式匹配。论文指出：虽然 N-gram 的 top-1 命中率不高（因此需要较长的 warmup），但 top-k 命中率足以驱动有效的投机解码。

---

## 3. Intra 投机（自投机解码 / Self-Speculative Decoding）

你的 Intra 路径（`forward_with_mamba_intra`）的核心思想来自 **自投机解码**（Self-Speculative Decoding）——不依赖外部草稿模型，而是利用目标模型自身的**内部组件**（层子集、SSM 子图等）作为草稿。

### 3a. 理论基础：CLaSp（2024）

| 项目         | 内容                                                       |
| ------------ | ---------------------------------------------------------- |
| **论文标题** | CLaSp: In-Context Layer Skip for Self-Speculative Decoding |
| **发表**     | ACL (aclanthology.org)                                     |
| **论文链接** | https://aclanthology.org/                                  |

**核心思路**：跳过目标模型的某些层来构建草稿模型——用浅层输出作为深层预测的近似。无需额外训练或参数。

### 3b. 与你项目最相关：Component-Aware Self-Speculative Decoding（2026）

| 项目         | 内容                                                                |
| ------------ | ------------------------------------------------------------------- |
| **论文标题** | Component-Aware Self-Speculative Decoding in Hybrid Language Models |
| **作者**     | Hector Borobia et al.                                               |
| **发表**     | arXiv:2605.01106 (2026.05)                                          |
| **论文链接** | https://arxiv.org/abs/2605.01106                                    |

**核心思路**：首次提出在**混合架构**（Hybrid LM，即 Mamba + Attention）中利用架构异质性做自投机——**隔离 SSM/线性注意力子图作为零成本内部草稿**。

**关键发现**：

- 并行混合架构（Mamba-2 + Attention 同层并行）接受率达 0.68（贪婪解码，K=2）
- 顺序混合架构接受率仅 0.038，**相差 18 倍**
- 架构的组件组合方式决定了自投机的可行性

**与你项目的关联**：你的 Mamba 混合模型正是「并行混合架构」（Mamba + Transformer 同层），你的 Intra 路径通过 `forward_with_mamba_intra` 在 Mamba 隐藏状态边界提取内部状态作为草稿，本质上就是 component-aware self-speculation 的一种实现。论文为你的方法提供了理论支撑：并行混合架构天然适合自投机。

### 3c. 其他自投机相关工作

- **FastVLM**：模仿学习 + 轻量草稿 + 完整模型验证，用于视觉语言模型
- **QuantSpec**：4-bit 量化 KV cache + 量化权重构建草稿
- **Draft on the Fly**：基于余弦相似度动态选择草稿层，无需训练

---

## 技术总结表

| 技术                                    | 核心思想                                         | 代表论文                        | 与你的关联                      |
| --------------------------------------- | ------------------------------------------------ | ------------------------------- | ------------------------------- |
| **Lookahead Decoding**（Jacobi 理论根） | Jacobi 迭代并行解非线性方程组，打破串行解码依赖  | Fu et al., ICML 2024            | 你项目的理论基础                |
| **动态 K**                              | 根据接受率/分布稳定性自适应调整每轮候选 token 数 | SpecDec++ (2024), DSDE (2025)   | ARL EMA 阈值调度                |
| **N-Gram 投机**                         | 从上下文中提取 N-gram 模式作为零成本草稿         | The N-Grammys, NeurIPS 2024     | ngram-n=4 缓存预测              |
| **Intra 投机**                          | 利用混合模型内部组件（SSM 子图）作为零成本草稿   | Component-Aware SSD, arXiv 2026 | Mamba 中间状态提取 + K=1 校正步 |

这四项技术组合形成了一条完整链：**Jacobi 并行迭代（框架） → 动态 K 调整窗口大小 → N-gram 提供外部草稿 → Intra 提供内部草稿**，在无需额外模型的情况下最大化接受长度。
