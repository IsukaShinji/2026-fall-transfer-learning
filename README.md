# 2026-fall-transfer-learning

> 2026 秋季《迁移学习》课程仓库：课件笔记 + 期末大作业「DANN 无监督域自适应复现与消融」（MNIST → MNIST-M）。
> English: Course materials & final project for *Transfer Learning* (Fall 2026) — reproducing **DANN** (Domain-Adversarial Neural Network) on MNIST → MNIST-M.

**汇报人 / Author**：黄浩博　**汇报日期**：2026-12-09

## 用的什么模型：DANN（Domain-Adversarial Neural Network）

复现 Ganin & Lempitsky（ICML 2015）的 DANN，在 MNIST → MNIST-M 上做**无监督域自适应**（目标域零标注）。

**架构（三组件 + 一个特殊层）**：

| 组件 | 结构 | 作用 |
|---|---|---|
| 特征提取器 G_f | Conv5×5(32) → Conv5×5(48) → FC 768 | 提取域不变特征 |
| 标签预测器 G_y | FC 100 → 100 → 10 | 源域标签分类 |
| 域判别器 G_d | FC 100 → 2 | 判别样本来自源域/目标域 |
| **梯度反转层 GRL** | 前向恒等，反向梯度 × (−λ) | 把域判别损失变成特征对齐的推力 |

**训练设定**（全部可在 `src/dann_mnistm.py` 参数里改）：

- 优化器：SGD（momentum=0.9），lr=0.01，batch=128（源/目标各 64），50 epoch，seed=42，Intel Arc XPU
- λ 调度：λ_p = 2/(1+e^(−10·p)) − 1（p 为训练进度）；学习率 μ_p = μ_0/(1+10p)^0.75
- 数据：源域 MNIST（黑底白字）→ 目标域 MNIST-M（**本实验自建版本：程序化纹理背景**，非论文的 BSDS500 版）

**核心结果**（synthetic 背景，同 seed）：

| 设定 | 目标域准确率 |
|---|---|
| source-only（不迁移） | 35.46% |
| DANN，γ=2（调度过慢） | 83.30% |
| DANN，γ=10（论文调度） | **92.95%** |
| DANN，λ 固定为 1 | 95.05% |

迁移增益 **+57.5 个百分点**；源域 99.32% → 98.31% 几乎不掉；域判别器 99.4% → 61.4% 逼近随机——两条证据共同证明迁移真实发生。消融结论（固定 λ 反而最好）与论文直觉相反，归因分析见笔记第 10 章。

## 目录结构

```
├── 课件/                  # 全部课程材料：课件 PDF/PPTX（不入库）+ 迁移学习笔记.md（入库）
│   ├── DANN原论文_Ganin2015.pdf · 期末要求.txt
├── 个人展示/              # 期末大作业
│   ├── src/               # dann_mnistm.py（训练）、plot_ablation.py（消融图）、env_check.py（环境自检）
│   ├── outputs/           # GIF 动图、曲线、CSV 指标、summary.json
│   ├── 汇报/              # 汇报 PPT（12 页）+ 讲稿.md + 逐字稿.md
│   └── README.md          # 项目说明与复现命令
```

## 复现

```bash
pip install -r 个人展示/requirements.txt -c 个人展示/constraints.txt
python 个人展示/src/dann_mnistm.py --mode dann --epochs 50
```

## 关于课件

`课件/` 下的 PDF/PPTX 为任课教师材料，受版权与体积（单个最大 152MB）限制**不入库**；自撰的《迁移学习笔记.md》（覆盖全部课件，含勘误与自测题）已入库，是本仓库最有价值的文件之一。

DANN 原论文（Ganin & Lempitsky, ICML 2015）本地副本在 `课件/DANN原论文_Ganin2015.pdf`，公开电子版可在 arXiv（ID: 1409.7495）获取。
