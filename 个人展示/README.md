# 迁移学习大作业：DANN on MNIST → MNIST-M

复现 Ganin & Lempitsky, ICML 2015《Unsupervised Domain Adaptation by Backpropagation》，
并生成**特征分布随训练逐步对齐的 t-SNE 演化 GIF**，用于 12 月 9 日课程汇报。

## 一句话说明它在做什么

同一个模型：**只用源域标签训练** → 目标域准确率崩掉；**加上域分类器 + 梯度反转层** → 目标域准确率回升。
两个数字之差，就是迁移学习的增益。论文报告的数值是 **52.25% → 76.66%**。

## 快速开始

```bash
# 1. 冒烟测试（约 1 分钟，确认环境通不通）
python src/dann_mnistm.py --smoke

# 2. 正式跑：先 source-only 再 DANN，两个实验都做（约 50-60 分钟）
python src/dann_mnistm.py --epochs 50

# 3. 只跑其中一个
python src/dann_mnistm.py --mode source --epochs 50
python src/dann_mnistm.py --mode dann   --epochs 50

# 4. 中断后不用重跑：特征已逐 epoch 存盘在 outputs/feat，直接重新出图
python src/dann_mnistm.py --render-only --mode both
```

在 PyCharm 里运行时，**解释器必须指向 `.venv2`**（Python 3.11.9 + torch 2.9.0+xpu）。
不要用 `.venv`（那是 CPU 版 torch，会白跑好几倍时间）。

## 中断是安全的

每完成一个 epoch 就做两件事：把 `metrics_*.csv` 追加一行，把该 epoch 的可视化特征
（固定 500+500 个样本的 768 维向量）存成 `outputs/feat/feat_{mode}_ep{NNN}.npy`。

所以中途被杀不会丢进度——重跑太贵，直接 `--render-only` 就能用已存盘的特征出 GIF 和曲线。
t-SNE 必须拿到全部帧才能一次性降维，因此不能等训练结束才在内存里攒帧。

**注意**：不带 `--resume` 的新一轮训练会自动**清除**该 mode 的旧特征文件
（防止上次训练的帧混进本次 GIF）；想保留旧特征续跑请加 `--resume`。

## 单次执行有 10 分钟上限时：分段跑

每个 epoch 结束还会写一份断点 `outputs/ckpt_{mode}.pt`（模型、优化器、RNG、epoch、指标）。
配合 `--resume` 可以把 50 个 epoch 拆成任意段，每段单独执行：

```bash
# 第一段：跑到 10
python src/dann_mnistm.py --mode dann --epochs 10 --no-render
# 之后每段在上一次基础上继续，直到 50
python src/dann_mnistm.py --mode dann --epochs 20 --resume --no-render
python src/dann_mnistm.py --mode dann --epochs 30 --resume --no-render
python src/dann_mnistm.py --mode dann --epochs 40 --resume --no-render
# 最后一段：跑完并渲染
python src/dann_mnistm.py --mode dann --epochs 50 --resume
```

`--epochs` 始终是**总目标轮数**，不是"再跑几轮"。中间段加 `--no-render` 跳过渲染，
最后一段去掉它，一次性出 GIF 和曲线。断点里存了 RNG 状态，续跑的调度与不中断基本一致。

**分段之间不能改参数**：断点里存了超参指纹（`--bs` / `--train-limit` / `--gamma` /
`--fixed-lambda` / `--lr`），`--resume` 时检测到不一致会直接拒绝——改了参数续跑会让
调度进度错位，静默产出错误的 λ/lr 曲线。确要换参数重跑，去掉 `--resume`（旧特征会被清掉）。

## 产出物（全部在 `outputs/`）

| 文件 | 内容 | 汇报用途 |
|---|---|---|
| `dann_tsne_evolution.gif` | DANN 的特征分布演化，源域蓝 / 目标域红 | **主图**，建议放 1 页专门讲 |
| `source_tsne_evolution.gif` | 同样的图但没有迁移 | 对照页："不做迁移时两域始终分离" |
| `dann_curves.png` | 源域 / 目标域 / 域判别器三条准确率曲线 | 讲解"域混淆"的证据 |
| `source_curves.png` | 基线曲线 | 对照 |
| `compare.png` | 两个实验的目标域准确率柱状图 | 结论页 |
| `metrics_source.csv` / `metrics_dann.csv` | 逐 epoch 指标 | 取数、复现 |
| `summary.json` | 完整结果与超参 | 存档 |

## 关于背景数据（重要，会影响你怎么讲）

论文的 MNIST-M 是把 MNIST 数字叠加到 **BSDS500** 真实照片补丁上，按
`I_out = |I_digit - I_background|` 逐通道取绝对差。本脚本的行为是：

1. 若 `--bsds-dir` 指向的目录里有 `.jpg` → 用真实照片补丁；
2. 否则**自动降级为程序化生成的彩色纹理背景**，并在日志里打印 `背景来源 = synthetic`。

```bash
# 用 BSDS500（推荐，与论文完全一致）
# 先下载解压：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
python src/dann_mnistm.py --epochs 50 --bsds-dir D:/data/BSR/BSDS500/data/images
```

**如果你最后跑出来 `背景来源 = synthetic`，请在 PPT 里明确标注**：
"因下载受限，目标域背景采用程序化生成的彩色纹理替代 BSDS500，域间隙比原设定更大"。
合成背景下源域→目标域的落差会比论文更大，这是正常的，不丢分；但不说明就会被问住。

## 关键实现约定（改动前先读）

这些点写错了不会报错，但结果会静悄悄地错：

| # | 约定 | 写错的后果 |
|---|---|---|
| 1 | 卷积通道顺序 **32 → 48** | 写成 48 → 32 时特征维度变 512，与分类器的 768 不匹配，直接报错 |
| 2 | `lambda` **只在 GRL 的 backward 里乘一次** | 若在 loss 里再乘一次，等效 `lambda²`，训练发散 |
| 3 | 源域 MNIST 要复制成 **3 通道** | 单通道进不了 3 通道输入的卷积层 |
| 4 | 目标域标签**只用于评估** | 混进 loss 就不是无监督域自适应了，答辩必被问 |
| 5 | 所有准确率都在**固定测试子集**上算 | 在训练 batch 上算域判别器准确率，它会一直很高，看不出域混淆 |
| 6 | t-SNE **全局拼接后只降一次** | 逐帧独立 t-SNE 会让坐标轴每帧都变，动画像在抽搐 |
| 7 | 数据**预计算成张量**（默认） | 惰性管线实测单步 0.24s，预计算后 0.035s，50 epoch 差一个多小时 |

## 时间与算力

实测（Intel Arc 核显，XPU，全量 60k 样本）：

| 项目 | 数值 |
|---|---|
| 单步耗时（预计算后） | 约 0.035 s |
| 单 epoch（938 步） | 约 30 s |
| 50 epoch，一个方法 | 约 25 分钟 |
| 两个方法都跑 | 约 50-60 分钟 |

想更快：`--bs 256`（步数减半，但单步变慢，实测没有净收益，所以默认保持论文的 128）。

## 想报告"平均±标准差"

```bash
python src/dann_mnistm.py --mode both --epochs 50 --seed 42 --out-dir outputs/s42
python src/dann_mnistm.py --mode both --epochs 50 --seed 43 --out-dir outputs/s43
python src/dann_mnistm.py --mode both --epochs 50 --seed 44 --out-dir outputs/s44
```

三个 `summary.json` 里取 `results[*].final.acc_target` 求均值与标准差即可。
每次约 1 小时，建议放在晚上跑。

## 常用参数

```
--mode {both,dann,source}   默认 both
--epochs N                 默认 50（论文未指定，50 是复现文献常用值）
--bs N                     默认 128（论文值）
--lr F                     默认 0.01（论文 mu_0）
--vis-every N              默认 2，每 N 个 epoch 采一帧；50 epoch 得 25 帧
--vis-per-domain N         默认 500，每帧每域样本数
--train-limit N            只用前 N 个样本，调试用
--bg-mode {auto,bsds,synthetic}   背景来源
--bsds-dir DIR             BSDS500 解压目录
--seed N                   默认 42
--gamma F                  λ 调度陡度，默认 10（论文值）；消融用 2
--fixed-lambda             消融：λ 全程固定为 1，不用调度
--make-samples             只生成源/目标样本网格图，不训练
--num-workers N            Windows 上建议 0
--render-only              不训练，只用 outputs/feat 下已存盘的特征重新出图
--resume                   从 outputs/ckpt_*.pt 续跑（分段跑用）
--no-render                本次不渲染 GIF（分段跑的中间段用）
--no-precompute            惰性数据管线（省内存，慢很多）
--smoke                    2 epoch + 4000 样本的快速验证
```

## 最终结果（2026-11-03，synthetic 背景，seed=42，各 50 epoch）

| 设定 | 源域 | 目标域 | 域判别器 |
|---|---|---|---|
| source-only（不迁移） | 99.32% | **35.46%** | n/a |
| DANN，γ=2（调度过慢） | 98.27% | 83.30% | 64.8% |
| DANN，γ=10（论文调度） | 98.31% | **92.95%** | 61.4% |
| DANN，λ 固定为 1 | 98.34% | 95.05% | 56.7% |

**迁移增益 +57.5 个百分点**（35.46% → 92.95%）。三条证据：目标域大幅回升；
域判别器从 99.4% 降到 61.4% 逼近随机；源域几乎不掉。

**多 seed 复测（2026-11-23）**：γ=10 三 seed 全收敛——92.95 / 95.98 / 91.85，
均值 **93.59% ± 2.14%**（极差 4.13 点）；固定 λ=1 三 seed 中两例正常（95.05 / 93.49%），
其中 3/5 在训练最初期（epoch 1~2）塌缩为常数预测（11.33%~11.35%，锁死在 ln10/ln2）。
因此原"固定 λ=1 最好"结论**已撤回**：γ=10 与固定 λ 的
2 点差异完全落在种子噪声内，而渐进调度在稳定性上的价值在本设定下得到支持
（终态权重诊断：崩溃 seed 判别器输出层权重达 32855、正常约 1.7~3，bias 标准差全为 0；支持梯度爆炸假说，但为终态快照、因果顺序为推断，3/5 只在本设定观察、不外推）。
CORAL 基线（同骨干/同数据/50 epoch）：目标域 **53.00%**（`outputs_coral/`）。
消融图：`outputs/ablation_curves.png`。

历史：基线最初跑到 38/50 被中断（当时目标域 36.09%），后补跑完整 50 epoch 收敛于
35.46%——**PPT 与逐字稿正文用 seed42 的 35.46 / 92.95 这套代表值**，多 seed 统计（93.59±2.14）作为稳定性证据，勿混用旧值。

## 文件说明

```
src/dann_mnistm.py   唯一源码文件，单文件可直接运行
data/MNIST/raw/      MNIST 原始分片（torchvision 自动下载）
outputs/             全部产出（已 gitignore）
```
