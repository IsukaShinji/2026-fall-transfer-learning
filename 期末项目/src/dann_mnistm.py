#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DANN: Unsupervised Domain Adaptation by Backpropagation  (MNIST -> MNIST-M)
===========================================================================
复现 Ganin & Lempitsky, ICML 2015，并输出特征分布的 t-SNE 演化 GIF。

本脚本做两件事，并给出二者的对比（这是汇报的核心论据）：
    1) source-only 基线：只用源域标签训练，是"不迁移"的对照
    2) DANN：在 1) 的基础上加域分类器 + 梯度反转层，是"迁移"的干预

依赖（本机 .venv2 已装齐，无需再装）：
    torch torchvision numpy matplotlib scikit-learn imageio tqdm pillow
    注意：必须用 Intel XPU 版本的 torch（torch 2.9.0+xpu），不要装 cuda 版。

运行（在 PyCharm 里把解释器指向 .venv2 后直接 Run，或用命令行）：
    # 冒烟测试：约 1 分钟，先确认环境通不通
    python src/dann_mnistm.py --smoke

    # 正式跑：两个实验都做（先 source-only 再 DANN）
    python src/dann_mnistm.py --epochs 50

    # 只跑 DANN
    python src/dann_mnistm.py --mode dann --epochs 50

产出（全部在 outputs/ 下）：
    dann_tsne_evolution.gif       DANN 的特征分布演化（汇报主图）
    source_tsne_evolution.gif     source-only 的对照（"没有迁移时会怎样"）
    dann_curves.png / source_curves.png   三条曲线：源/目标/域判别器准确率
    compare.png                   两个实验的目标域准确率对比柱状图
    metrics_*.csv / summary.json  逐 epoch 指标与最终结果

关键实现约定（易错点，逐条在代码里也有注释）：
    * GRL 的 lambda 只在 backward 里乘一次，loss 里绝不再乘（否则等效 lambda^2）
    * 卷积通道顺序 32 -> 48（写成 48 -> 32 会让特征维度变成 512，与分类器 768 不匹配）
    * 源域 MNIST 必须复制成 3 通道，才能进同一个特征提取器
    * 目标域标签只用于评估，绝不参与任何 loss
    * 所有准确率（含域判别器）都在固定的测试子集上算，不在训练 batch 上算
    * t-SNE 先全局拼接再一次性降维，按 epoch 切分，避免逐帧坐标乱跳
    * 数据预计算成张量：否则 DataLoader 每步都要做一次 numpy->torch 转换，
      实测会把单步耗时从几毫秒拖到几百毫秒，50 个 epoch 要多跑一个多小时。
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# ----------------------------------------------------------------------------
# 0. 基础设施
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

_NUM_CLASSES = 10
_FEAT_DIM = 48 * 4 * 4  # 28 -> conv5 -> 24 -> pool -> 12 -> conv5 -> 8 -> pool -> 4
_IMG = (3, 28, 28)
_METRIC_KEYS = ["epoch", "loss_y", "loss_d", "acc_source", "acc_target",
                "acc_domain", "lambda_last", "lr_last"]


def setup_console():
    """
    Windows 控制台默认 GBK，打印中文可能抛 UnicodeEncodeError，这里统一成 UTF-8。
    同时打开行缓冲：默认 stdout 重定向到文件时是全缓冲，日志要等几 KB 才落盘，
    训练一小时里你会看不到任何进度。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass


def set_seed(seed: int):
    """固定全部随机源。不固定的后果：每次跑出来的曲线不同，PPT 数字对不上。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str | None = None) -> torch.device:
    """优先 XPU（Intel 核显），其次 CUDA，最后回落到 CPU。绝不硬编码 .cuda()。"""
    if pref:
        return torch.device(pref)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_matplotlib_cjk() -> bool:
    """有中文字体就用中文标签，没有就退回英文，保证图里不出现方框（tofu）。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


# ----------------------------------------------------------------------------
# 1. 梯度反转层（GRL）
# ----------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """
    前向：恒等映射，输入输出完全一致。
    反向：把上游梯度乘以 -lambda。

    实现要点：lambda 必须在 forward 时通过 ctx 存下来，不能在建层时就固化 ——
    因为 lambda 随训练进度变化（见 lambda_schedule），固化会导致调度失效。
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 负号 = 让特征提取器"反向优化"域分类器（即最大化域损失）
        # 这里乘的是唯一的 lambda，loss 里不再乘
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradientReversal.apply(x, lambd)


# ----------------------------------------------------------------------------
# 2. 模型（严格对齐论文 MNIST 架构：32 -> 48 两个 5x5 卷积）
# ----------------------------------------------------------------------------

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),   # 28 -> 24
            nn.ReLU(True),
            nn.MaxPool2d(2),                   # 24 -> 12
            nn.Conv2d(32, 48, kernel_size=5),  # 12 -> 8
            nn.ReLU(True),
            nn.MaxPool2d(2),                   # 8 -> 4
            nn.Flatten(),                      # 48*4*4 = 768
        )
        self.classifier = nn.Sequential(
            nn.Linear(_FEAT_DIM, 100), nn.ReLU(True),
            nn.Linear(100, 100), nn.ReLU(True),
            nn.Linear(100, _NUM_CLASSES),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(_FEAT_DIM, 100), nn.ReLU(True),
            nn.Linear(100, 2),
        )

    def forward(self, x, lambd: float = 0.0):
        feat = self.extractor(x)
        logits_y = self.classifier(feat)
        logits_d = self.discriminator(grad_reverse(feat, lambd))
        return logits_y, logits_d, feat


# ----------------------------------------------------------------------------
# 3. 数据
# ----------------------------------------------------------------------------

class TensorView(Dataset):
    """预计算好的张量数据集：__getitem__ 只做索引，把数据管线从瓶颈里摘出来。"""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data, self.labels = data, labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


class LazyMNISTM(Dataset):
    """--no-precompute 时的惰性版本：每次按 |数字 - 背景| 现算。内存省，但慢一个量级。"""

    def __init__(self, base, backgrounds: np.ndarray):
        self.base, self.bg = base, torch.from_numpy(
            backgrounds.astype(np.float32) / 255.0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]                      # (1,28,28) [0,1]
        out = torch.abs(img.repeat(3, 1, 1) - self.bg[idx % len(self.bg)])
        return (out - 0.5) / 0.5, label


def _synth_backgrounds(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    程序化彩色背景（BSDS500 不可得时的降级方案）。
    做法：4x4 低分辨率随机色块 -> 双三次上采样到 28x28 -> 加轻微噪声，
    近似自然照片的低频色彩结构，且与数字笔画形成明显色差。
    用 torch 的 batched interpolate 一次算完，避免逐张 PIL 造成几十秒等待。
    返回 uint8 (n, 3, 28, 28)。
    """
    low = torch.from_numpy(rng.integers(0, 256, size=(n, 3, 4, 4)).astype(np.float32))
    img = F.interpolate(low, size=(28, 28), mode="bicubic", align_corners=False)
    img = img + torch.from_numpy(rng.normal(0, 12, size=(n, 3, 28, 28)).astype(np.float32))
    return img.clamp(0, 255).to(torch.uint8).numpy()


def _load_bsds_backgrounds(bsds_dir: Path | None, n: int, rng: np.random.Generator):
    """从 BSDS500 的 .jpg 里随机裁 28x28 补丁。找不到图就返回 None 让调用方降级。"""
    if bsds_dir is None or not bsds_dir.exists():
        return None
    files = sorted(bsds_dir.rglob("*.jpg"))
    if len(files) < 10:
        return None
    from PIL import Image

    out = np.empty((n, 3, 28, 28), dtype=np.uint8)
    for i in range(n):
        p = files[int(rng.integers(0, len(files)))]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (64, 64), (128, 128, 128))
        w, h = im.size
        if w < 28 or h < 28:
            im = im.resize((max(w, 28), max(h, 28)))
            w, h = im.size
        x, y = int(rng.integers(0, w - 27)), int(rng.integers(0, h - 27))
        out[i] = np.asarray(im.crop((x, y, x + 28, y + 28)), dtype=np.uint8).transpose(2, 0, 1)
    return out


def _make_target_tensor(raw, bg: np.ndarray, chunk: int = 10000) -> torch.Tensor:
    """
    论文 Sec 4.1 的构造：I_out = |I_digit - I_background|，逐通道取绝对差。
    数字先复制成 3 通道，才能和 3 通道背景逐通道相减。
    分块算，避免同时存在三份 188MB 的张量。
    输出已归一化到 [-1,1]，与源域保持一致。
    """
    bg_t = torch.from_numpy(bg.astype(np.float32) / 255.0)          # (nb,3,28,28)
    nb = bg_t.shape[0]
    n = len(raw)
    out = torch.empty((n, 3, 28, 28), dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        digit = raw.data[s:e].float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        idx = torch.arange(s, e) % nb
        out[s:e] = (torch.abs(digit - bg_t[idx]) - 0.5) / 0.5
    return out


def _make_source_tensor(raw) -> torch.Tensor:
    """源域：灰度数字复制成 3 通道，归一化到 [-1,1]。等价 Grayscale(3) + Normalize。"""
    x = raw.data.float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - 0.5) / 0.5


def build_datasets(data_dir: Path, bg_mode: str, bsds_dir: Path | None,
                   seed: int, precompute: bool):
    """
    需要两套 MNIST：
      * 源域用的：3 通道 + 归一化到 [-1,1]，直接喂网络；
      * 合成 MNIST-M 用的：保持单通道 [0,1]，因为要和 [0,1] 的背景做逐通道绝对差，
        若先归一化再相减，公式就不再是论文的 |I_digit - I_bg| 了。
    """
    from torchvision import datasets, transforms

    data_dir.mkdir(parents=True, exist_ok=True)
    # transform=None：直接拿 raw.data（uint8 张量），由我们自己批量转换
    raw_tr = datasets.MNIST(str(data_dir), train=True, download=True, transform=None)
    raw_te = datasets.MNIST(str(data_dir), train=False, download=True, transform=None)
    y_tr = torch.as_tensor(raw_tr.targets).clone()
    y_te = torch.as_tensor(raw_te.targets).clone()

    # 训练集与测试集用互不相同的随机流生成背景，避免"测试集见过背景"导致准确率虚高
    rng_tr, rng_te = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    bg_tr = bg_te = None
    used = "synthetic"

    if bg_mode in ("auto", "bsds"):
        t0 = time.time()
        bg_tr = _load_bsds_backgrounds(bsds_dir, min(len(raw_tr), 12000), rng_tr)
        bg_te = _load_bsds_backgrounds(bsds_dir, min(len(raw_te), 3000), rng_te)
        if bg_tr is not None and bg_te is not None:
            used = "bsds500"
            print(f"[data] 已用 BSDS500 生成背景，耗时 {time.time() - t0:.1f}s")
        else:
            print("[data] 未找到可用的 BSDS500 图片，降级为程序化纹理背景")

    if bg_tr is None or bg_te is None:
        t0 = time.time()
        # 背景池不必与样本数一一对应（用 idx % len(bg) 取用），
        # 12k / 3k 已足够多样，且能把生成时间压到几秒。
        bg_tr = _synth_backgrounds(min(len(raw_tr), 12000), rng_tr)
        bg_te = _synth_backgrounds(min(len(raw_te), 3000), rng_te)
        print(f"[data] 程序化背景生成完成，耗时 {time.time() - t0:.1f}s")

    t0 = time.time()
    if precompute:
        src_tr = TensorView(_make_source_tensor(raw_tr), y_tr)
        src_te = TensorView(_make_source_tensor(raw_te), y_te)
        tgt_tr = TensorView(_make_target_tensor(raw_tr, bg_tr), y_tr)
        tgt_te = TensorView(_make_target_tensor(raw_te, bg_te), y_te)
        print(f"[data] 张量预计算完成，耗时 {time.time() - t0:.1f}s")
    else:
        tf = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        # 惰性路径下，源域用标准 transform；目标域用 LazyMNISTM
        raw_tr2 = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
        raw_te2 = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)
        src_tr, src_te = raw_tr2, raw_te2
        tgt_tr, tgt_te = LazyMNISTM(raw_tr, bg_tr), LazyMNISTM(raw_te, bg_te)

    return src_tr, tgt_tr, src_te, tgt_te, used


def paired_batches(src_loader, tgt_loader, steps: int, total_steps: int):
    """
    每批 = 一半源域（带标签）+ 一半目标域（不带标签），拼成一个 batch。
    显式拼接而不是依赖 DataLoader 的随机顺序 —— 因为后面要靠切片取源域那半。
    """
    si, ti = iter(src_loader), iter(tgt_loader)
    done = 0
    while done < steps:
        try:
            xs, ys = next(si)
        except StopIteration:
            si = iter(src_loader)
            xs, ys = next(si)
        try:
            xt, _ = next(ti)          # 目标域标签丢弃，绝不进入训练
        except StopIteration:
            ti = iter(tgt_loader)
            xt, _ = next(ti)
        done += 1
        yield xs, ys, xt, done / max(total_steps, 1)


# ----------------------------------------------------------------------------
# 4. 调度（论文 Eq.6 / Eq.7）
# ----------------------------------------------------------------------------

def lambda_schedule(p: float, gamma: float = 10.0) -> float:
    """lambda_p = 2/(1+exp(-gamma*p)) - 1，从 0 平滑升到 1。"""
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def lr_schedule(p: float, mu0: float = 0.01, alpha: float = 10.0, beta: float = 0.75) -> float:
    """mu_p = mu0 / (1 + alpha*p)^beta。"""
    return mu0 / (1.0 + alpha * p) ** beta


# ----------------------------------------------------------------------------
# 5. 评估
# ----------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: DANN, loader, device, domain: bool = False) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits_y, logits_d, _ = model(x, 0.0)
        pred = logits_d.argmax(1) if domain else logits_y.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def domain_accuracy(model: DANN, src_x, tgt_x, device):
    """
    在固定的测试子集上算域判别器准确率 —— 必须用测试样本、不能用训练 batch。
    原因：域判别器自己在被优化，在训练分布上它会一直很高，看不出"域混淆"。
    source-only 模式返回 None：此时域判别器从未被训练，数字没有意义，
    画进曲线反而会误导成"域混淆已经做得很好"。
    """
    model.eval()
    c = t = 0
    for x, d in ((src_x, 0), (tgt_x, 1)):
        logits_d = model.discriminator(model.extractor(x.to(device)))
        c += (logits_d.argmax(1) == d).sum().item()
        t += logits_d.size(0)
    return c / max(t, 1)


@torch.no_grad()
def collect_features(model: DANN, src_x, tgt_x, device) -> np.ndarray:
    """提取固定子集的特征并缓存。必须落到 CPU（t-SNE 走 sklearn，不吃 XPU）。"""
    model.eval()
    feats = [model.extractor(x.to(device)).detach().float().cpu().numpy()
             for x in (src_x, tgt_x)]
    return np.concatenate(feats, axis=0)


# ----------------------------------------------------------------------------
# 6. 训练
# ----------------------------------------------------------------------------

def _rng_state():
    st = dict(py=random.getstate(), np=np.random.get_state(), th=torch.get_rng_state())
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            st["th_xpu"] = torch.xpu.get_rng_state()
        except Exception:
            pass
    return st


def _load_rng(st):
    if not st:
        return
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    torch.set_rng_state(st["th"])
    if st.get("th_xpu") is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.set_rng_state(st["th_xpu"])
        except Exception:
            pass


def train(mode: str, args, device, data, fixed, out_dir: Path):
    """
    mode = "source"：只用源域标签（基线）
    mode = "dann"  ：源域标签 + 域对抗（本文方法，GRL 用 lambda 调度）
    """
    src_tr, tgt_tr, src_te, tgt_te, _ = data
    src_fixed, tgt_fixed, _, _ = fixed

    set_seed(args.seed)  # 两个实验用同一随机流，保证对比公平

    if args.train_limit:
        src_tr = Subset(src_tr, range(min(args.train_limit, len(src_tr))))
        tgt_tr = Subset(tgt_tr, range(min(args.train_limit, len(tgt_tr))))

    bs_half = args.bs // 2
    src_loader = DataLoader(src_tr, batch_size=bs_half, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)
    tgt_loader = DataLoader(tgt_tr, batch_size=bs_half, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)

    steps = min(len(src_loader), len(tgt_loader))
    total_steps = steps * args.epochs
    print(f"[{mode}] train={len(src_tr)} steps/epoch={steps} epochs={args.epochs} device={device}")

    model = DANN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    feat_dir = out_dir / "feat"
    feat_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / f"metrics_{mode}.csv"
    ckpt_path = out_dir / f"ckpt_{mode}.pt"

    # 续跑：模型/优化器/RNG/epoch 全部恢复，特征文件按 epoch 重建索引。
    # 非 resume 的新一轮必须清掉旧特征：否则中途被杀时，finally 渲染会把
    # 上一次训练的帧混进本次 GIF（行数相同时行数过滤兜不住）。
    start_epoch, resumed, global_step = 1, False, 0
    history = []
    frame_paths, frame_epochs = [], []
    if args.resume:
        for p in sorted(feat_dir.glob(f"feat_{mode}_ep*.npy")):
            frame_paths.append(p)
            frame_epochs.append(int(p.stem.split("_ep")[1]))
        if not ckpt_path.exists():
            print(f"[{mode}] 警告：--resume 但找不到 {ckpt_path}，将从 epoch 1 全新开始")
    else:
        stale = sorted(feat_dir.glob(f"feat_{mode}_ep*.npy"))
        if stale:
            for p in stale:
                p.unlink()
            print(f"[{mode}] 已清除旧特征 {len(stale)} 个（新一轮从零开始）")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 分段之间改了 bs / train-limit / gamma 等参数会让 total_steps 与调度错位，
        # 静默产出错误的 λ/lr 曲线——直接拒绝比悄悄跑完更有用。
        fp_now = dict(bs=args.bs, train_limit=args.train_limit,
                      gamma=args.gamma, fixed_lambda=args.fixed_lambda, lr=args.lr)
        fp_old = ck.get("hparams")
        if fp_old is not None and fp_old != fp_now:
            raise SystemExit(f"[{mode}] 断点超参与本次不一致：{fp_old} vs {fp_now}。"
                             f"分段跑必须保持同一组参数；确要重跑请去掉 --resume。")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        history = ck.get("history", [])
        global_step = ck.get("global_step", 0)
        _load_rng(ck.get("rng"))
        resumed = True
        print(f"[{mode}] 从 epoch {ck['epoch']} 续跑，目标 epoch {args.epochs}")
        if start_epoch > args.epochs:
            print(f"[{mode}] 已完成到 epoch {ck['epoch']}，无需继续")
            return dict(mode=mode, elapsed_sec=0.0, interrupted=False,
                        final=history[-1] if history else None, resumed=True)

        # CSV 可能在"已写行、未存 ckpt"的窗口里多出重复 epoch 行，按断点截齐
        if metric_path.exists():
            with open(metric_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            kept = [r for r in rows if int(r["epoch"]) <= ck["epoch"]]
            if len(kept) != len(rows):
                with open(metric_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=_METRIC_KEYS)
                    w.writeheader()
                    w.writerows(kept)
                print(f"[{mode}] CSV 已按断点截齐：{len(rows)} -> {len(kept)} 行")

    mf = open(metric_path, "a" if (resumed and metric_path.exists()) else "w",
              newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=_METRIC_KEYS)
    if not (resumed and metric_path.exists()):
        writer.writeheader()

    t_start = time.time()
    interrupted = False

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            run_y = run_d = 0.0
            nb = 0

            for xs, ys, xt, _ in paired_batches(src_loader, tgt_loader, steps, total_steps):
                p = global_step / max(total_steps - 1, 1)      # 训练进度 0 -> 1
                if mode == "dann":
                    # 消融开关：--fixed-lambda 固定 λ=1（无调度），用于展示
                    # "对抗信号一开始就拉满会把源任务学崩"，呼应课件的负迁移一节
                    lambd = 1.0 if args.fixed_lambda else lambda_schedule(p, args.gamma)
                else:
                    lambd = 0.0
                for g in opt.param_groups:
                    g["lr"] = lr_schedule(p, args.lr)

                xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
                x = torch.cat([xs, xt], dim=0)                 # 前半源域，后半目标域
                d = torch.cat([torch.zeros(xs.size(0), dtype=torch.long),
                               torch.ones(xt.size(0), dtype=torch.long)]).to(device)

                logits_y, logits_d, _ = model(x, lambd)
                l_y = F.cross_entropy(logits_y[: xs.size(0)], ys)   # 只对源域算标签损失
                l_d = F.cross_entropy(logits_d, d)                  # 全部样本算域损失

                # 关键：这里只做加法。lambda 已经在 GRL 的 backward 里乘过，
                # 若写成 l_y + lambd * l_d 就是双重缩放，训练直接崩。
                loss = (l_y + l_d) if mode == "dann" else l_y

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                run_y += l_y.item()
                run_d += l_d.item() if mode == "dann" else 0.0
                nb += 1
                global_step += 1

            # ---- 每个 epoch 结束：评估 + 缓存特征 ----
            acc_s = accuracy(model, DataLoader(src_te, batch_size=512), device)
            acc_t = accuracy(model, DataLoader(tgt_te, batch_size=512), device)
            acc_d = domain_accuracy(model, src_fixed, tgt_fixed, device) \
                if mode == "dann" else None

            row = dict(epoch=epoch,
                       loss_y=run_y / max(nb, 1),
                       loss_d=run_d / max(nb, 1) if mode == "dann" else None,
                       acc_source=acc_s, acc_target=acc_t, acc_domain=acc_d,
                       lambda_last=(1.0 if args.fixed_lambda else
                                    lambda_schedule(global_step / max(total_steps - 1, 1), args.gamma))
                       if mode == "dann" else 0.0,
                       lr_last=lr_schedule(global_step / max(total_steps - 1, 1), args.lr))
            writer.writerow(_fmt_row(row))
            mf.flush()          # 逐 epoch 落盘：中途被杀也留有完整 CSV
            history.append(row)

            if epoch % args.vis_every == 0 or epoch == args.epochs:
                # 特征立即存盘。t-SNE 必须拿到全部帧才能一次性降维，
                # 所以不能等训练结束才攒在内存里——那样子中断一次就全丢了。
                fp = feat_dir / f"feat_{mode}_ep{epoch:03d}.npy"
                np.save(fp, collect_features(model, src_fixed, tgt_fixed, device))
                frame_paths.append(fp)
                frame_epochs.append(epoch)

            # 断点：单次执行有 10 分钟上限时，靠它分段把 50 个 epoch 跑完
            torch.save(dict(epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                            history=history, global_step=global_step, rng=_rng_state(),
                            hparams=dict(bs=args.bs, train_limit=args.train_limit,
                                         gamma=args.gamma, fixed_lambda=args.fixed_lambda,
                                         lr=args.lr)),
                       ckpt_path)

            dom = f"{acc_d:.4f}" if acc_d is not None else "  n/a "
            print(f"[{mode}] ep{epoch:3d}/{args.epochs}  "
                  f"src {acc_s:.4f}  tgt {acc_t:.4f}  dom {dom}  "
                  f"lam {row['lambda_last']:.3f}  lr {row['lr_last']:.5f}")

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[{mode}] 收到中断，用已完成的 {len(frame_epochs)} 帧继续出图")

    finally:
        mf.close()
        # 即使中断，也已存盘的特征足以出图；--no-render 供分段跑时跳过渲染。
        # 渲染失败不能让 50 个 epoch 的训练成果跟着报废，所以单独兜住。
        if frame_paths and not args.no_render:
            try:
                render_from_disk(mode, out_dir, args)
            except Exception as e:
                print(f"[{mode}] 渲染失败（{e}）。训练指标已写入 {metric_path}，"
                      f"特征在 {feat_dir}，可用 --render-only 单独重出")

    elapsed = time.time() - t_start
    return dict(mode=mode, elapsed_sec=round(elapsed, 1), interrupted=interrupted,
                final=history[-1] if history else None, resumed=resumed)


def _fmt_row(r):
    return {k: ("" if r.get(k) is None else
                (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]))
            for k in _METRIC_KEYS}


# ----------------------------------------------------------------------------
# 7. t-SNE 演化 GIF（本项目的视觉主体）
# ----------------------------------------------------------------------------

def render_from_disk(mode, out_dir: Path, args):
    """
    从磁盘上已保存的特征重建 t-SNE 与 GIF。
    单独拆出来是为了：训练中断后无需重跑，直接 `--render-only` 就能出图。
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    feat_dir = out_dir / "feat"
    paths = sorted(feat_dir.glob(f"feat_{mode}_ep*.npy"))
    if not paths:
        print(f"[{mode}] 没有找到已保存的特征，跳过渲染")
        return
    epochs = [int(p.stem.split("_ep")[1]) for p in paths]

    acc_t, acc_d = {}, {}
    mp = out_dir / f"metrics_{mode}.csv"
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                acc_t[int(r["epoch"])] = float(r["acc_target"])
                acc_d[int(r["epoch"])] = float(r["acc_domain"]) if r["acc_domain"] else None

    feats = [np.load(p) for p in paths]
    # 行数必须一致（同一评估子集）。若混入了旧冒烟测试等留下的 200 行文件，
    # np.split 会在这里抛 ValueError，且一帧都出不来 —— 统一按多数行数过滤。
    counts = [f.shape[0] for f in feats]
    major = max(set(counts), key=counts.count)
    if len(set(counts)) > 1:
        keep = [(p, f) for p, f in zip(paths, feats) if f.shape[0] == major]
        print(f"[{mode}] 警告：发现 {len(paths) - len(keep)} 个行数不一致的特征文件"
              f"（{sorted(set(counts) - {major})} 行 vs 主流 {major} 行），已剔除")
        paths = [p for p, _ in keep]
        feats = [f for _, f in keep]
        if len(feats) < 2:
            print(f"[{mode}] 剔除后不足 2 帧，无法渲染")
            return
    epochs = [int(p.stem.split("_ep")[1]) for p in paths]
    n_per = feats[0].shape[0] // 2
    big = np.concatenate(feats, axis=0)
    print(f"[{mode}] t-SNE 输入 {big.shape}（{len(paths)} 帧），先 PCA 降维")

    # PCA 预降维：sklearn 官方推荐做法，几万点能快一个数量级，分布形态几乎不变
    n_comp = min(50, big.shape[0] - 1, big.shape[1])
    if big.shape[0] > n_comp:
        big = PCA(n_components=n_comp, random_state=42).fit_transform(big)

    perplexity = min(30, max(5, (big.shape[0] - 1) // 3))
    emb = TSNE(n_components=2, init="pca", learning_rate="auto",
               perplexity=perplexity, random_state=42).fit_transform(big)

    coords = np.split(emb, len(feats), axis=0)
    cjk = setup_matplotlib_cjk()
    lab_src = "\u6e90\u57df" if cjk else "Source"
    lab_tgt = "\u76ee\u6807\u57df" if cjk else "Target"

    frame_dir = out_dir / "frames" / mode
    frame_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    png_paths = []
    for i, (xy, ep) in enumerate(zip(coords, epochs)):
        ax.clear()
        ax.set_facecolor("white")
        ax.scatter(xy[n_per:, 0], xy[n_per:, 1], s=22, c="#d62728",
                   alpha=0.6, linewidths=0, label=lab_tgt)
        ax.scatter(xy[:n_per, 0], xy[:n_per, 1], s=22, c="#1f77b4",
                   alpha=0.6, linewidths=0, label=lab_src)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=13)
        ax.set_title("DANN \u7279\u5f81\u5206\u5e03\u6f14\u5316 (MNIST \u2192 MNIST-M)"
                     if mode == "dann" else
                     "source-only \u7279\u5f81\u5206\u5e03\u6f14\u5316 (MNIST \u2192 MNIST-M)",
                     fontsize=11)
        dv = acc_d.get(ep)
        ax.text(0.02, 0.97,
                f"Epoch {ep}\n"
                f"\u76ee\u6807\u57df\u51c6\u786e\u7387: {acc_t.get(ep, float('nan')) * 100:.1f}%\n"
                + (f"\u57df\u5224\u522b\u5668\u51c6\u786e\u7387: {dv * 100:.1f}%" if dv is not None
                   else "\u57df\u522b\u5668\u51c6\u786e\u7387: n/a\uff08\u672a\u8bad\u7ec3\uff09"),
                transform=ax.transAxes, va="top", ha="left", fontsize=13)
        pth = frame_dir / f"frame_{i:03d}.png"
        fig.savefig(pth, bbox_inches="tight", facecolor="white")
        png_paths.append(pth)
    plt.close(fig)

    gif_path = out_dir / f"{mode}_tsne_evolution.gif"
    # \u9996\u5e27 600ms\u3001\u4e2d\u95f4 330ms\u3001\u672b\u5e27 900ms\uff0c\u8ba9\u89c2\u4f17\u770b\u6e05\u8d77\u70b9\u548c\u7ec8\u70b9\u3002
    # \u7528 PIL \u800c\u4e0d\u662f imageio\uff1aPIL \u76f4\u63a5\u652f\u6301\u9010\u5e27 duration\uff08\u6beb\u79d2\uff09\uff0c
    # imageio \u7684 GIF \u63d2\u4ef6\u4f1a\u628a duration \u5f53 fps \u89e3\u91ca\uff0c\u4f20 list \u4f1a\u76f4\u63a5\u62a5\u9519\u3002
    durations = ([600] + [330] * (len(png_paths) - 2) + [900]) if len(png_paths) > 2 \
        else [500] * len(png_paths)
    from PIL import Image
    imgs = [Image.open(p) for p in png_paths]
    imgs[0].save(str(gif_path), save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0)
    print(f"[{mode}] GIF \u5df2\u751f\u6210\uff1a{gif_path}\uff08{len(png_paths)} \u5e27\uff09")

    if mp.exists():
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        hist = [dict(epoch=int(r["epoch"]),
                     acc_source=float(r["acc_source"]),
                     acc_target=float(r["acc_target"]),
                     acc_domain=float(r["acc_domain"]) if r["acc_domain"] else None)
                for r in rows]
        plot_curves(hist, mode, out_dir, cjk)


def plot_curves(history, mode, out_dir: Path, cjk: bool):
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in history]
    has_dom = any(r["acc_domain"] is not None for r in history)

    fig, ax1 = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax1.plot(ep, [r["acc_source"] * 100 for r in history],
             label="源域准确率" if cjk else "Source", lw=1.8)
    ax1.plot(ep, [r["acc_target"] * 100 for r in history],
             label="目标域准确率" if cjk else "Target", lw=1.8)
    if has_dom:
        ax1.plot(ep, [r["acc_domain"] * 100 for r in history],
                 label="域判别器准确率" if cjk else "Domain clf", lw=1.4, ls="--")
    ax1.axhline(50, color="gray", lw=0.8, ls=":")
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("准确率 (%)", fontsize=13)
    ax1.set_title(("DANN 训练曲线" if mode == "dann" else "source-only 训练曲线")
                  if cjk else f"{mode} training curves", fontsize=11)
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.tick_params(labelsize=12)
    fig.tight_layout()
    p = out_dir / f"{mode}_curves.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[{mode}] 曲线已保存：{p}")


def compare_from_csv(out_dir: Path):
    """
    只跑单个 mode 时也要能出对比图：直接读两个 metrics_*.csv 的末行。
    这样 source-only 与 DANN 分两次跑也能拿到同一张对比柱状图。
    """
    import matplotlib.pyplot as plt

    cjk = setup_matplotlib_cjk()
    names, vals = [], []
    for mode, label in (("source", "source-only"), ("dann", "DANN")):
        mp = out_dir / f"metrics_{mode}.csv"
        if not mp.exists():
            return
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        if not rows:
            return
        names.append(label)
        vals.append(float(rows[-1]["acc_target"]) * 100)
    if len(vals) < 2:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    bars = ax.bar(names, vals, color=["#9aa0a6", "#1f77b4"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}%", ha="center", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("目标域准确率 (%)")
    ax.set_title("迁移前后对比" if cjk else "Before vs after adaptation", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out_dir / "compare.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[compare] 已保存：{p}（{names[0]} {vals[0]:.2f}% vs {names[1]} {vals[1]:.2f}%）")


def plot_compare(results, out_dir: Path):
    import matplotlib.pyplot as plt

    cjk = setup_matplotlib_cjk()
    names, vals = [], []
    for r in results:
        if r.get("final"):
            names.append("source-only" if r["mode"] == "source" else "DANN")
            vals.append(r["final"]["acc_target"] * 100)
    if len(vals) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    bars = ax.bar(names, vals, color=["#9aa0a6", "#1f77b4"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f}%", ha="center", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel("目标域准确率 (%)")
    ax.set_title("迁移前后对比" if cjk else "Before vs after adaptation", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = out_dir / "compare.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[compare] 已保存：{p}")


# ----------------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------------

def build_fixed_subset(src_te, tgt_te, n, seed):
    """
    固定可视化 / 评估子集：同一次运行里始终用同一批样本，
    否则每帧的点不是同一批，动画会假抖动。
    """
    g = np.random.default_rng(seed + 7)
    n = min(n, len(src_te), len(tgt_te))
    idx = g.choice(len(src_te), size=n, replace=False)

    xs, ys, xt, yt = [], [], [], []
    for i in idx:
        a, b = src_te[int(i)]
        c, d = tgt_te[int(i)]
        xs.append(a); ys.append(b); xt.append(c); yt.append(d)
    return torch.stack(xs), torch.stack(xt), torch.tensor(ys), torch.tensor(yt)


def make_samples_grid(data, out_dir: Path, n: int = 10):
    """汇报第 2 页的素材：上排源域（黑底白字），下排目标域（彩色背景），同列同标签。"""
    import matplotlib.pyplot as plt

    cjk = setup_matplotlib_cjk()
    _, _, src_te, tgt_te, _ = data
    g = np.random.default_rng(0)
    idx = g.choice(len(src_te), size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(1.05 * n, 2.6), dpi=150)
    for j, i in enumerate(idx):
        a, _ = src_te[int(i)]
        b, _ = tgt_te[int(i)]
        axes[0, j].imshow(a.permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, j].imshow(b.permute(1, 2, 0) * 0.5 + 0.5)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            for s in axes[r, j].spines.values():
                s.set_visible(False)
    fig.text(0.012, 0.72, "源域\nMNIST" if cjk else "Source\nMNIST", fontsize=10, va="center")
    fig.text(0.012, 0.28, "目标域\nMNIST-M" if cjk else "Target\nMNIST-M", fontsize=10, va="center")
    fig.suptitle("同一个任务，两个分布" if cjk else "One task, two distributions", fontsize=12)
    fig.tight_layout(rect=(0.05, 0, 1, 0.95))
    p = out_dir / "sample_grid.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[samples] 已保存：{p}")


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="DANN on MNIST -> MNIST-M with t-SNE evolution GIF")
    ap.add_argument("--mode", choices=["both", "dann", "source"], default="both")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--vis-every", type=int, default=2, help="每几个 epoch 采一帧")
    ap.add_argument("--vis-per-domain", type=int, default=500,
                    help="每帧每域取多少样本（500+500 足够，25 帧约 90MB 内存）")
    ap.add_argument("--train-limit", type=int, default=0, help="只用前 N 个样本，调试用")
    ap.add_argument("--bg-mode", choices=["auto", "bsds", "synthetic"], default="auto")
    ap.add_argument("--bsds-dir", type=str, default="", help="BSDS500 解压目录，留空则自动降级")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 上建议 0")
    ap.add_argument("--gamma", type=float, default=10.0, help="λ 调度陡度（论文值 10）")
    ap.add_argument("--fixed-lambda", action="store_true",
                    help="消融：λ 全程固定为 1，不用调度（用于负迁移展示）")
    ap.add_argument("--make-samples", action="store_true",
                    help="只生成源/目标样本网格图，不训练")
    ap.add_argument("--no-precompute", action="store_true", help="惰性数据管线，省内存但慢很多")
    ap.add_argument("--device", type=str, default="", help="留空自动选择 xpu/cuda/cpu")
    ap.add_argument("--render-only", action="store_true",
                    help="不训练，只用 outputs/feat 下已存盘的特征重新出图")
    ap.add_argument("--resume", action="store_true",
                    help="从 outputs/ckpt_*.pt 续跑，用于分段把 50 epoch 跑完")
    ap.add_argument("--no-render", action="store_true",
                    help="本次不渲染 GIF（分段跑的中间段用，最后一段再渲染）")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：2 epoch + 小样本，约 1 分钟")
    args = ap.parse_args()

    if args.bs % 2 != 0:
        ap.error(f"--bs 必须是偶数（batch 由源/目标各 bs//2 拼成），收到 {args.bs}")

    if args.smoke:
        args.epochs = 2
        args.vis_every = 1
        args.vis_per_domain = 100
        args.train_limit = args.train_limit or 4000

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device or None)

    if args.render_only:
        for m in (["source", "dann"] if args.mode == "both" else [args.mode]):
            render_from_disk(m, out_dir, args)
        return

    t0 = time.time()
    src_tr, tgt_tr, src_te, tgt_te, bg_used = build_datasets(
        data_dir, args.bg_mode, Path(args.bsds_dir) if args.bsds_dir else None,
        args.seed, precompute=not args.no_precompute)
    print(f"[data] 背景来源 = {bg_used}，数据构建耗时 {time.time() - t0:.1f}s")

    if args.make_samples:
        make_samples_grid((src_tr, tgt_tr, src_te, tgt_te, bg_used), out_dir)
        return

    fixed = build_fixed_subset(src_te, tgt_te, args.vis_per_domain, args.seed)
    print(f"[data] 固定评估子集：{fixed[0].shape[0]} 源 + {fixed[1].shape[0]} 目标")

    results = [train(m, args, device, (src_tr, tgt_tr, src_te, tgt_te, bg_used),
                     fixed, out_dir)
               for m in (["source", "dann"] if args.mode == "both" else [args.mode])]

    # source / dann 分开跑时各自会写一次 summary——按 mode 合并而不是覆盖，
    # 否则先跑完的实验结果会被后一次单 mode 运行冲掉。
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            old = None
        if isinstance(old, dict) and isinstance(old.get("results"), list):
            new_modes = {r["mode"] for r in results}
            merged = [r for r in old["results"] if r.get("mode") not in new_modes]
            results = merged + results

    summary_path.write_text(
        json.dumps(dict(background=bg_used, device=str(device), results=results),
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if len(results) == 2:
        plot_compare(results, out_dir)
    else:
        # 分两次跑（先 source 后 dann）时，用两份 CSV 合成同一张对比图
        compare_from_csv(out_dir)

    print("\n================ 结果 ================")
    for r in results:
        f = r.get("final") or {}
        dom = f.get("acc_domain")
        print(f"{r['mode']:>6}: 目标域 {(f.get('acc_target', 0) * 100):.2f}%  "
              f"源域 {(f.get('acc_source', 0) * 100):.2f}%  "
              f"域判别器 {('%.2f%%' % (dom * 100)) if dom is not None else 'n/a'}  "
              f"耗时 {r['elapsed_sec']}s")
    if len(results) == 2:
        b = results[0]["final"]["acc_target"] if results[0]["final"] else 0.0
        d = results[1]["final"]["acc_target"] if results[1]["final"] else 0.0
        print(f"迁移增益: +{(d - b) * 100:.2f} 个百分点")
    print(f"全部产出在：{out_dir}")


if __name__ == "__main__":
    main()
