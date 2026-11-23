"""CORAL 基线：与 DANN 同骨干、同数据、同优化器，仅把对抗损失换成 CORAL 二阶对齐。

用法：python src/coral_baseline.py --epochs 50 --seed 42
输出：outputs_coral/metrics_coral.csv、summary_coral.json
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dann_mnistm import (DATA_DIR, DANN, accuracy, build_datasets,
                         lr_schedule, paired_batches, pick_device,
                         set_seed, setup_console)

OUT = HERE.parent / "outputs_coral"


def coral_loss(fs: torch.Tensor, ft: torch.Tensor, device: torch.device) -> torch.Tensor:
    """对齐源/目标特征的均值与协方差（Sun et al., 2016）。"""
    ms, mt = fs.mean(0), ft.mean(0)
    n, d = fs.shape
    eye = torch.eye(d, device=device) * 1e-4
    cs = (fs - ms).T @ (fs - ms) / max(n - 1, 1) + eye
    ct = (ft - mt).T @ (ft - mt) / max(n - 1, 1) + eye
    return (ms - mt).pow(2).mean() + (cs - ct).pow(2).sum() / (d * d)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = pick_device(args.device or None)
    OUT.mkdir(parents=True, exist_ok=True)
    src_tr, tgt_tr, src_te, tgt_te, used = build_datasets(
        DATA_DIR, "synthetic", None, args.seed, True)
    sl = torch.utils.data.DataLoader(src_tr, batch_size=args.bs // 2, shuffle=True, drop_last=True)
    tl = torch.utils.data.DataLoader(tgt_tr, batch_size=args.bs // 2, shuffle=True, drop_last=True)
    te = torch.utils.data.DataLoader(tgt_te, batch_size=512, shuffle=False)
    se = torch.utils.data.DataLoader(src_te, batch_size=512, shuffle=False)

    model = DANN().to(dev)          # 只用 extractor + classifier；discriminator 不参与
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    steps = min(len(sl), len(tl))
    t0 = time.time()
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        p = (epoch - 1) / max(args.epochs - 1, 1)
        for g in opt.param_groups:
            g["lr"] = lr_schedule(p, args.lr)
        for xs, ys, xt, _ in paired_batches(sl, tl, steps, steps):
            xs, ys, xt = xs.to(dev), ys.to(dev), xt.to(dev)
            opt.zero_grad()
            logits_s, _, fs = model(xs, 0.0)
            _, _, ft = model(xt, 0.0)
            loss = F.cross_entropy(logits_s, ys) + coral_loss(fs, ft, dev)
            loss.backward()
            opt.step()
        row = dict(epoch=epoch, acc_source=accuracy(model, se, dev),
                   acc_target=accuracy(model, te, dev))
        hist.append(row)
        print(f"[coral] epoch {epoch:2d}/{args.epochs} "
              f"source {row['acc_source']:.4f} target {row['acc_target']:.4f}", flush=True)

    final = dict(mode="coral", background=used, device=str(dev), seed=args.seed,
                 elapsed_sec=round(time.time() - t0, 1),
                 acc_source=hist[-1]["acc_source"], acc_target=hist[-1]["acc_target"])
    with (OUT / "metrics_coral.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "acc_source", "acc_target"])
        w.writeheader(); w.writerows(hist)
    (OUT / "summary_coral.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[coral] 完成：目标域 {final['acc_target']:.2%}，耗时 {final['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
