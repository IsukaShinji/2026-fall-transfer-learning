#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把消融实验的四条 acc_target 曲线画到一张图里。"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dann_mnistm import setup_console, setup_matplotlib_cjk  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

RUNS = [
    ("source",            OUT / "metrics_source.csv",                       "source-only（不迁移）", "#9aa0a6", ":"),
    ("dann_gamma2",       OUT / "ablation_gamma2/metrics_dann.csv",          "DANN，γ=2（调度过慢）", "#2ca02c", "-."),
    ("dann_gamma10",      OUT / "metrics_dann.csv",                          "DANN，γ=10（论文值）", "#1f77b4", "-"),
    ("dann_fixed1",       OUT / "ablation_fixed/metrics_dann.csv",           "DANN，λ 固定 1", "#d62728", "--"),
]


def main():
    setup_console()
    cjk = setup_matplotlib_cjk()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    found = 0
    for _, path, label, color, ls in RUNS:
        if not path.exists():
            print(f"跳过（无文件）：{path}")
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        v = [float(r["acc_target"]) * 100 for r in rows]
        ax.plot(ep, v, label=label, color=color, ls=ls, lw=1.8)
        found += 1
        print(f"{label}: 末值 {v[-1]:.2f}%（{len(v)} epoch）")
    if not found:
        print("没有任何数据")
        return

    ax.axhline(50, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("目标域准确率 (%)" if cjk else "Target accuracy (%)")
    ax.tick_params(labelsize=12)
    ax.set_title("λ 调度消融（synthetic 背景，各 50 epoch）" if cjk
                 else "Lambda schedule ablation", fontsize=12)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = OUT / "ablation_curves.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存：{p}")


if __name__ == "__main__":
    main()
