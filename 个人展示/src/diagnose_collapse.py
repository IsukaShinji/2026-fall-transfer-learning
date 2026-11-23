"""只读诊断：对比正常/崩溃种子的最终 checkpoint，定位塌缩机制。
不训练、不改任何产物。用法：python src/diagnose_collapse.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dann_mnistm import DANN, set_seed

CKPTS = {
    "normal_seed42": HERE.parent / "outputs/ablation_fixed/ckpt_dann.pt",
    "normal_seed43": HERE.parent / "outputs_fixed_seed43/ckpt_dann.pt",
    "collapsed_seed44": HERE.parent / "outputs_fixed_seed44/ckpt_dann.pt",
    "collapsed_seed45": HERE.parent / "outputs_fixed_seed45/ckpt_dann.pt",
    "collapsed_seed46": HERE.parent / "outputs_fixed_seed46/ckpt_dann.pt",
}


def layer_stats(sd):
    lines = []
    for k, v in sd.items():
        name = k
        a = v.detach().float()
        naninf = int((~torch.isfinite(a)).sum())
        lines.append(
            f"  {name:26s} L2={a.norm():9.3f} mean={a.mean():+.4f} "
            f"std={a.std():.4f} absmax={a.abs().max():.3f} "
            f"near0={100*(a.abs()<1e-6).float().mean():5.1f}% naninf={naninf}"
        )
    return "\n".join(lines)


def forward_stats(model):
    """同一批固定随机输入，看特征/输出是否塌成常数。"""
    torch.manual_seed(0)
    x = torch.randn(64, 3, 28, 28)
    with torch.no_grad():
        feat = model.extractor(x)
        logits_y, logits_d, _ = model(x, 0.0)
    # 每个样本的特征向量之间的平均距离（塌缩则≈0）
    fc = feat - feat.mean(0, keepdim=True)
    feat_spread = fc.norm(dim=1).mean().item()
    # 输出 logits 跨样本的方差（塌缩则≈0）
    logit_std = logits_y.std().item()
    pred = logits_y.argmax(1)
    top1_frac = (pred == pred[0]).float().mean().item()
    return feat_spread, logit_std, top1_frac, logits_y[0][:5].tolist()


def main():
    # 参考：全新初始化
    set_seed(42)
    fresh = DANN()
    print("=" * 100)
    print("参考：全新初始化 (seed42)")
    print("=" * 100)
    print(layer_stats(fresh.state_dict()))
    print("forward:", forward_stats(fresh))

    for name, path in CKPTS.items():
        print("\n" + "=" * 100)
        print(f"{name}  ({path.name})")
        print("=" * 100)
        if not path.exists():
            print("  缺失")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        print(f"  epoch={ck.get('epoch')}  hparams={ck.get('hparams')}")
        print(layer_stats(sd))
        model = DANN()
        model.load_state_dict(sd)
        model.eval()
        fs, ls, frac, sample = forward_stats(model)
        print(f"  forward: 特征跨样本离散度={fs:.4f}  logits跨样本std={ls:.4f}  "
              f"首类占比={frac:.1%}  首样本logits前5={[round(v,3) for v in sample]}")


if __name__ == "__main__":
    main()
