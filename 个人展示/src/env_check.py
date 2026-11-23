"""环境自检脚本：确认 Intel Arc XPU 可用，并给出粗略算力参考。

用法（任选其一）：
    Windows 终端  :  .venv2\\Scripts\\python.exe env_check.py
    PyCharm       :  右键 Run 'env_check'（解释器需为 .venv2）
    Git Bash      :  ./.venv2/Scripts/python.exe env_check.py

注意：必须用 .venv2（torch 2.9.0+xpu）。用 .venv 会得到纯 CPU 版本，
      不会报错，但训练速度差数倍，容易被误判成"代码写错了"。
"""

import platform
import sys
import time


def line(title: str) -> None:
    print(f"\n{'=' * 52}\n{title}\n{'=' * 52}")


def main() -> int:
    line("基础环境")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"解释器路径  : {sys.executable}")
    print(f"操作系统    : {platform.system()} {platform.release()}")

    line("PyTorch 与设备")
    try:
        import torch
    except ImportError:
        print("[FAIL] 未安装 torch。请确认解释器指向 .venv2")
        return 1

    print(f"torch       : {torch.__version__}")
    cpu_ok = True
    print(f"CPU 可用    : {cpu_ok}")

    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"XPU 可用    : {xpu_ok}")
    if xpu_ok:
        print(f"设备名      : {torch.xpu.get_device_name(0)}")
        prop = torch.xpu.get_device_properties(0)
        print(f"计算单元    : {prop.max_compute_units} EU")
        print(f"显存总量    : {prop.total_memory / 1024 ** 3:.1f} GB（与内存共享）")
        print(f"fp16 支持   : {bool(prop.has_fp16)}")
    else:
        print("[WARN] XPU 不可用 —— 检查是否误用了 .venv（纯 CPU 版）")

    line("依赖检查")
    for mod in ("numpy", "matplotlib", "PIL", "torchvision", "sklearn", "tqdm"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{mod:<12}: {ver}")
        except ImportError:
            print(f"{mod:<12}: [缺失]")

    if not xpu_ok:
        return 0

    line("粗略算力（4096x4096 matmul，仅供参考）")
    dev = "xpu"
    for dtype in (torch.float32, torch.float16):
        a = torch.randn(4096, 4096, device=dev, dtype=dtype)
        b = torch.randn(4096, 4096, device=dev, dtype=dtype)
        for _ in range(3):          # 预热
            a @ b
        torch.xpu.synchronize()
        start = time.time()
        repeats = 10
        for _ in range(repeats):
            a @ b
        torch.xpu.synchronize()
        elapsed = (time.time() - start) / repeats
        tflops = 2 * 4096 ** 3 / elapsed / 1e12
        name = str(dtype).split(".")[-1]
        print(f"{name:<10}: {tflops:6.2f} TFLOPS  ({elapsed * 1000:.1f} ms / matmul)")

    line("结论")
    print("XPU 就绪。本机适合：MNIST / CIFAR 级别的 CNN、ResNet-18 @ <=128px。")
    print("不适合：224px 大分辨率长训练、ViT-L 以上模型、大规模语言模型微调。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
