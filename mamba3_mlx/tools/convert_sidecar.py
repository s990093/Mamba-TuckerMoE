"""One-time conversion: PyTorch-style .npz → MLX-native sidecar.

Usage:
  python tools/convert_sidecar.py <checkpoint.npz> [--dtype bf16|fp32|fp16]

After running this once, load_checkpoint() will automatically use the sidecar
via mmap (instant load) on every subsequent call.
"""
import argparse
import sys
import time
from pathlib import Path

# Resolve repo root so mamba3_mlx is importable from any cwd.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import mlx.core as mx

from mamba3_mlx.utils.config import Mamba3Config
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import convert_to_mlx_native, _sidecar_path

_DTYPE_MAP = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Source .npz checkpoint path")
    p.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    p.add_argument("--out", default=None, help="Override output path")
    p.add_argument("--force", action="store_true", help="Re-convert even if sidecar exists")
    args = p.parse_args()

    dtype = _DTYPE_MAP[args.dtype]
    src = args.checkpoint
    sidecar = Path(args.out) if args.out else _sidecar_path(src, dtype)

    if sidecar.exists() and not args.force:
        print(f"[convert-sidecar] already exists: {sidecar}")
        print("  Use --force to re-convert.")
        return

    print(f"[convert-sidecar] source  : {src}")
    print(f"[convert-sidecar] target  : {sidecar}  (dtype={args.dtype})")
    print(f"[convert-sidecar] building model skeleton …")

    cfg = Mamba3Config()
    model = Mamba3LanguageModel(cfg)

    print(f"[convert-sidecar] converting (key-mapping + mx.savez) …")
    t0 = time.time()
    out = convert_to_mlx_native(model, src, out_path=str(sidecar) if args.out else None,
                                dtype=dtype)
    elapsed = time.time() - t0
    size_mb = out.stat().st_size / 1_048_576
    print(f"[convert-sidecar] done in {elapsed:.1f}s  →  {out}  ({size_mb:.0f} MB)")
    print(f"[convert-sidecar] subsequent load_checkpoint() calls will use mmap instantly.")


if __name__ == "__main__":
    main()
