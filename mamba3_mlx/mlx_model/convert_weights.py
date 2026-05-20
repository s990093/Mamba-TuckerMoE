"""
Convert PyTorch checkpoint to MLX format (.npz).

Usage:
    python convert_weights.py --pt_path checkpoint.pt --output checkpoint.npz
"""

import argparse
import numpy as np
import mlx.core as mx

try:
    import torch
except ImportError:
    torch = None


def convert_pytorch_to_mlx(pt_path, output_path=None):
    """
    Convert PyTorch checkpoint to MLX .npz format.

    Args:
        pt_path: Path to PyTorch .pt checkpoint
        output_path: Path to save .npz (default: replace .pt with .npz)

    Returns:
        Dict of converted weights
    """
    if torch is None:
        raise ImportError("torch required for conversion from .pt files")

    print(f"[Convert] Loading PyTorch checkpoint from {pt_path}...")
    checkpoint = torch.load(pt_path, map_location="cpu")

    # Extract model state dict
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"[Convert] Found {len(state_dict)} parameters")

    # Convert to MLX arrays
    mlx_params = {}
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            # Convert to numpy then to MLX array
            np_array = param.detach().cpu().numpy()
            mlx_params[name] = np_array
            print(f"  {name}: {np_array.shape} {np_array.dtype}")
        else:
            # Keep non-tensor data as-is
            mlx_params[name] = param

    # Handle embedding weight sharing (lm_head from embedding)
    if "embed.weight" in mlx_params and "lm_head.weight" not in mlx_params:
        mlx_params["lm_head.weight"] = mlx_params["embed.weight"]
        print("[Convert] Shared embedding weight with lm_head")

    # Save to .npz
    if output_path is None:
        output_path = pt_path.replace(".pt", ".npz")

    print(f"[Convert] Saving to {output_path}...")
    np.savez(output_path, **mlx_params)
    print(f"[Convert] Done!")

    return mlx_params


def load_npz_weights(npz_path):
    """
    Load MLX weights from .npz file.

    Args:
        npz_path: Path to .npz checkpoint

    Returns:
        Dict of numpy arrays
    """
    print(f"[Load] Loading weights from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    weights = {name: data[name] for name in data.files}
    print(f"[Load] Loaded {len(weights)} parameters")
    return weights


def save_npz_weights(weights, output_path):
    """
    Save weights to .npz format.

    Args:
        weights: Dict of numpy arrays or MLX arrays
        output_path: Path to save .npz file
    """
    print(f"[Save] Saving {len(weights)} parameters to {output_path}...")

    # Convert MLX arrays to numpy if needed
    np_weights = {}
    for name, param in weights.items():
        if isinstance(param, mx.array):
            np_weights[name] = np.array(param)
        else:
            np_weights[name] = np.array(param)

    np.savez(output_path, **np_weights)
    print(f"[Save] Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch → MLX weights")
    parser.add_argument("--pt_path", type=str, required=True,
                        help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output .npz (default: .pt → .npz)")

    args = parser.parse_args()

    # Convert
    convert_pytorch_to_mlx(args.pt_path, args.output)


if __name__ == "__main__":
    main()
