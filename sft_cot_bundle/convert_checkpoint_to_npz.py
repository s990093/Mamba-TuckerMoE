#!/usr/bin/env python3
"""Convert PyTorch checkpoint to NPZ format."""

import torch
import numpy as np
import json
from pathlib import Path
import sys

def convert_tensor_dict_to_numpy(d, prefix=""):
    """Recursively convert torch tensors to numpy arrays."""
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, torch.Tensor):
            result[new_key] = value.cpu().numpy()
        elif isinstance(value, dict):
            result.update(convert_tensor_dict_to_numpy(value, f"{new_key}_"))
        elif isinstance(value, (list, tuple)):
            # Skip complex structures like these
            print(f"Skipping {new_key} (type: {type(value).__name__})")
        else:
            # Store scalars as numpy arrays
            try:
                result[new_key] = np.array(value)
            except:
                print(f"Skipping {new_key} (cannot convert: {type(value).__name__})")
    return result

def main():
    checkpoint_path = Path("/home/hungwei/llm/sft_cot_bundle/output/checkpoint_sft_cot_s18500.pt")
    output_path = Path("/home/hungwei/llm/sft_cot_bundle/output/checkpoint_sft_cot_s18500.npz")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Model state dict size: {len(checkpoint.get('model', {}))} parameters")

    # Convert to numpy
    print("Converting to numpy arrays...")
    numpy_data = {}

    # Process model weights
    if 'model' in checkpoint:
        print(f"  Converting model weights...")
        numpy_data.update(convert_tensor_dict_to_numpy(checkpoint['model'], 'model_'))

    # Process metadata
    for key in ['global_step', 'sft', 'router_temperature']:
        if key in checkpoint:
            numpy_data[key] = np.array(checkpoint[key])
            print(f"  Added {key}")

    # Process config as JSON string
    if 'config' in checkpoint:
        config_str = json.dumps(checkpoint['config'], indent=2, default=str)
        numpy_data['config_json'] = np.array(config_str, dtype=object)
        print(f"  Added config as JSON string")

    print(f"\nTotal arrays in NPZ: {len(numpy_data)}")

    # Save as NPZ
    print(f"Saving to NPZ: {output_path}")
    np.savez_compressed(str(output_path), **numpy_data)

    output_size = output_path.stat().st_size / (1024**3)
    print(f"NPZ file size: {output_size:.2f} GB")

    return str(output_path)

if __name__ == "__main__":
    npz_path = main()
    print(f"\nConversion complete: {npz_path}")
