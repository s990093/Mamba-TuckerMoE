import mlx.core as mx
import time
from inference.lib.mlx_hybrid_infer import Mamba3Config, Mamba3LanguageModel
import argparse

def main():
    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6, mimo_rank=4, num_kv_heads=4,
        chunk_size=64, use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=32, kmoe_r2=512, kmoe_r3=256,
        ffn_expand=6
    )
    config.tucker_scalar_fuse = True
    model = Mamba3LanguageModel(config, 32007)
    import mlx.nn as nn
    nn.quantize(model, group_size=64, bits=4)
    model.apply(lambda x: x.astype(mx.bfloat16) if x.dtype != mx.uint32 else x)
    mx.eval(model.parameters())
    
    # Warmup
    x = mx.zeros((1, 1), dtype=mx.int32)
    caches = [None] * len(model.backbone.layers)
    seq_pos = mx.array(10, dtype=mx.int32)
    rt = mx.array(0.5, dtype=mx.bfloat16)

    def step(x, c, sp):
        return model(x, caches=c, seq_pos=sp, router_temp=rt)

    compiled_step = mx.compile(step)

    # force compile
    out, caches = compiled_step(x, caches, seq_pos)
    mx.eval(out, caches)

    # Benchmark
    start = time.perf_counter()
    steps = 100
    for _ in range(steps):
        out, caches = compiled_step(x, caches, seq_pos)
        mx.eval(out, caches)
    end = time.perf_counter()
    print(f"End-to-End FPS: {steps / (end - start):.2f} tok/s")

if __name__ == '__main__':
    main()
