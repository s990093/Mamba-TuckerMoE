import mlx.core as mx
import time
from inference.lib.mlx_hybrid_infer import Mamba3Config, Mamba3LanguageModel
import mlx.nn as nn

def main():
    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6, mimo_rank=4, num_kv_heads=4,
        chunk_size=64, use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=32, kmoe_r2=512, kmoe_r3=256,
        ffn_expand=6
    )
    config.tucker_scalar_fuse = True
    model = Mamba3LanguageModel(config, 32007)
    nn.quantize(model, group_size=64, bits=4)
    model.apply(lambda x: x.astype(mx.bfloat16) if x.dtype != mx.uint32 else x)
    mx.eval(model.parameters())

    rt = mx.array(0.5, dtype=mx.bfloat16)
    seq_pos = mx.array(10, dtype=mx.int32)

    # 1. Profile Mamba3Block
    x1 = mx.zeros((1, 1, 768), dtype=mx.bfloat16)
    mamba_layer = model.backbone.layers[0]
    out, mamba_c = mamba_layer(x1, cache=None, router_temp=rt)
    mx.eval(mamba_c)

    def step_mamba(x, c): return mamba_layer(x, cache=c, router_temp=rt)
    comp_mamba = mx.compile(step_mamba)
    out, mamba_c = comp_mamba(x1, mamba_c); mx.eval(out, mamba_c)
    
    start = time.perf_counter()
    for _ in range(100): out, mamba_c = comp_mamba(x1, mamba_c); mx.eval(out, mamba_c)
    print(f"Mamba3Block: {100 / (time.perf_counter() - start):.2f} iter/s ({1000 * (time.perf_counter() - start) / 100:.2f} ms/iter)")

    # 2. Profile TransformerBlock
    x2 = mx.zeros((1, 1, 768), dtype=mx.bfloat16)
    tf_layer = model.backbone.layers[-1]
    k_cache = mx.zeros((1, 12, 2048, 64), dtype=mx.bfloat16)
    v_cache = mx.zeros((1, 12, 2048, 64), dtype=mx.bfloat16)
    tf_c = (k_cache, v_cache)
    
    def step_tf(x, c, sp): return tf_layer(x, cache=c, seq_pos=sp, router_temp=rt)
    comp_tf = mx.compile(step_tf)
    out, tf_c = comp_tf(x2, tf_c, seq_pos); mx.eval(out, tf_c)
    
    start = time.perf_counter()
    for _ in range(100): out, tf_c = comp_tf(x2, tf_c, seq_pos); mx.eval(out, tf_c)
    print(f"TransformerBlock: {100 / (time.perf_counter() - start):.2f} iter/s ({1000 * (time.perf_counter() - start) / 100:.2f} ms/iter)")

    # 3. Profile TuckerMoE only
    moe = mamba_layer.out_proj
    def step_moe(x): return moe(x, rt, scalar_fuse=True)
    comp_moe = mx.compile(step_moe)
    out = comp_moe(x1); mx.eval(out)
    start = time.perf_counter()
    for _ in range(100): out = comp_moe(x1); mx.eval(out)
    print(f"TuckerMoE: {100 / (time.perf_counter() - start):.2f} iter/s ({1000 * (time.perf_counter() - start) / 100:.2f} ms/iter)")

if __name__ == '__main__':
    main()
