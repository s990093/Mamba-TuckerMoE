import mlx.core as mx
import time

def fused_tucker_metal_basic(probs: mx.array, x_shared: mx.array, g_sel: mx.array) -> mx.array:
    """
    Custom Metal Kernel for TuckerMoE Fused Einsum:
    Σ_{k,r} p_{b,k} * x_{b,r} * G_{b,k,r,s} -> out_{b,s}
    
    probs: (B, K)
    x_shared: (B, R1)
    g_sel: (B, K, R1, R2)
    Returns: (B, R2)
    """
    B, K = probs.shape
    _, R1 = x_shared.shape
    _, _, _, R2 = g_sel.shape
    
    source = """
        uint s = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        
        if (b >= B || s >= R2) return;
        
        T sum = 0.0;
        for (uint k = 0; k < K; ++k) {
            T prob = probs[b * K + k];
            for (uint r = 0; r < R1; ++r) {
                T x_val = x_shared[b * R1 + r];
                T g_val = g_sel[b * K * R1 * R2 + k * R1 * R2 + r * R2 + s];
                sum += prob * x_val * g_val;
            }
        }
        out[b * R2 + s] = sum;
    """
    
    kernel = mx.fast.metal_kernel(
        name=f"fused_tucker_einsum_b{B}_k{K}_r{R1}_{R2}",
        input_names=["probs", "x_shared", "g_sel"],
        output_names=["out"],
        source=source,
        header=f"constant uint B = {B}; constant uint K = {K}; constant uint R1 = {R1}; constant uint R2 = {R2};"
    )
    
    outputs = kernel(
        inputs=[probs, x_shared, g_sel],
        template=[("T", probs.dtype)],
        grid=(R2, B, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, R2)],
        output_dtypes=[probs.dtype],
    )
    return outputs[0]


def fused_tucker_metal_sram(probs: mx.array, x_shared: mx.array, g_sel: mx.array) -> mx.array:
    """
    Same fused einsum, but caches x_shared in threadgroup SRAM per output-s tile.
    This cuts repeated global reads of the latent vector and gives a small tile-size experiment.
    """
    B, K = probs.shape
    _, R1 = x_shared.shape
    _, _, _, R2 = g_sel.shape
    tg_size = 128
    bank_shift = 5
    r1_pad = R1 + ((R1 + ((1 << bank_shift) - 1)) >> bank_shift)

    source = """
        uint s = thread_position_in_grid.x;
        uint b = thread_position_in_grid.y;
        uint lid = thread_index_in_threadgroup;

        threadgroup float x_sram[R1_PAD];
        for (uint r = lid; r < R1; r += TG_SIZE) {
            x_sram[r + (r >> BANK_SHIFT)] = float(x_shared[b * R1 + r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (b >= B || s >= R2) return;

        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            float prob = float(probs[b * K + k]);
            uint g_base = b * K * R1 * R2 + k * R1 * R2 + s;
            uint r = 0;
            for (; r + 3 < R1; r += 4) {
                float x0 = x_sram[r + (r >> BANK_SHIFT)];
                float x1 = x_sram[(r + 1) + ((r + 1) >> BANK_SHIFT)];
                float x2 = x_sram[(r + 2) + ((r + 2) >> BANK_SHIFT)];
                float x3 = x_sram[(r + 3) + ((r + 3) >> BANK_SHIFT)];
                float g0 = float(g_sel[g_base + r * R2]);
                float g1 = float(g_sel[g_base + (r + 1) * R2]);
                float g2 = float(g_sel[g_base + (r + 2) * R2]);
                float g3 = float(g_sel[g_base + (r + 3) * R2]);
                sum += prob * (x0 * g0 + x1 * g1 + x2 * g2 + x3 * g3);
            }
            for (; r < R1; ++r) {
                sum += prob * x_sram[r + (r >> BANK_SHIFT)] * float(g_sel[g_base + r * R2]);
            }
        }
        out[b * R2 + s] = T(sum);
    """

    kernel = mx.fast.metal_kernel(
        name=f"fused_tucker_einsum_sram_b{B}_k{K}_r{R1}_{R2}",
        input_names=["probs", "x_shared", "g_sel"],
        output_names=["out"],
        source=source,
        header=(
            f"constant uint B = {B}; constant uint K = {K}; constant uint R1 = {R1}; "
            f"constant uint R2 = {R2}; constant uint TG_SIZE = {tg_size}; "
            f"constant uint BANK_SHIFT = {bank_shift}; constant uint R1_PAD = {r1_pad};"
        )
    )

    outputs = kernel(
        inputs=[probs, x_shared, g_sel],
        template=[("T", probs.dtype)],
        grid=(R2, B, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[(B, R2)],
        output_dtypes=[probs.dtype],
    )
    return outputs[0]


def fused_tucker_metal(probs: mx.array, x_shared: mx.array, g_sel: mx.array) -> mx.array:
    return fused_tucker_metal_sram(probs, x_shared, g_sel)

def benchmark():
    # Model parameters for TuckerMoE. Decode is usually B=1; prefill is larger.
    B = 64      # Batch size / sequence length
    K = 2       # Top K experts
    R1 = 256    # Tucker latent dim in benchmark_mlx.py config
    R2 = 512    # Tucker core output dim in benchmark_mlx.py config
    
    # Initialize random data
    probs = mx.random.uniform(shape=(B, K)).astype(mx.float32)
    x_shared = mx.random.normal(shape=(B, R1)).astype(mx.float32)
    g_sel = mx.random.normal(shape=(B, K, R1, R2)).astype(mx.float32)
    mx.eval(probs, x_shared, g_sel)
    
    # Verify correctness
    out_ref = mx.einsum("bk,br,bkrs->bs", probs, x_shared, g_sel)
    out_basic = fused_tucker_metal_basic(probs, x_shared, g_sel)
    out_sram = fused_tucker_metal_sram(probs, x_shared, g_sel)
    mx.eval(out_ref, out_basic, out_sram)
    
    diff_basic = mx.abs(out_ref - out_basic).max().item()
    diff_sram = mx.abs(out_ref - out_sram).max().item()
    diff = max(diff_basic, diff_sram)
    print(f"✅ Correctness Check: basic diff={diff_basic:.6f}, sram diff={diff_sram:.6f}")
    if diff > 1e-4:
        print("⚠️ Warning: Output mismatch between MLX native and Custom Metal!")
        
    # Warmup
    for _ in range(10):
        out_ref = mx.einsum("bk,br,bkrs->bs", probs, x_shared, g_sel)
        mx.eval(out_ref)
        
        out_basic = fused_tucker_metal_basic(probs, x_shared, g_sel)
        out_sram = fused_tucker_metal_sram(probs, x_shared, g_sel)
        mx.eval(out_basic, out_sram)
        
    # Benchmark Native MLX Einsum
    trials = 100
    
    t0 = time.perf_counter()
    for _ in range(trials):
        out_ref = mx.einsum("bk,br,bkrs->bs", probs, x_shared, g_sel)
        mx.eval(out_ref)
    native_time = (time.perf_counter() - t0) * 1000 / trials
    
    # Benchmark original Custom Metal Kernel
    t1 = time.perf_counter()
    for _ in range(trials):
        out_metal = fused_tucker_metal_basic(probs, x_shared, g_sel)
        mx.eval(out_metal)
    basic_time = (time.perf_counter() - t1) * 1000 / trials

    # Benchmark SRAM Custom Metal Kernel
    t2 = time.perf_counter()
    for _ in range(trials):
        out_metal = fused_tucker_metal_sram(probs, x_shared, g_sel)
        mx.eval(out_metal)
    sram_time = (time.perf_counter() - t2) * 1000 / trials
    
    print("\n📊 Microbenchmark Results (Fused TuckerMoE)")
    print("===========================================")
    print(f"Shape: B={B}, K={K}, R1={R1}, R2={R2}")
    print(f"Native MLX (einsum):  {native_time:.4f} ms")
    print(f"Basic Metal Kernel:   {basic_time:.4f} ms  ({native_time / basic_time:.2f}x vs native)")
    print(f"SRAM Metal Kernel:    {sram_time:.4f} ms  ({native_time / sram_time:.2f}x vs native)")
    print(f"SRAM vs Basic:        {basic_time / sram_time:.2f}x")
    print("===========================================")
    
if __name__ == "__main__":
    benchmark()
