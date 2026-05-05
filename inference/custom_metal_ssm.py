import mlx.core as mx
import time

def chunk_scan_mlx_native(la_c: mx.array, u_c: mx.array) -> mx.array:
    """
    Original MLX implementation of intra-chunk scan.
    la_c: (B, nc, Lc, H)
    u_c: (B, nc, Lc, H, D) 
    """
    b, nc, lc, h = la_c.shape
    d = u_c.shape[-1]
    
    _CLIP = mx.array(40.0, dtype=la_c.dtype)
    P = mx.cumsum(la_c, axis=2)
    diff = mx.expand_dims(P, 3) - mx.expand_dims(P, 2)
    diff = mx.clip(diff, -_CLIP, _CLIP)
    
    idx = mx.arange(lc)
    mask2d = (idx[:, None] >= idx[None, :]).astype(diff.dtype)
    mask5 = mx.reshape(mask2d, (1, 1, lc, lc, 1))
    
    decay5 = mx.exp(diff) * mask5
    M = mx.transpose(decay5, (0, 1, 4, 2, 3))
    
    u_flat = mx.transpose(u_c, (0, 1, 3, 2, 4)).reshape(b, nc, h, lc, d)
    h_flat = mx.matmul(M, u_flat)
    
    h_intra = mx.transpose(h_flat.reshape(b, nc, h, lc, d), (0, 1, 3, 2, 4))
    return h_intra

def chunk_scan_metal(la_c: mx.array, u_c: mx.array) -> mx.array:
    """
    Custom Metal Kernel for SSM Intra-chunk scan.
    la_c: (B, nc, Lc, H)
    u_c: (B, nc, Lc, H, D)
    """
    B, nc, Lc, H = la_c.shape
    _, _, _, _, D = u_c.shape
    
    source = """
        uint d = thread_position_in_grid.x;
        uint h = thread_position_in_grid.y;
        uint b_c = thread_position_in_grid.z;
        
        if (d >= D || h >= H || b_c >= B * nc) return;
        
        T h_val = 0.0;
        for (uint t = 0; t < Lc; ++t) {
            T la = la_c[b_c * Lc * H + t * H + h];
            T u  = u_c[b_c * Lc * H * D + t * H * D + h * D + d];
            
            // Clip log alpha to avoid inf/nan (similar to MLX clip 40)
            la = la > 40.0 ? 40.0 : la;
            la = la < -40.0 ? -40.0 : la;
            
            h_val = metal::exp(la) * h_val + u;
            out[b_c * Lc * H * D + t * H * D + h * D + d] = h_val;
        }
    """
    
    kernel = mx.fast.metal_kernel(
        name="ssm_chunk_scan",
        input_names=["la_c", "u_c"],
        output_names=["out"],
        source=source,
        header=f"constant uint B = {B}; constant uint nc = {nc}; constant uint Lc = {Lc}; constant uint H = {H}; constant uint D = {D};"
    )
    
    # 執行並派發
    outputs = kernel(
        inputs=[la_c, u_c],
        template=[("T", la_c.dtype)],
        grid=(D, H, B * nc),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, nc, Lc, H, D)],
        output_dtypes=[u_c.dtype],
    )
    return outputs[0]

def benchmark():
    B = 2
    nc = 8
    Lc = 64
    H = 32
    D = 128
    
    print(f"Setting up benchmark with Shape: B={B}, nc={nc}, Lc={Lc}, H={H}, D={D}")
    
    la_c = mx.random.normal(shape=(B, nc, Lc, H)).astype(mx.float32) * 0.1 - 1.0
    u_c = mx.random.normal(shape=(B, nc, Lc, H, D)).astype(mx.float32)
    mx.eval(la_c, u_c)
    
    # Verify correctness
    ref_out = chunk_scan_mlx_native(la_c, u_c)
    metal_out = chunk_scan_metal(la_c, u_c)
    mx.eval(ref_out, metal_out)
    
    diff = mx.abs(ref_out - metal_out).max().item()
    print(f"✅ Correctness Check: Max Diff = {diff:.6f}")
    if diff > 1e-4:
        print("⚠️ Warning: Output mismatch between MLX native and Custom Metal!")
        
    trials = 50
    for _ in range(5): # warmup
        o = chunk_scan_mlx_native(la_c, u_c)
        mx.eval(o)
        o2 = chunk_scan_metal(la_c, u_c)
        mx.eval(o2)
        
    t0 = time.perf_counter()
    for _ in range(trials):
        o = chunk_scan_mlx_native(la_c, u_c)
        mx.eval(o)
    native_time = (time.perf_counter() - t0) * 1000 / trials
    
    t1 = time.perf_counter()
    for _ in range(trials):
        o2 = chunk_scan_metal(la_c, u_c)
        mx.eval(o2)
    metal_time = (time.perf_counter() - t1) * 1000 / trials
    
    print("\n📊 Microbenchmark Results (SSM Chunk Scan)")
    print("===========================================")
    print(f"Native MLX (O(N^2) matmul): {native_time:.4f} ms")
    print(f"Custom Metal Kernel (O(N)): {metal_time:.4f} ms")
    print(f"Speedup:                    {native_time / metal_time:.2f}x")
    print("===========================================")

if __name__ == "__main__":
    benchmark()
