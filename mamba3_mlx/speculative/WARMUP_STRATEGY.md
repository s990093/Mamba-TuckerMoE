# Speculative Decoding Warmup Strategy Explained

## Why `--warmup_tokens 1024` Exists (and Why It's Not for Demos)

### The Problem It Solves

The `--warmup_tokens` parameter in `run_sjd_warm.py` addresses a **cache cold-start problem** in speculative decoding benchmarks:

```
Cold Start (first 100 tokens):
- N-gram cache: empty → low hit rate → ARL ≈ 2-3
- Suffix retriever: empty → no matches → falls back to carry seed

Warm State (after 1000+ tokens):
- N-gram cache: populated with common patterns → high hit rate
- Suffix retriever: large buffer → catches long repeated phrases
- Result: ARL ≈ 5-7, speedup 3×+
```

### What `--warmup_tokens` Does

**It does NOT warm up the model itself.** Instead:

1. **Generates `warmup_tokens` using standard AR-sampling** (untimed)
2. **Populates draft caches** (n-gram + suffix retrieval) from the generated text
3. **Then runs the timed benchmark** starting from a fresh prompt but with warm caches

This simulates the **steady-state performance** a user sees in:
- Multi-turn conversations
- Long document generation
- Continued sessions (not cold starts)

### Why 1024 Is Too Long for Demos

The default `--warmup_tokens 1024` is designed for **benchmarking**, not interactive use:

| Use Case | Recommended Warmup | Reason |
|----------|-------------------|--------|
| **Benchmark (steady-state)** | 1024-2048 | Measure peak performance after caches saturate |
| **Demo (interactive)** | 0-256 | Show realistic cold-start → warm transition |
| **Production (multi-turn)** | 0 (use session history) | Caches warm naturally from conversation |

### Recommended Warmup Strategy for Demos

For interactive demos, use **much shorter warmup** or **no warmup at all**:

```bash
# Demo 1: Cold start (realistic first-time user experience)
python -m mamba3_mlx.speculative.run_sjd_warm \
    --warmup_tokens 0 \
    --max_tokens 256 \
    --K 16

# Demo 2: Warm start (simulate 2nd+ turn in conversation)
python -m mamba3_mlx.speculative.run_sjd_warm \
    --warmup_tokens 128 \
    --max_tokens 256 \
    --K 16

# Demo 3: Show cold → warm transition (most realistic)
python -m mamba3_mlx.speculative.run_jacobi_sampling \
    --max_tokens 512 \
    --K 16 \
    --verbose
# (No warmup; watch ARL grow naturally as caches populate)
```

### Alternative: Use Session History for Warmup

For production/demo, **pre-populate caches from actual conversation history** instead of synthetic warmup:

```python
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.speculative.drafts import SuffixRetriever

# Initialize caches
ngram = NGramCache(n=4)
retriever = SuffixRetriever()

# Warm from previous turns (not synthetic generation)
conversation_history = [...]  # tokens from turns 1-N
ngram.update_sequence(conversation_history)
retriever.extend(conversation_history)

# Now run SJD with warm caches
result = jacobi_decode_sampling(
    model, current_prompt_ids, gen_config,
    K=16,
    use_ngram=True, ngram_n=4,
    use_retrieval=True,
    cache_warmup_tokens=conversation_history,  # ← key parameter
    ...
)
```

### Why Benchmarks Use 1024

The paper results (3.32× speedup) are measured at **steady-state** because:

1. **Real-world usage**: Users don't start fresh every query (multi-turn chat, document continuation)
2. **Fair comparison**: AR-sampling doesn't benefit from caches, so we measure SJD at its best
3. **Cache saturation**: N-gram cache needs ~500-1000 tokens to cover common patterns; suffix retriever needs even more for long-phrase matches

### Summary Table

| Scenario | `--warmup_tokens` | Why |
|----------|-------------------|-----|
| **Paper benchmark** | 1024-2048 | Measure peak steady-state performance |
| **Interactive demo** | 0-128 | Show realistic user experience |
| **Production (1st turn)** | 0 | Cold start is unavoidable |
| **Production (2nd+ turn)** | Use session history | Natural warmup from conversation |
| **Stress test** | 4096+ | Test cache limits / memory usage |

### Code Example: Adaptive Warmup for Demos

```python
def demo_warmup_strategy(session_length: int) -> int:
    """Return appropriate warmup based on session state."""
    if session_length == 0:
        return 0  # First turn: show cold start
    elif session_length < 256:
        return session_length  # Use actual history
    else:
        return 256  # Cap at 256 for demo speed
```

### Key Takeaway

**For demos, use `--warmup_tokens 0` or `--warmup_tokens 128` to show realistic performance.**

The 1024 default in `run_sjd_warm.py` is a **benchmarking tool**, not a demo configuration. It answers the question: *"How fast is SJD when caches are fully warm?"* — which is the steady-state performance users see after a few turns of conversation.

For interactive demos, you want to show the **cold → warm transition** or use actual conversation history for warmup, not synthetic pre-generation.
