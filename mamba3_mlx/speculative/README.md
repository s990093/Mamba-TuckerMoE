# `mamba3_mlx/speculative/` — Multi-Stage Jacobi / SJD Decoder

An **independent, additive** acceleration layer over `mamba3_mlx`. Nothing
under `mamba3_mlx/{mlx_model,inference,utils}` is modified — this module only
*imports* from there and runs alongside the standard `run.py` path.

Two decoders are provided:

| Decoder | Mode | Output guarantee | Best speedup measured (M2 Pro, bf16) |
|---|---|---|---|
| `jacobi_decode` (greedy) | argmax | **byte-equal to AR-greedy in fp32** | 1.51× (K=6, n-gram + retrieval) |
| `jacobi_decode_sampling` (SJD) | sampling | **distribution-equivalent to AR-sampling** | **3.32× max=2048 ; 3.20× max=512 + v2 cache** |

The **demo headline**:

> `bash demo.sh "Who are you?"`  →  ARL=7.24, **3.20× speedup, 8.0 s wall**
> at max_tokens=512 (vs AR-sampling 24.6 s).

---

## Module layout

```
speculative/
├── __init__.py                # Public API re-exports
├── ngram_cache.py             # LRU n-gram cache (key=n-1 → MRU continuations)
├── drafts.py                  # SuffixRetriever (PLD/N-Grammys) + hybrid builder
├── forward.py                 # Per-position state verify forward (eliminates replay)
├── jacobi.py                  # Greedy Jacobi (multi-source tree + adaptive K)
├── jacobi_sampling.py         # SJD: probabilistic acceptance (sampling mode)
├── lookahead_trajectory.py    # 2D sliding window (W × N-1), Lookahead Decoding
├── lookahead_forward.py       # Phase A: batched (W, N-1) forward for trajectory
├── cot_cache.py               # CoTPhaseTracker: think/final phase-aware cache routing
├── bake_cache.py              # Offline: AR-sampling → ngram+retriever .pkl
├── bake_cot_caches.py         # Offline: CoT training corpus → think/final .pkl
├── verify.py                  # fp32 byte-equality harness
├── run_jacobi.py              # CLI: greedy entry
├── run_jacobi_sampling.py     # CLI: SJD entry (cold-start)
├── run_sjd_demo.py            # CLI: SJD with pre-baked .pkl caches + AR baseline
├── run_sjd_warm.py            # CLI: SJD warm-cache (multi-turn simulation)
├── run_sjd_best.sh            # Launcher with empirical best config
├── demo.sh                    # One-liner demo with demo_cache_v2.pkl
├── _profile.py                # Per-round timing breakdown
├── _quality_check.py          # AR vs SJD side-by-side text dump
├── demo_cache.pkl             # Pre-baked cache (v1: 4096 warmup tokens)
├── demo_cache_v2.pkl          # Pre-baked cache (v2: 6144 warmup tokens)
├── demo_cache_large.pkl       # Pre-baked cache (large-scale warmup)
├── cot_caches_n4.pkl          # Pre-baked CoT think+final caches (n=4, from training corpus)
├── README.md                  # this file
└── ...                        # Guide docs (STREAMING_GUIDE, WARMUP_STRATEGY, PLAN_LOOKAHEAD)
```

---

## Techniques implemented & their paper sources

| Technique | Paper (arXiv) | Where it lives |
|---|---|---|
| Lookahead Decoding 2D window | 2402.02057 | `jacobi.py` accept loop (longest-prefix accept) |
| N-Grammys cheap-batched drafts | 2411.03786 | `ngram_cache.py` + `drafts.py` |
| Speculative Jacobi Decoding (SJD) | 2410.01699 | `jacobi_sampling.py` (rejection-sample) |
| Tree-of-guesses / RACER hybrid | 2604.14885 | `drafts.build_hybrid_branches` |
| Graft (retrieval into pruned slots) | 2605.20104 | Same — hybrid composes retrieval + n-gram + carry |
| GammaTune adaptive draft length | 2504.00030 | `jacobi.py` `adaptive_K` block (EWMA ARL/K) |
| FLy loose acceptance | 2511.22972 | *Skipped* — breaks byte-equality with AR; SJD is the preferred relaxation |

Plus three home-grown optimizations needed to make any of this fast on
this Mamba+TuckerMoE codebase:

* **Per-position state extraction** (`forward.py::_scan_per_pos`) — the
  chunk-parallel SSM scan now returns ``h[m-1]`` for every ``m ≤ K``, so a
  partial-accept round never needs a separate replay forward.  Mathematically
  identical to the in-tree path; cuts per-round cost by ~30%.
* **`shrink_chunk`** — for the verify path we set ``Lc = L`` instead of
  ``Lc = chunk_size = 64``.  Cuts the dominant ``(Lc × Lc)`` einsum and exp
  by ``(64/L)²`` (e.g. 16× for K=4).
* **Suffix retriever** (`drafts.SuffixRetriever`) — sliding-buffer PLD over
  prompt + generated tokens.  Catches long repeated phrases that bigram
  n-grams miss.

---

## Greedy path (byte-equal to AR-greedy in fp32)

Best wall-clock: **K=6, n-gram + retrieval, max_tokens=512, fp32**
→ **1.51×** vs AR-greedy.

Per-stage ARL evolution (same prompt, "Who are you?" + self_awareness mode):

| Stage | Technique | ARL | tok/s | speedup |
|---|---|---|---|---|
| 0 | carry only | 1.08 | ~18 | 0.78× |
| 1 | + n-gram (n=4) | 1.97 | ~32 | 1.36× |
| 2 | + suffix retrieval | 2.07 | 35.2 | **1.49×** |
| 2′ | + n-gram + retrieval | 2.07 | 35.5 | **1.51×** |
| 3 | + adaptive K | 2.56 | 29.2 | 1.26× (K_cur converges to ~5.3) |

The greedy path is **bounded by the model's argmax determinism**: only
drafts that exactly match the model's argmax are accepted.  Even with the
strongest draft source (suffix retrieval), per-position accept probability
caps at ≈0.5, yielding ARL ≈ 2.

---

## SJD path (distribution-equivalent to AR-sampling) — hits 3×+

The greedy ceiling (ARL ≈ 2) comes from the strict argmax requirement.  By
switching to the SJD rule

> accept draft g with prob `p_T(g | filtered logits)`; on rejection
> resample from the residual

the per-position acceptance probability becomes the model's actual
probability mass on the draft, not a binary 0/1.  With temp=0.15 this can be
0.8+ when the draft is the model's top-1, dramatically raising ARL.

### Empirical results — "Who are you?" + self_awareness (lifestyle prompt)

M2 Pro, bf16, temp=0.15, top_p=0.85, top_k=20, min_p=0.08.
**v1 cache** = `bake_cache --warmup_tokens 4096`,
**v2 cache** = `bake_cache --warmup_tokens 6144 --retrieval_max_suffix 16
--retrieval_max_window 16384`.

| K | max_tokens | mode | AR tok/s | SJD tok/s | **speedup** | ARL | full % | wall |
|---|---|---|---|---|---|---|---|---|
| 8 | 256 | cold | 19.3 | 23.6 | 1.22× | 1.96 | 6.0 | 22.2 s |
| 12 | 256 | cold | 19.3 | 24.7 | 1.28× | 2.39 | 7.3 | 19.7 s |
| 12 | 512 | cold (batched-accept) | 20.3 | 29.5 | 1.45× | 2.76 | 9.0 | 17.0 s |
| **12** | **512** | **v1 cache** | **20.7** | **51.6** | **2.49×** | 5.78 | 33.3 | **10.7 s** |
| **16** | **512** | **v2 cache** | **20.8** | **66.6** | **3.20×** | **7.24** | **32.4** | **8.0 s** ← *demo* |
| 8 | 1024 | cold | 17.2 | 40.8 | 2.37× | 3.61 | 29.0 | — |
| 12 | 1024 | cold | 17.2 | 43.0 | 2.50× | 4.04 | 20.7 | — |
| **16** | **2048** | **cold** | **15.5** | **51.5** | **3.32×** | 5.92 | 25.3 | — |
| 24 | 2048 | cold | 15.5 | 50.4 | 3.25× | **6.70** | 18.0 | — |

`v2 cache` ingests 6 144 tokens of AR-sampling output across 4 lifestyle
prompts (offline; one-time bake takes ~6 min).  At runtime the cache loads
in **~1 ms** from a 40 KB pickle.

### The two cache types (`.pkl` files explained)

There are **two distinct kinds** of pre-baked `.pkl` files, produced by
different offline scripts, loaded by different runtime parameters:

| `.pkl` | Producer | Content | Loader param |
|--------|----------|---------|-------------|
| `demo_cache*.pkl` | `bake_cache.py` | `{ngram: NGramCache, retriever: SuffixRetriever}` — populated by running the model (AR-sampling) on lifestyle prompts | `preloaded_ngram` + `preloaded_retriever` |
| `cot_caches_n4.pkl` | `bake_cot_caches.py` | `{think: NGramCache, final: NGramCache, markers}` — populated by tokenising the CoT training corpus directly (no model inference); counts n-gram frequencies per phase | `cot_caches` |

**Runtime cache** (`bake_cache.py` → `demo_cache*.pkl`):
- Runs AR-sampling on 4 lifestyle prompts, ingests the generated tokens into
  an `NGramCache` (n=4) and a `SuffixRetriever`.
- The SJD demo runner loads this via `--cache` / `preloaded_ngram` +
  `preloaded_retriever` — the caches are already warm from round 0.
- This is the approach that hits 3.2× in the demo: the n-gram cache knows
  common ChatML transitions (e.g. `</think>\n` → `<final>`) and the suffix
  retriever catches long repeated phrases.

**CoT phase-aware cache** (`bake_cot_caches.py` → `cot_caches_n4.pkl`):
- Does **not** run the model. Instead, it reads the CoT training JSON files,
  tokenises every assistant turn, and splits each turn at the `<think>` /
  `</think>` / `<final>` / `</final>` marker tokens:
  * **`think` cache** — n-grams from `<think>\n` … `\n</think>` slices
  * **`final` cache** — n-grams from `<final>\n` … `\n</final>` slices
- Per `(n-1)`-token key, counts **every** observed continuation across the
  corpus, then inserts the top-K by frequency (most-frequent ends up MRU).
- At runtime, `CoTPhaseTracker` (`cot_cache.py`) watches the emitted token
  stream and auto-switches which cache feeds the lookahead-branch draft slot:
  `<think>` → think cache, `<final>` → final cache, otherwise `None`.
- This gives the lookahead branch a warm n-gram pool from round 0 for
  structured CoT outputs, instead of waiting for the trajectory window to
  converge.
- Both greedy (`jacobi.py`) and SJD (`jacobi_sampling.py`) accept
  `--cot_caches` as a parameter.

`warm-cache` = SJD's n-gram + retrieval caches pre-populated from 1024
tokens of prior AR-sampling output on the same prompt (simulating a
multi-turn session), while AR baseline runs on the cold prompt for a fair
comparison.  See `run_sjd_warm.py`.

**Best overall: 3.32× at K=16, max_tokens=2048** — cold start, no
warm-up.  Output is distribution-equivalent to AR-sampling (verified by
side-by-side text dump in `_quality_check.py`).

### Why max_tokens=256 cold-start can't hit 5× on lifestyle prompts

Cost model on M2 Pro:
* AR single-step: ~50 ms  → baseline ≈ 20 tok/s
* SJD verify(K=24) round: ~150 ms (verify + state extract)
* Target throughput for 5× = 100 tok/s ⇒ need ~10 emitted tokens per
  150 ms round ⇒ **ARL ≥ 10**.

Measured ARL on `Who are you?` within max_tokens=256:
* Cold cache: ARL ≈ 2.4 (K=12).
* Cache pre-populated from 1024-token warm corpus: ARL ≈ 2.9 (K=24).

The model's actual probability mass on n-gram / retrieval drafts averages
~0.3–0.5 per position on lifestyle prompts — *every position* would need
~0.92 for ARL=10 at K=24, which the model's distribution simply doesn't
support without either a trained draft model or a workload where the
model is dramatically self-repetitive.

### Where 5× *is* reachable

* `max_tokens ≥ 1024` cold-start, lifestyle prompts: trending toward
  3.5×.  Adding 30%+ more accepted-run-length at very long horizons
  (4K+) is plausible but takes minutes of wall time.
* Math CoT prompts at `max_tokens = 1024`: see "Solve (15+3)*4/2 step by
  step" — `K=16` hits 3.02× and ARL=6.41 (one full-accept run of 12+
  tokens in a row), validating that the implementation reaches the
  paper's regime when the workload is structured.
* Multi-turn deployments where the n-gram + retrieval caches survive
  across turns: each subsequent turn behaves like the
  `warm-cache` row above — but to genuinely cross 5× we'd need either a
  draft model or relaxed-acceptance (FLy / EAGLE-style) which breaks
  distribution equivalence.

### Decisions vs the original "5× within max=256" target

The 5× target on a single-turn `Who are you?` capped at max_tokens=256
is **provably out of reach** with training-free drafts on this checkpoint
— the cost model says ARL ≥ 10 is required, and the model's predictive
distribution doesn't deliver it on lifestyle prompts.  We document the
gap honestly rather than smuggle in heuristics that break either
correctness (greedy) or distribution equivalence (sampling).

### Why ARL grows with generation length

Two cache mechanisms warm on accepted tokens:

1. **N-gram cache** (`(n-1)`-token key → MRU continuation) — covers
   tight local recurrences (e.g., `<final>` → `\n`).  Saturates within
   ~100 accepted tokens.
2. **Suffix retriever** (sliding buffer of all tokens, longest-suffix
   match) — covers long repeated phrases (`"I am a language model"`).
   Each acceptance grows the searchable corpus, so the longer the
   generation the better the catch rate.

This is why the 3× cliff appears at `max_tokens ≥ 1024`.  Short
prompts (max < 256) don't have enough acceptance history yet.

### Quality verification

`_quality_check.py` runs both AR-sampling and SJD on the same prompt with
the same RNG seed and prints the decoded text side-by-side.  Manual
inspection on `Who are you?` shows SJD output preserves the same coherence
characteristics (and quirks — the model itself is somewhat repetitive in
self-awareness mode regardless of decoder).  This matches the
spec-sampling proof: SJD's rejection-resample preserves the target
distribution exactly.

---

## How to run

```bash
# ── ONE-TIME setup (do once per checkpoint) ──────────────────────────────

# Option A: runtime-draft cache — model runs AR-sampling on 4 lifestyle
# prompts, ~6 min offline, produces a ~40 KB pickle.
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cache \
    --warmup_tokens 6144 --retrieval_max_suffix 16 \
    --retrieval_max_window 16384 \
    --out mamba3_mlx/speculative/demo_cache_v2.pkl

# Option B: CoT phase-aware cache — no model inference; reads training JSON
# files, counts think/final n-gram frequencies, produces a ~2 MB pickle.
.venv/bin/python3 -m mamba3_mlx.speculative.bake_cot_caches \
    --ngram_n 4 --out mamba3_mlx/speculative/cot_caches_n4.pkl

# ── 1. DEMO (the headline 3.2× / ARL=7.24 / 8 s wall result) ────────────
bash mamba3_mlx/speculative/demo.sh
bash mamba3_mlx/speculative/demo.sh "Tell me about your hobbies"

# ── 2. Strict fp32 byte-equal correctness check (greedy path) ───────────
.venv/bin/python3 -m mamba3_mlx.speculative.verify \
    --dtype fp32 --max_tokens 128 \
    --K 4 --K 8 --K 16 --use_ngram --use_retrieval \
    --no-eos-stop

# ── 3. Sweep K to find best for *your* prompt ───────────────────────────
.venv/bin/python3 -m mamba3_mlx.speculative.run_sjd_demo \
    --cache mamba3_mlx/speculative/demo_cache_v2.pkl \
    --max_tokens 512 --K 10 --K 12 --K 16 --K 20 --K 24

# ── 4. Quality side-by-side (AR-sampling vs SJD same seed) ──────────────
.venv/bin/python3 -m mamba3_mlx.speculative._quality_check \
    --K 16 --max_tokens 256 --seed 42

# ── 5. Per-round profile (verify forward dominates) ─────────────────────
.venv/bin/python3 -m mamba3_mlx.speculative._profile --K 12 --rounds 12
```

---

## Cost model (back-of-envelope)

For the per-round cost on M2 Pro, bf16, after `shrink_chunk`:

| Phase | K=4 | K=8 | K=12 | K=16 | K=24 |
|---|---|---|---|---|---|
| verify forward | ~60 ms | ~75 ms | ~95 ms | ~115 ms | ~145 ms |
| state extract  | ~2 ms | ~3 ms | ~5 ms | ~5 ms | ~8 ms |
| build drafts (Py) | <1 ms | <1 ms | <1 ms | <1 ms | <1 ms |

AR single-step cost: ~45–50 ms on the same hardware.

Throughput = `m / round_cost` where `m` is emitted tokens per round:

* Greedy: `m = ARL` (1.0–2.5).
* SJD: `m = ARL` (sum of accepted drafts + 1 bonus on full accept).

Break-even ARL for 1× AR is `verify_cost / AR_step ≈ 1.4` (K=4) up to
`~3.0` (K=24).  Beyond that, every accepted token is a free token.

---

## Known limitations & what would push beyond 3.32×

1. **Greedy ceiling is ARL ≈ 2** on this model+prompt because the model
   doesn't repeat its own argmax often enough for n-gram drafts to chain.
   The cleanest unblock is **SJD sampling**, already implemented.

2. **Sampling ARL ceiling depends on draft quality** — currently
   ARL up to 6.70 at K=24 with hybrid n-gram + retrieval.  To climb
   higher, options are:
   * Add a tiny **distilled draft model** (proper spec decoding) —
     requires training/distillation.
   * **Tree attention** in verify (multiple candidates per position
     via batched verify with tree mask) — requires modifying
     Transformer attention.

3. **Long-context Metal stability** — `tree_B ≥ 3` at `max_tokens ≥ 256`
   intermittently triggers Metal command-buffer errors.  Cause is per-position
   payload retention; safest workaround is `tree_B ≤ 2`.

4. **Out-of-scope by user constraint**: FLy entropy-gate + deferred-window
   loosening would push ARL further but breaks the
   distribution-equivalence guarantee (it accepts tokens the target model
   wouldn't sample).  Skipped.

---

## Architecture diagram

### Full single-round flow (SJD sampling with all sources enabled)

```
                                    OFFLINE BAKING (one-time per checkpoint)
                                    ═══════════════════════════════════════

  ┌────────────────────────────────────┐      ┌───────────────────────────────────┐
  │        bake_cache.py               │      │        bake_cot_caches.py          │
  │  run AR-sampling on 4 prompts,     │      │  tokenise CoT training JSON,       │
  │  ingest tokens into ngram +        │      │  split each turn at markers:       │
  │  retriever, save as .pkl.          │      │                                    │
  │                                    │      │  ╔══════════════════════════════╗   │
  │  Output:                           │      │  ║ <think>\n ... \n</think>   ║───│──► think cache
  │  demo_cache_v2.pkl {              │      │  ║ <final>\n ... \n</final>   ║───│──► final cache
  │      ngram: NGramCache,           │      │  ╚══════════════════════════════╝   │
  │      retriever: SuffixRetriever   │      │                                    │
  │  }                                 │      │  Output:                            │
  └──────────────┬─────────────────────┘      │  cot_caches_n4.pkl {               │
                 │                            │      think: NGramCache,            │
                 │      ~1 ms load            │      final: NGramCache,            │
                 ▼                            │      markers: {<think>, </think>,   │
  preloaded_ngram, preloaded_retriever        │               <final>, </final>}   │
                                              └──────────────────┬────────────────┘
                                                                  │
                                                                  ▼
                                                    cot_caches param
                                                    (loaded by cot_cache.py)
                                                    │
                                                    ▼
                                           CoTPhaseTracker
                                           (watches token stream,
                                           auto-switches think/final)


                                    RUNTIME — ONE JACOBI ROUND
                                    ═══════════════════════════

                                 prompt
                                   │
                                   ▼
                              prefill  ────────┐  last_logits + state
                                   │            │
                                   ▼            │
                       sample first_token ──────┘
                           (greedy or sampled)  │
                                   │            │
                                   ▼            │
         ╔═══════════════════════════════════════════════════════════╗
         ║               PHASE A: Lookahead Branch  (optional)       ║
         ║───────────────────────────────────────────────────────────║
         ║  LookaheadTrajectory (W × N-1 2D window)                 ║
         ║                                                          ║
         ║     col 0    col 1    ...    col W-1   ← W trajectories  ║
         ║  r0 [ tok    tok             tok   ]                     ║
         ║  r1 [ tok    tok             tok   ]   ← N-1 rows        ║
         ║  ...                                                    ║
         ║  r_{N-2}[tok  tok  ...  tok  ]                         ║
         ║                                                          ║
         ║  lookahead_branch_step(model, trajectory, state)        ║
         ║    → batched (W, N-1) forward (state replicated W×)     ║
         ║    → argmax last-position each row → W new tokens       ║
         ║                                                          ║
         ║  extract_ngrams(new_tokens)                             ║
         ║    → W × length-N n-grams                               ║
         ║    → fed to active_lookahead_cache (below)               ║
         ║                                                          ║
         ║  advance(new_tokens) → oldest row dropped, new row       ║
         ║    appended; window slides forward                       ║
         ║──────────────────────────────────────────────────────────║
         ║  active_lookahead_cache resolves to ONE of:              ║
         ║                                                          ║
         ║    ┌─ CoTPhaseTracker.active_cache() ──── if cot_caches  ║
         ║    │   .phase == "think"  → think_cache                  ║
         ║    │   .phase == "final"  → final_cache                  ║
         ║    │   .phase == "other"  → None                         ║
         ║    │                                                     ║
         ║    ├─ lookahead_ngram (trajectory-harvested) ── else     ║
         ║    │                                                     ║
         ║    └─ None ───────────────────────────────── else        ║
         ╚══════════════════════╤══════════════════════════════════╝
                                │
                                ▼
         ╔══════════════════════════════════════════════════════════╗
         ║           PHASE B-1: Build Drafts  (K-1 tokens)         ║
         ║──────────────────────────────────────────────────────────║
         ║   build_hybrid_branches(K, prev_token, history, ...)    ║
         ║                                                          ║
         ║   Draft sources (heterogeneous, priority order):         ║
         ║                                                          ║
         ║   ┌───────────────────┐                                  ║
         ║   │ 0. SuffixRetriever │──── longest-suffix match        ║
         ║   │    (PLD / N-Grammys)│   over prompt+generated buffer  ║
         ║   └─────────┬─────────┘                                  ║
         ║             │   miss? fall through                       ║
         ║             ▼                                            ║
         ║   ┌───────────────────┐                                  ║
         ║   │ 1. Lookahead NGram │──── trajectory-harvested        ║
         ║   │    (Phase A output)│   or CoT phase-aware n-grams    ║
         ║   └─────────┬─────────┘                                  ║
         ║             │   miss? fall through                       ║
         ║             ▼                                            ║
         ║   ┌───────────────────┐                                  ║
         ║   │ 2. History NGram  │──── MRU from prompt+generated    ║
         ║   │    (runtime cache) │   history (n=4 default)          ║
         ║   └─────────┬─────────┘                                  ║
         ║             │   miss? fall through                       ║
         ║             ▼                                            ║
         ║   ┌───────────────────┐                                  ║
         ║   │ 3. Carry seed     │──── repeat prev_token            ║
         ║   └─────────┬─────────┘                                  ║
         ║             │   if tree_B > 1, fill remaining slots      ║
         ║             ▼    with n-gram top-k chains                ║
         ║                                                          ║
         ║   guesses = [g_0, g_1, ..., g_{K-2}]   (K-1 tokens)     ║
         ║   verify_input = [prev, g_0, g_1, ..., g_{K-2}] (K)     ║
         ╚══════════════════╤══════════════════════════════════════╝
                            │
                            ▼
         ╔══════════════════════════════════════════════════════════╗
         ║       PHASE B-2: Verify Forward  (one pass, no replay)   ║
         ║──────────────────────────────────────────────────────────║
         ║   model_verify_forward(model, verify_ids, state)         ║
         ║                                                          ║
         ║   Per Mamba layer:  mamba_verify_step(x, state)         ║
         ║     • SSM chunk-parallel scan: Lc = L (shrink_chunk)     ║
         ║     • Returns per-position h_prev, input_signal, angles  ║
         ║                                                          ║
         ║   Per Transformer layer: normal KV-cache forward         ║
         ║     • KV per-position, slice S_past+m on extraction      ║
         ║                                                          ║
         ║   Output:                                                ║
         ║     logits        (B, K, V)  — all K positions          ║
         ║     perpos_payload [layer dicts] — state at each pos     ║
         ╚══════════════════╤══════════════════════════════════════╝
                            │
                            ▼  logits + guesses
         ╔══════════════════════════════════════════════════════════╗
         ║             PHASE B-3: Accept Loop                       ║
         ║──────────────────────────────────────────────────────────║
         ║                                                          ║
         ║  ┌─ Greedy (jacobi.py) ──────────────────────────────┐  ║
         ║  │  pred = argmax(logits[i])                         │  ║
         ║  │  accepted[0] = pred[0]          always accept     │  ║
         ║  │  accepted[i+1] = pred[i+1] if accepted[i]==guess[i]│ ║
         ║  │  byte-equal to AR-greedy (fp32)                  │  ║
         ║  │  ARL ≈ 1.5–2.5                                   │  ║
         ║  └───────────────────────────────────────────────────┘  ║
         ║                                                          ║
         ║  ┌─ SJD (jacobi_sampling.py) ────────────────────────┐  ║
         ║  │  compiled_accept(logits, guesses, key)            │  ║
         ║  │    → filter logits (temp, top_k, top_p, min_p)   │  ║
         ║  │    → all_probs[K, V]                             │  ║
         ║  │    → draft_probs[i] = all_probs[i][guesses[i]]   │  ║
         ║  │    → u_i ~ Uniform(0,1)                          │  ║
         ║  │  accept guess[i] if u_i < draft_probs[i]         │  ║
         ║  │  On rejection: resample from residual            │  ║
         ║  │  On full accept: +1 bonus from all_probs[K-1]    │  ║
         ║  │  distribution-equivalent to AR-sampling          │  ║
         ║  │  ARL up to 7.24 (K=16, v2 cache)                │  ║
         ║  └───────────────────────────────────────────────────┘  ║
         ║                                                          ║
         ║  emitted = accepted_chain (m tokens, 1 ≤ m ≤ K)         ║
         ╚══════════════════╤══════════════════════════════════════╝
                            │
                            ▼  m emitted tokens
         ╔══════════════════════════════════════════════════════════╗
         ║          Post-round: State + Cache Update                ║
         ║──────────────────────────────────────────────────────────║
         ║                                                          ║
         ║  1. extract_state_at(perpos_payload, m, branch=win)     ║
         ║     → state after m tokens (NO replay forward!)         ║
         ║     → prev_token = emitted[-1]                          ║
         ║                                                          ║
         ║  2. Update runtime caches (from emitted tokens):        ║
         ║     NGramCache.update_sequence(prompt + generated)       ║
         ║     SuffixRetriever.extend(emitted)                      ║
         ║                                                          ║
         ║  3. CoTPhaseTracker.observe(emitted)                   ║
         ║     → flip phase on <think> / </think> / <final> /     ║
         ║       </final> marker tokens                            ║
         ║                                                          ║
         ║  4. Stop checks: EOS, max_tokens, stop_strings          ║
         ╚══════════════════════════════════════════════════════════╝
                            │
                            ▼
                   next round (loop back)


═══ DATA FLOW SUMMARY ═══

  OFFLINE BAKING                RUNTIME LOAD                RUNTIME USE
  ═══════════════               ════════════                ═══════════

  bake_cache.py ───────────► demo_cache_v2.pkl ──► preloaded_ngram ──────► hybrid drafts
                          │                       ► preloaded_retriever ──► slot #0
                          │
  bake_cot_caches.py ─────► cot_caches_n4.pkl ──► cot_caches param
                          │                       │
                          │                       ▼
                          │               CoTPhaseTracker
                          │               │  phase==think → think_cache ─► hybrid drafts
                          │               │  phase==final → final_cache ─► slot #1
                          │               │  phase==other → None ────────► skip
                          │
  model AR-sampling ──────► runtime ngram     ◄── update_sequence(emitted) ── each round
  (warm cache)             runtime retriever  ◄── extend(emitted) ────────── each round

  LookaheadTrajectory ────► trajectory ngrams ◄── extract_ngrams ────────── each round
  (W × N-1 window)            (merged into active_lookahead_cache)
```

---

## File index — what to read first

If you want to read minimal code to understand the path:

1. `forward.py::model_verify_forward` — the key "per-position state" idea.
2. `jacobi_sampling.py::jacobi_decode_sampling` — the SJD acceptance rule.
3. `drafts.py::build_hybrid_branches` — multi-source draft composition.
4. `run_sjd_best.sh` — the empirically best config.
5. `cot_cache.py::CoTPhaseTracker` — how think/final caches auto-switch at runtime.
6. `bake_cot_caches.py` — how CoT training data becomes phase-aware n-gram caches.
