"""Inference middleware that wraps the CoT stream splitter, the format
guard, the reasoning-budget watchdog, the dynamic close-bias ramp, and the
optional multi-stage ``<final>`` injection into a single object.

Why a middleware class
----------------------
Before this class was introduced, the four guarantees we wanted at inference
time were scattered across :func:`chat_demo._stream_generate` as a pile of
closures and nonlocal flags (``_force_reasoning_stop``, ``yield_payloads``,
``think_tokens``, ``splitter._mode``, ``_guard_row``).  That made it hard to:

* Reason about what state mutates when.
* Unit-test the inference-time format guarantees in isolation.
* Add new policies (e.g. dynamic ramp, ``<final>`` injection) without
  touching the streaming loop.

The middleware exposes a small, ordered surface that the streaming loop
calls per decode step.  Everything that *can* be done without touching the
model lives here; everything that needs the model (cache-advancing prefill
for the ``<final>`` injection) is delegated through a ``model_apply``
callable provided by the caller — so this module stays MLX-only and free
of FastAPI / WebSocket concerns.

High-level data flow
--------------------
::

    model.logits[..,-1,:]                  → transform_logits(row)  ─┐
        ↓ sample_decode_token                                        │
    sampled tid                            → step(tid)               │
        ↓ split into reasoning / final events                        │
    "just exited <think>"?                 → maybe_inject_final(...) │
        ↓ continuation prefill on `<final>\\n` advances caches        │
    new logits row                         → transform_logits(row) ←─┘

All knobs are funneled through :class:`CotMiddlewareConfig` and
:meth:`CotMiddleware.config_from_args`.  WS payloads can override the master
``enabled`` flag *and* the ``force_final_inject`` flag per turn.

Performance notes
-----------------
* ``transform_logits`` is one ``mx.array`` add (ban mask) + one optional
  scalar-multiplied one-hot add (close bias).  No vocab-sized allocations
  per step; both arrays are built once at construction.
* ``step(tid)`` is pure Python (string decode + splitter feed + counter
  bookkeeping) — orders of magnitude cheaper than the model forward.
* ``maybe_inject_final`` fires at most *once* per turn and runs a tiny
  ``len("<final>\\n")``-token continuation prefill — also negligible vs.
  the rest of the turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator
import time

import mlx.core as mx

from cot_format_fsm import (
    CotStreamSplitter,
    FormatGuard,
    FormatGuardConfig,
    build_format_guard,
    describe_format_guard,
    encode_final_inject_ids,
    merge_format_guard_stop_ids,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CotMiddlewareConfig:
    """Single source of truth for every inference-time guarantee.

    All values are *soft*: if the tokenizer can't resolve a literal or the
    caller disables a knob, the corresponding code path turns into a no-op.
    """

    # Master switch.  When False, transform_logits is identity and the
    # middleware *only* acts as a CoT stream splitter / metric collector.
    enabled: bool = True

    # Logit ban: prevent <|im_start|> mid-answer.
    ban_im_start: bool = True

    # Static close-bias floor for non-think modes (between / final).
    close_bias_value: float = 4.0

    # Maximum close-bias value at the budget tail (dynamic ramp peak).
    # Set equal to close_bias_value to disable ramping.
    close_bias_max: float = 16.0

    # Think-token count after which the ramp starts.  0 = auto-derive from
    # the budget in ``config_from_args``.
    close_bias_start: int = 0

    # Hard cap on tokens inside <think> before we synthesize a stop notice.
    # 0 disables the watchdog entirely.
    reasoning_budget: int = 2000

    # Multi-stage <final> injection: when the splitter just exited <think>,
    # encode "<final>\n" and run it through the model so caches advance and
    # the next sampled token is conditioned on the structural transition.
    force_final_inject: bool = True

    # Minimum tokens the model must produce inside <final> before </final>
    # or <|im_end|> are allowed.  Prevents the model from closing the
    # answer block immediately after injection with zero content.
    final_min_tokens: int = 16

    @classmethod
    def config_from_args(cls, args: Any) -> "CotMiddlewareConfig":
        """Build from ``argparse.Namespace`` (CLI defaults).  Auto-derives
        ``close_bias_start = reasoning_budget // 2`` when the user leaves it
        at 0 so the ramp kicks in over the second half of the budget."""
        reasoning_budget = int(getattr(args, "reasoning_budget", 2000) or 0)
        close_bias_start = int(getattr(args, "close_bias_start", 0) or 0)
        if close_bias_start <= 0 and reasoning_budget > 0:
            close_bias_start = max(1, reasoning_budget // 2)
        return cls(
            enabled=bool(getattr(args, "format_guard", True)),
            ban_im_start=bool(getattr(args, "ban_im_start", True)),
            close_bias_value=float(getattr(args, "close_bias", 4.0)),
            close_bias_max=float(getattr(args, "close_bias_max", 16.0)),
            close_bias_start=close_bias_start,
            reasoning_budget=reasoning_budget,
            force_final_inject=bool(getattr(args, "force_final_inject", True)),
            final_min_tokens=int(getattr(args, "final_min_tokens", 16) or 0),
        )


# ---------------------------------------------------------------------------
# Resident process-level resources (tokenizer-resolved guard + final ids).
# Built once per `_load_model` and shared across every turn.
# ---------------------------------------------------------------------------
@dataclass
class CotMiddlewareDeps:
    """Heavy state that the middleware needs but doesn't *own* — passed in
    by the chat server when constructing a per-turn middleware.  Resolved
    once at process startup (tokenizer ids never change)."""

    tokenizer: Any
    vocab_size: int
    guard: FormatGuard
    final_inject_ids: tuple[int, ...]
    stop_ids: frozenset[int]
    cot_literal_tags: tuple[str, ...] = ("<think>", "</think>", "<final>", "</final>")

    @classmethod
    def build(
        cls,
        *,
        tokenizer: Any,
        vocab_size: int,
        existing_stop_ids: Iterable[int] | None,
        cfg: CotMiddlewareConfig,
    ) -> "CotMiddlewareDeps":
        """Resolve tokenizer ids once.  Returns deps ready for per-turn use."""
        guard = build_format_guard(
            tokenizer,
            vocab_size=vocab_size,
            cfg=FormatGuardConfig(
                enabled=cfg.enabled,
                ban_im_start=cfg.ban_im_start,
                close_bias_value=cfg.close_bias_value,
                close_bias_max=cfg.close_bias_max,
                close_bias_start=cfg.close_bias_start,
            ),
        )
        stop_ids = merge_format_guard_stop_ids(
            tokenizer, existing=existing_stop_ids, enabled=cfg.enabled,
        )
        final_ids = tuple(encode_final_inject_ids(tokenizer)) if cfg.force_final_inject else ()
        return cls(
            tokenizer=tokenizer,
            vocab_size=int(vocab_size),
            guard=guard,
            final_inject_ids=final_ids,
            stop_ids=stop_ids,
        )

    def describe(self) -> str:
        return describe_format_guard(self.guard, self.tokenizer)


# ---------------------------------------------------------------------------
# Per-turn middleware
# ---------------------------------------------------------------------------
# Type alias: callable that runs a multi-token continuation prefill on the
# existing caches.  Signature mirrors the bare model forward:
#   model_apply(x_ids[1, N], caches, seq_pos) -> (logits[1, N, V], caches')
ModelApply = Callable[[mx.array, Any, mx.array], tuple[mx.array, Any]]


class CotMiddleware:
    """Per-turn orchestrator.  Owns the splitter, accumulated reasoning
    text, token counters, and the budget watchdog state for *one* chat
    turn.  Re-instantiated on every call so multi-turn state never leaks."""

    def __init__(
        self,
        *,
        deps: CotMiddlewareDeps,
        cfg: CotMiddlewareConfig,
        reasoning: bool,
        model_apply: ModelApply | None = None,
        router_temp: mx.array | None = None,
    ) -> None:
        self.deps = deps
        self.cfg = cfg
        self.reasoning = bool(reasoning)
        self._model_apply = model_apply
        self._router_temp = router_temp  # unused at present; reserved for
                                          # future router-bias hooks
        self.splitter = CotStreamSplitter(start_in_think=self.reasoning)

        # Per-turn mutable state.
        self._reasoning_acc: list[str] = []
        self._think_tokens = 0
        self._final_tokens = 0
        self._budget_hit = False
        self._final_injected = False
        self._generated_ids: list[int] = []
        self._prev_decoded_text = ""

        # Tokenizer specials cache (set once).
        self._special_ids: set[int] = set(
            getattr(deps.tokenizer, "all_special_ids", []) or []
        )

    # === Read-only state for the caller ====================================
    @property
    def mode(self) -> str:
        return self.splitter.mode

    @property
    def think_tokens(self) -> int:
        return self._think_tokens

    @property
    def reasoning_markdown(self) -> str:
        return "".join(self._reasoning_acc)

    @property
    def final_injected(self) -> bool:
        return self._final_injected

    @property
    def budget_hit(self) -> bool:
        return self._budget_hit

    # === Layer 1: logit transform =========================================
    def current_close_bias(self) -> float:
        """Scalar close-bias value for the current splitter mode.

        * ``think``  → linear ramp from ``close_bias_value`` (at
          ``close_bias_start``) to ``close_bias_max`` (at ``reasoning_budget``).
        * ``between`` / ``final`` → static ``close_bias_value`` (only when
          ``close_bias_start > 0``; otherwise the ramp is "off" globally).
        * other modes → 0.
        """
        if not self.cfg.enabled or self.cfg.close_bias_start <= 0:
            return 0.0
        mode = self.mode
        if mode == "think":
            t = self._think_tokens
            if t < self.cfg.close_bias_start:
                return 0.0
            budget = self.cfg.reasoning_budget
            if budget <= self.cfg.close_bias_start:
                return float(self.cfg.close_bias_max)
            denom = max(1, budget - self.cfg.close_bias_start)
            progress = min(max((t - self.cfg.close_bias_start) / denom, 0.0), 1.0)
            return float(
                self.cfg.close_bias_value
                + progress * (self.cfg.close_bias_max - self.cfg.close_bias_value)
            )
        if mode in ("between", "final"):
            return float(self.cfg.close_bias_value)
        return 0.0

    def transform_logits(self, row: mx.array) -> mx.array:
        """Apply ban mask + dynamic close-bias before sampling.  Always safe
        to call regardless of mode — returns the input unchanged when the
        guard is off."""
        cb = self.current_close_bias()
        if cb > 0 and self.mode == "think" and self._think_tokens == self.cfg.close_bias_start:
            print(f"[mw] close_bias ramp started at think t={self._think_tokens} (bias={cb:.1f}→{self.cfg.close_bias_max:.1f})")
        row = self.deps.guard.apply(
            row, mode=self.mode, close_bias_scalar=cb,
        )
        # Ban </final> and <|im_end|> during the first N tokens of final
        # mode so the model can't close with zero answer content.
        if (
            self.cfg.enabled
            and self.mode == "final"
            and self._final_tokens < self.cfg.final_min_tokens
            and self.deps.guard._final_ban_mask is not None
        ):
            if self._final_tokens == 0:
                print(f"[mw] final_min guard: banning </final> for first {self.cfg.final_min_tokens} tokens")
            row = row + self.deps.guard._final_ban_mask.astype(row.dtype)
        return row

    # === Layer 2: post-sample step ========================================
    def step(self, tid: int, *, n_out: int, elapsed_s_fn: Callable[[], float]) -> Iterator[dict]:
        """Decode ``tid`` → text → splitter events → yield WS-ready dicts.

        Also:
          * Tracks ``think_tokens`` while inside ``<think>``.
          * Fires the reasoning-budget watchdog at the limit.
          * Yields a ``{"__stop__": True}`` sentinel when the splitter or
            watchdog says the turn is over.
        """
        chunk = self._decode_chunk(tid)
        stopped = False
        prev_mode = self.mode
        for kind, payload in self.splitter.feed(chunk):
            if kind == "reasoning":
                self._reasoning_acc.append(payload)
                yield {"type": "reasoning", "markdown": "".join(self._reasoning_acc)}
            elif kind == "final":
                el = elapsed_s_fn()
                yield {
                    "type": "token",
                    "text": payload,
                    "n": n_out,
                    "tok_s": round(n_out / max(el, 1e-9), 1),
                }
            elif kind == "stop":
                stopped = True
        if self.mode != prev_mode:
            print(f"[mw] mode {prev_mode} → {self.mode} at n_out={n_out}")

        # Think-token counter (only meaningful while we're still inside the
        # think block — once the splitter moves to `between` we freeze it).
        if self.reasoning and self.mode == "think":
            self._think_tokens += 1
            # Watchdog: hard stop if we ran out of think budget.
            if (
                self.cfg.reasoning_budget > 0
                and self._think_tokens >= self.cfg.reasoning_budget
                and not self._budget_hit
            ):
                stopped = True
                yield from self._fire_reasoning_budget(n_out=n_out, elapsed_s_fn=elapsed_s_fn)

        # Final-token counter — tracks how many tokens have been generated
        # inside the <final> block so the minimum-content guard works.
        if self.mode == "final":
            self._final_tokens += 1

        if stopped:
            yield {"__stop__": True}

    def _decode_chunk(self, tid: int) -> str:
        """Convert a single sampled id into a streaming-safe string.  CoT
        literal special tokens are surfaced verbatim so the splitter can
        route them; other specials map to ``''`` or ``\\n``."""
        if tid in self._special_ids:
            try:
                tok = str(self.deps.tokenizer.convert_ids_to_tokens(tid))
            except Exception:
                tok = ""
            if tok in self.deps.cot_literal_tags:
                return tok
            tok_n = tok.strip().lower().replace(" ", "")
            if tok_n == "<|im_end|>":
                return "<|im_end|>"
            if tok_n in {"<s>", "</s>", "<eos>", "<|eot_id|>", "<|endoftext|>"}:
                return "\n"
            return ""

        self._generated_ids.append(tid)
        try:
            full_text = self.deps.tokenizer.decode(
                self._generated_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if full_text.startswith(self._prev_decoded_text):
                chunk = full_text[len(self._prev_decoded_text):]
            else:
                chunk = self.deps.tokenizer.decode(
                    [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False,
                )
            self._prev_decoded_text = full_text
        except Exception:
            chunk = f"<{tid}>"
        return chunk

    # === Layer 3: budget watchdog =========================================
    def _fire_reasoning_budget(
        self, *, n_out: int, elapsed_s_fn: Callable[[], float],
    ) -> Iterator[dict]:
        """Emit a synthetic notice + force the splitter into ``done``.  The
        caller still needs to break the decode loop on the ``__stop__``
        sentinel ``step`` yields after this."""
        self._budget_hit = True
        print(f"[mw] ⚠ reasoning budget hit: {self.cfg.reasoning_budget} tokens without </think>")
        note = (
            f"\n\n_⚠ Reasoning budget exhausted ({self.cfg.reasoning_budget} tokens) "
            f"without `</think>` — forcing stop._\n"
        )
        self._reasoning_acc.append(note)
        yield {"type": "reasoning", "markdown": "".join(self._reasoning_acc)}
        final_msg = (
            f"_⚠ Forced stop after {self.cfg.reasoning_budget} reasoning tokens without "
            f"`</think>`. The model never closed its chain-of-thought; try **Direct** mode, "
            f"simplify the prompt, or lower the temperature._"
        )
        el = elapsed_s_fn()
        yield {
            "type": "token",
            "text": final_msg,
            "n": n_out,
            "tok_s": round(n_out / max(el, 1e-9), 1),
        }
        self.splitter.force_done()

    # === Layer 4: multi-stage <final> injection ============================
    def maybe_inject_final(
        self,
        *,
        caches: Any,
        pos: int,
    ) -> tuple[Any, int, mx.array | None, bool, float]:
        """If the splitter just transitioned to ``between`` and config asks
        for it, run a one-shot continuation prefill on ``<final>\\n`` so the
        model's caches commit to the structural transition.  Returns
        ``(caches, pos, last_logits_row, did_inject, ms_spent)`` — when
        ``did_inject=True`` the caller should sample its *next* token from
        ``last_logits_row`` (instead of doing another single-step decode).

        At most once per turn: the second call is a no-op."""
        if (
            not self.cfg.enabled
            or not self.cfg.force_final_inject
            or self._final_injected
            or self.mode != "between"
            or not self.deps.final_inject_ids
            or self._model_apply is None
        ):
            return caches, pos, None, False, 0.0

        self._final_injected = True
        ids = list(self.deps.final_inject_ids)
        t0 = time.perf_counter()
        x = mx.array([ids], dtype=mx.int32)
        seq_pos = mx.array(int(pos), dtype=mx.int32)
        try:
            logits, caches = self._model_apply(x, caches, seq_pos)
            mx.eval(logits, caches)
        except Exception as exc:
            print(f"[mw] WARN: <final> injection failed → fallback. ({exc})")
            self._final_injected = False
            return caches, pos, None, False, 0.0
        ms = (time.perf_counter() - t0) * 1000.0
        new_pos = int(pos + len(ids))
        last_row = logits[0, -1, :]
        print(f"[mw] ✓ <final> injected into cache ({len(ids)} ids, {ms:.1f}ms, pos {pos}→{new_pos})")

        # Push the same text through the splitter so its mode advances from
        # `between` → `final` and its buffer reflects the trailing "\n".
        self.splitter.feed("<final>\n")
        # If the splitter has somehow ended up `done` (e.g. saw <|im_end|>
        # earlier in the buffer) we don't override it; otherwise nudge it.
        if self.splitter.mode == "between":
            self.splitter.set_mode("final")
            self.splitter.clear_buffer()
        return caches, new_pos, last_row, True, ms

    # === Layer 5: stop-id routing =========================================
    def should_break(self, tid: int) -> bool:
        """``True`` when the sampled id is in the EOS-ish set the loop
        should treat as turn-end (covers ChatML ``<|im_end|>``, EOS, and any
        single-token ``</final>`` we managed to resolve)."""
        return bool(self.deps.stop_ids and tid in self.deps.stop_ids)

    # === Layer 6: end-of-stream flush =====================================
    def flush(self, *, n_out: int, elapsed_s_fn: Callable[[], float]) -> Iterator[dict]:
        """Emit any text left in the splitter buffer after EOS / max-tokens."""
        for kind, payload in self.splitter.flush():
            if kind == "reasoning":
                self._reasoning_acc.append(payload)
                yield {"type": "reasoning", "markdown": "".join(self._reasoning_acc)}
            else:
                el = elapsed_s_fn()
                yield {
                    "type": "token",
                    "text": payload,
                    "n": n_out,
                    "tok_s": round(n_out / max(el, 1e-9), 1),
                }

    # === Debug =============================================================
    def health_report(self) -> dict[str, Any]:
        """Snapshot for the post-turn rich console dump.  Compact dict of
        what the middleware actually did this turn."""
        return {
            "enabled": bool(self.cfg.enabled),
            "splitter_final_mode": self.mode,
            "think_tokens": self._think_tokens,
            "reasoning_budget": self.cfg.reasoning_budget,
            "budget_hit": self._budget_hit,
            "final_injected": self._final_injected,
            "final_tokens": self._final_tokens,
            "final_min_tokens": self.cfg.final_min_tokens,
            "close_bias_window": (self.cfg.close_bias_start, self.cfg.reasoning_budget),
            "close_bias_range": (self.cfg.close_bias_value, self.cfg.close_bias_max),
        }


# ---------------------------------------------------------------------------
# Sugar so call sites stay one-liner clean.
# ---------------------------------------------------------------------------
def render_health_line(report: dict[str, Any]) -> str:
    """Format a `health_report` dict as a single rich-friendly line."""
    if not report or not report.get("enabled"):
        return "middleware: disabled"
    bits: list[str] = []
    bits.append(f"mode={report['splitter_final_mode']}")
    bits.append(f"think={report['think_tokens']}/{report['reasoning_budget']}")
    if report.get("budget_hit"):
        bits.append("[yellow]budget_hit[/]")
    if report.get("final_injected"):
        ft = report.get("final_tokens", 0)
        fm = report.get("final_min_tokens", 0)
        bits.append(f"[green]final_injected[/]  final_tok={ft}(min={fm})")
    cs, cb = report.get("close_bias_window", (0, 0))
    cv, cm = report.get("close_bias_range", (0.0, 0.0))
    bits.append(f"close_bias=({cv:.1f}→{cm:.1f} from t={cs} to t={cb})")
    return "  ".join(bits)
