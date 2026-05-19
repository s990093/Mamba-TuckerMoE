"""Low-level CoT / ChatML primitives shared by the inference middleware.

This module holds the *engine-agnostic* building blocks the streaming
generator needs:

* :class:`CotStreamSplitter` — pure-Python state machine that consumes
  decoded text (with ChatML / CoT tags) and emits ``(kind, text)`` events.
  Used by the middleware to know what *mode* the model is in and to demux
  reasoning vs. final output for the UI.
* :class:`FormatGuard` — tokenizer-resolved bias vectors plus a constant-time
  ``apply`` that takes a logits row and returns a biased row.  Implements
  the "ban + soft close-bias" layer of the inference middleware.
* :func:`build_format_guard`, :func:`merge_format_guard_stop_ids`,
  :func:`describe_format_guard` — helpers used at startup.

Higher-level orchestration (reasoning budget, dynamic ramp, multi-stage
``<final>`` injection) lives in :mod:`cot_middleware`.

Design notes
------------
- ``FormatGuard`` keeps **two** kinds of pre-built MLX arrays:
    * ``_ban_mask``        – static ``-inf`` mask, applied while ``mode != "done"``.
    * ``_close_one_hot``   – per-mode one-hot vector at the closing-tag id;
      scaled by a *scalar* bias provided by the caller (the middleware
      computes a ramped value).  This lets us do **one scatter-update +
      one scalar multiply + one add** per decode step, no per-step graph
      compilation, no per-step allocation of vocab-sized buffers.
- We deliberately do **not** ship a full Regex/FSM engine.  For the fixed
  ChatML + CoT skeleton our SFT produces, ban + close-bias + multi-stage
  injection (in the middleware) is sufficient.  Note at the bottom for
  how to upgrade to a BPE-prefix trie when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import mlx.core as mx


# ---------------------------------------------------------------------------
# CoT stream splitter (FSM over decoded text)
# ---------------------------------------------------------------------------
class CotStreamSplitter:
    """Stateful splitter that consumes streamed decoded text from the model
    (which may include ``<think>…</think>`` and ``<final>…</final>`` blocks)
    and emits ``(kind, text)`` events with ``kind`` in
    ``{"reasoning", "final", "stop"}``.

    Buffers partial tag suffixes so a tag spanning two chunks never leaks
    into output. When the prompt already injected ``<think>\\n`` (reasoning
    mode), pass ``start_in_think=True`` so the splitter starts inside the
    think block.

    Modes
    -----
    * ``head``    — before any block (also tolerates a plain answer with no
      tags). If ``start_in_think=False`` we treat this as a loose final.
    * ``think``   — inside ``<think>…``
    * ``between`` — after ``</think>``, waiting for ``<final>`` or ``stop``
    * ``final``   — inside ``<final>…``
    * ``done``    — terminal
    """

    TAGS = ("<think>", "</think>", "<final>", "</final>", "<|im_end|>")
    MAX_TAG_LEN = max(len(t) for t in TAGS)

    def __init__(self, start_in_think: bool = False) -> None:
        self._buf = ""
        self._mode = "think" if start_in_think else "head"
        self._loose_final = not start_in_think

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def done(self) -> bool:
        return self._mode == "done"

    def force_done(self) -> None:
        """Drop the buffer and mark the splitter terminal.  Used by the
        middleware when a hard budget cutoff or other engine-level signal
        ends the turn."""
        self._buf = ""
        self._mode = "done"

    def set_mode(self, mode: str) -> None:
        """Manually transition to ``mode`` (used by the middleware after a
        ``<final>\\n`` cache injection so the splitter stays coherent with
        the model's cache state)."""
        if mode in ("head", "think", "between", "final", "done"):
            self._mode = mode

    def clear_buffer(self) -> None:
        self._buf = ""

    def _safe_tail_cut(self) -> str:
        """Pop & return prefix of ``_buf`` that is safe to flush; keep tail
        that could still be the start of any TAG. Mutates ``_buf``."""
        s = self._buf
        n = len(s)
        keep = 0
        for k in range(1, min(self.MAX_TAG_LEN, n) + 1):
            suf = s[n - k:]
            if any(t.startswith(suf) for t in self.TAGS):
                keep = k
        emit = s[: n - keep]
        self._buf = s[n - keep:]
        return emit

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[tuple[str, str]] = []
        progressed = True
        while progressed:
            progressed = False
            if self._mode == "done":
                self._buf = ""
                break
            if self._mode in ("head", "between"):
                ti = self._buf.find("<think>")
                fi = self._buf.find("<final>")
                ei = self._buf.find("<|im_end|>")
                fe = self._buf.find("</final>")
                hits = [(i, n) for i, n in [(ti, "think"), (fi, "final"), (ei, "stop"), (fe, "stop")] if i >= 0]
                if hits:
                    i, name = min(hits)
                    # Preserve any bridged text before the matched tag:
                    # * in `head` mode this is a tagless answer prefix,
                    # * in `between` mode it's the whitespace / preamble
                    #   the model emitted after `</think>` before EOS / next tag.
                    if i > 0 and (
                        (self._mode == "head" and self._loose_final)
                        or self._mode == "between"
                    ):
                        out.append(("final", self._buf[:i]))
                    if name == "think":
                        self._buf = self._buf[i + len("<think>"):]
                        self._mode = "think"
                    elif name == "final":
                        self._buf = self._buf[i + len("<final>"):]
                        self._mode = "final"
                    else:
                        self._mode = "done"
                        out.append(("stop", ""))
                    progressed = True
                else:
                    if self._mode == "head" and self._loose_final:
                        emit = self._safe_tail_cut()
                        if emit:
                            out.append(("final", emit))
                    elif self._mode == "between":
                        # After </think>, stream may contain whitespace or a short
                        # preamble before <final>; never drop it.
                        emit = self._safe_tail_cut()
                        if emit:
                            out.append(("final", emit))
                    else:
                        self._buf = (
                            self._buf[-(self.MAX_TAG_LEN - 1):]
                            if len(self._buf) > self.MAX_TAG_LEN else self._buf
                        )
            elif self._mode == "think":
                idx_close = self._buf.find("</think>")
                idx_end = self._buf.find("<|im_end|>")
                hits = [(i, n) for i, n in [(idx_close, "close"), (idx_end, "stop")] if i >= 0]
                if hits:
                    i, name = min(hits)
                    if i > 0:
                        out.append(("reasoning", self._buf[:i]))
                    if name == "close":
                        self._buf = self._buf[i + len("</think>"):]
                        self._mode = "between"
                    else:
                        self._mode = "done"
                        out.append(("stop", ""))
                    progressed = True
                else:
                    emit = self._safe_tail_cut()
                    if emit:
                        out.append(("reasoning", emit))
            elif self._mode == "final":
                fe = self._buf.find("</final>")
                ie = self._buf.find("<|im_end|>")
                hits = [(i, "stop") for i in (fe, ie) if i >= 0]
                if hits:
                    i, _ = min(hits)
                    if i > 0:
                        out.append(("final", self._buf[:i]))
                    self._mode = "done"
                    out.append(("stop", ""))
                    progressed = True
                else:
                    emit = self._safe_tail_cut()
                    if emit:
                        out.append(("final", emit))
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining safe text at end-of-stream. Drop a trailing
        partial tag-prefix so fragments like ``</th`` never leak."""
        s = self._buf
        n = len(s)
        keep_drop = 0
        for k in range(1, min(self.MAX_TAG_LEN, n) + 1):
            suf = s[n - k:]
            if any(t.startswith(suf) for t in self.TAGS):
                keep_drop = k
        emit = s[: n - keep_drop]
        out: list[tuple[str, str]] = []
        if emit:
            if self._mode == "think":
                out.append(("reasoning", emit))
            elif self._mode == "final" or (self._mode == "head" and self._loose_final):
                out.append(("final", emit))
            elif self._mode == "between":
                # EOS / max-tokens before <final> — still surface bridged text as the answer.
                out.append(("final", emit))
        self._buf = ""
        self._mode = "done"
        return out


# ---------------------------------------------------------------------------
# Format guard (logit ban + scalable close-bias)
# ---------------------------------------------------------------------------
_BAN_LITERALS: tuple[str, ...] = ("<|im_start|>", "</s>", "<|im_end|>")

# Per splitter-mode, the first-token strings of the *expected closing tag*.
# We try several variants so a tokenizer that doesn't have CoT tags as
# dedicated specials can still find a single-id match.
_MODE_CLOSE_VARIANTS: dict[str, tuple[str, ...]] = {
    "think":   ("</think>", "</",  "</think"),
    "between": ("<final>",  "<final"),
    "final":   ("</final>", "</",  "<|im_end|>"),
}

# Canonical id sequence for the multi-stage `<final>\n` injection.  We never
# bias these as a *first-token* close target (that's `between` mode above);
# the middleware encodes this string and runs it through the model when it
# wants to commit the structural transition deterministically.
_FINAL_INJECT_LITERAL = "<final>\n"


def _safe_encode(tokenizer: Any, text: str) -> list[int]:
    """Encode without prepending specials so we recover the *raw* id sequence
    the model would emit verbatim.  Returns ``[]`` on failure."""
    try:
        out = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            out = tokenizer.encode(text)
        except Exception:
            return []
    except Exception:
        return []
    return [int(t) for t in (out or []) if isinstance(t, int) or hasattr(t, "__index__")]


def _resolve_single_id(tokenizer: Any, text: str) -> int | None:
    """Try ``convert_tokens_to_ids`` first (handles dedicated special tokens),
    fall back to ``encode``.  Returns ``None`` unless the literal resolves to
    exactly one id."""
    try:
        tid = tokenizer.convert_tokens_to_ids(text)
        if isinstance(tid, int) and tid >= 0:
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is None or tid != unk:
                return tid
    except Exception:
        pass
    ids = _safe_encode(tokenizer, text)
    if len(ids) == 1 and ids[0] >= 0:
        return ids[0]
    return None


def encode_final_inject_ids(tokenizer: Any) -> list[int]:
    """Encode ``<final>\\n`` for multi-stage injection.  Returns ``[]`` if
    the tokenizer can't produce a clean id sequence (in which case the
    middleware falls back to bias-only behaviour)."""
    return _safe_encode(tokenizer, _FINAL_INJECT_LITERAL)


@dataclass
class FormatGuardConfig:
    """Knobs exposed to the CLI / WS payload.

    ``close_bias_value`` and ``close_bias_max`` are *informational* on this
    type — the middleware computes a ramped scalar per step and passes it
    to :meth:`FormatGuard.apply` directly.  Both are kept here so the
    startup banner (``describe_format_guard``) can show the configured
    ramp window without depending on the middleware class."""

    enabled: bool = True
    ban_im_start: bool = True
    close_bias_value: float = 4.0
    close_bias_max: float = 16.0
    close_bias_start: int = 0


@dataclass
class FormatGuard:
    cfg: FormatGuardConfig
    vocab_size: int
    ban_ids: tuple[int, ...] = ()
    think_ban_ids: tuple[int, ...] = ()
    final_ban_ids: tuple[int, ...] = ()
    close_first_id_by_mode: dict[str, int] = field(default_factory=dict)

    _ban_mask: mx.array | None = None
    _think_ban_mask: mx.array | None = None
    _final_ban_mask: mx.array | None = None
    _close_one_hot_by_mode: dict[str, mx.array] = field(default_factory=dict)
    _neg_inf: float = -1e9

    def __post_init__(self) -> None:
        if not self.cfg.enabled:
            return
        if self.ban_ids:
            mask = mx.zeros((self.vocab_size,), dtype=mx.float32)
            for tid in self.ban_ids:
                if 0 <= tid < self.vocab_size:
                    mask[tid] = self._neg_inf
            self._ban_mask = mask
            mx.eval(self._ban_mask)
        if self.think_ban_ids:
            mask = mx.zeros((self.vocab_size,), dtype=mx.float32)
            for tid in self.think_ban_ids:
                if 0 <= tid < self.vocab_size:
                    mask[tid] = self._neg_inf
            self._think_ban_mask = mask
            mx.eval(self._think_ban_mask)
        if self.final_ban_ids:
            mask = mx.zeros((self.vocab_size,), dtype=mx.float32)
            for tid in self.final_ban_ids:
                if 0 <= tid < self.vocab_size:
                    mask[tid] = self._neg_inf
            self._final_ban_mask = mask
            mx.eval(self._final_ban_mask)
        for mode, tid in self.close_first_id_by_mode.items():
            if 0 <= tid < self.vocab_size:
                oh = mx.zeros((self.vocab_size,), dtype=mx.float32)
                oh[tid] = 1.0
                self._close_one_hot_by_mode[mode] = oh
        if self._close_one_hot_by_mode:
            mx.eval(*self._close_one_hot_by_mode.values())

    # === Apply path ========================================================
    def apply(
        self,
        logits_row: mx.array,
        *,
        mode: str,
        close_bias_scalar: float = 0.0,
    ) -> mx.array:
        """Return a (potentially biased) logits row of the same shape.

        * Adds the ban mask while ``mode != "done"``.
        * Adds ``close_bias_scalar * one_hot(close_id_for_mode)`` when a
          close-id is known for ``mode`` and ``close_bias_scalar > 0``.
          The caller (middleware) decides the scalar — typically a ramped
          value for ``mode == "think"`` and a static value for ``between``
          / ``final``.
        * Returns the input unchanged when neither knob has any effect.
        """
        if not self.cfg.enabled or mode == "done":
            return logits_row
        out = logits_row
        if self._ban_mask is not None:
            out = out + self._ban_mask.astype(out.dtype)
        if mode == "think" and self._think_ban_mask is not None:
            out = out + self._think_ban_mask.astype(out.dtype)
        if close_bias_scalar and close_bias_scalar > 0:
            oh = self._close_one_hot_by_mode.get(mode)
            if oh is not None:
                out = out + (oh * float(close_bias_scalar)).astype(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Builders / introspection
# ---------------------------------------------------------------------------
def build_format_guard(
    tokenizer: Any,
    *,
    vocab_size: int,
    cfg: FormatGuardConfig,
) -> FormatGuard:
    """Resolve tokenizer ids once, return a ready-to-use :class:`FormatGuard`."""
    # Use backend vocab size if available (handles extended vocabularies like CoT tokens)
    # This fixes an issue where tokenizer.vocab_size may be smaller than the actual vocab
    actual_vocab_size = vocab_size
    if hasattr(tokenizer, "backend_tokenizer") and hasattr(tokenizer.backend_tokenizer, "get_vocab"):
        try:
            backend_vocab = tokenizer.backend_tokenizer.get_vocab()
            if backend_vocab:
                actual_vocab_size = max(actual_vocab_size, max(backend_vocab.values()) + 1)
        except Exception:
            pass

    ban_ids: list[int] = []
    if cfg.enabled and cfg.ban_im_start:
        for lit in _BAN_LITERALS:
            tid = _resolve_single_id(tokenizer, lit)
            if tid is not None and 0 <= tid < actual_vocab_size:
                ban_ids.append(tid)

    # (think_ban: formerly banned <|im_end|> here, now covered by
    #  global ban_ids which blocks </s> + <|im_end|> in all non-done modes.)
    think_ban_ids: list[int] = []

    # Ban </final> inside <final> for the first N tokens so the model
    # can't close with zero answer content.  (<|im_end|> and </s> are
    # already covered by the global ban_ids.)
    final_ban_ids: list[int] = []
    if cfg.enabled:
        for lit in ("</final>",):
            tid = _resolve_single_id(tokenizer, lit)
            if tid is not None and 0 <= tid < actual_vocab_size:
                final_ban_ids.append(tid)

    close_map: dict[str, int] = {}
    if cfg.enabled and (cfg.close_bias_start > 0 or cfg.close_bias_value > 0):
        for mode, variants in _MODE_CLOSE_VARIANTS.items():
            for v in variants:
                tid = _resolve_single_id(tokenizer, v)
                if tid is not None and 0 <= tid < actual_vocab_size:
                    close_map[mode] = tid
                    break

    return FormatGuard(
        cfg=cfg,
        vocab_size=int(actual_vocab_size),  # Use the actual vocab size
        ban_ids=tuple(sorted(set(ban_ids))),
        think_ban_ids=tuple(sorted(set(think_ban_ids))),
        final_ban_ids=tuple(sorted(set(final_ban_ids))),
        close_first_id_by_mode=close_map,
    )


def describe_format_guard(guard: FormatGuard, tokenizer: Any) -> str:
    """Human-readable summary for the startup banner."""
    if not guard.cfg.enabled:
        return "format_guard: disabled"
    bits: list[str] = ["format_guard: enabled"]
    if guard.ban_ids:
        try:
            toks = ", ".join(
                f"{tid} `{tokenizer.convert_ids_to_tokens(tid)}`"
                for tid in guard.ban_ids
            )
        except Exception:
            toks = ", ".join(str(t) for t in guard.ban_ids)
        bits.append(f"ban=[{toks}]")
    else:
        bits.append("ban=(none)")
    if guard.think_ban_ids:
        try:
            toks = ", ".join(
                f"{tid} `{tokenizer.convert_ids_to_tokens(tid)}`"
                for tid in guard.think_ban_ids
            )
        except Exception:
            toks = ", ".join(str(t) for t in guard.think_ban_ids)
        bits.append(f"think_ban=[{toks}]")
    if guard.final_ban_ids:
        try:
            toks = ", ".join(
                f"{tid} `{tokenizer.convert_ids_to_tokens(tid)}`"
                for tid in guard.final_ban_ids
            )
        except Exception:
            toks = ", ".join(str(t) for t in guard.final_ban_ids)
        bits.append(f"final_ban=[{toks}]")
    if guard.close_first_id_by_mode:
        try:
            close_str = ", ".join(
                f"{mode}→{tid} `{tokenizer.convert_ids_to_tokens(tid)}`"
                for mode, tid in guard.close_first_id_by_mode.items()
            )
        except Exception:
            close_str = ", ".join(f"{m}→{t}" for m, t in guard.close_first_id_by_mode.items())
        bits.append(
            f"close_bias(+{guard.cfg.close_bias_value:.1f}→+{guard.cfg.close_bias_max:.1f} "
            f"from t={guard.cfg.close_bias_start})=[{close_str}]"
        )
    else:
        bits.append("close_bias=(off)")
    return " · ".join(bits)


def merge_format_guard_stop_ids(
    tokenizer: Any,
    *,
    existing: Iterable[int] | None,
    enabled: bool,
) -> frozenset[int]:
    """Union of ``existing`` (already-resolved EOS / im_end ids) and any
    single-token closing literals (``</final>``).  Multi-token closings are
    *not* added here — they'd truncate halfway through a tag; the stream
    splitter's ``stop`` event covers those."""
    out: set[int] = set(int(x) for x in (existing or ()))
    if enabled:
        # Get actual vocab size (same fix as in build_format_guard)
        actual_vocab_size = 32000
        if hasattr(tokenizer, "backend_tokenizer") and hasattr(tokenizer.backend_tokenizer, "get_vocab"):
            try:
                backend_vocab = tokenizer.backend_tokenizer.get_vocab()
                if backend_vocab:
                    actual_vocab_size = max(actual_vocab_size, max(backend_vocab.values()) + 1)
            except Exception:
                pass

        for lit in ("</final>",):
            tid = _resolve_single_id(tokenizer, lit)
            if tid is not None and 0 <= tid < actual_vocab_size:
                out.add(tid)
    return frozenset(out)


# ---------------------------------------------------------------------------
# TODO(trie): If a future SFT run leaves the model unable to even *start*
# closing tags with a single id, replace ``close_first_id_by_mode`` with a
# full prefix trie keyed on ``(mode, current_suffix_state)``.  The trie
# nodes hold the set of allowed next ids and a flag for "tag completed →
# switch mode".  We can then mask non-allowed ids while inside a partial
# closing tag, turning the guard into a true (CoT-skeleton-only)
# constrained-decoding layer.
# ---------------------------------------------------------------------------
