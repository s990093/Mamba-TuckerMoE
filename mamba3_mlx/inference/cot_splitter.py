# -*- coding: utf-8 -*-
"""
CoT stream splitter for Mamba chat server.

Handles incremental token text, emitting typed events:
  ("reasoning", text)   — accumulated <think>…</think> content
  ("token",     text)   — incremental <final>…</final> content
  ("stop",      "")     — end of generation

State machine:
  reasoning=True:   think → between → final → done
  reasoning=False:  head  → final   → done   (or head → done directly)

Robustness rules (model output is often malformed):
  • <think> inside think mode       → silently skip (model re-opened a think block)
  • </think>/<think> inside final   → silently strip (stray CoT tags in answer)
  • <|im_start|> anywhere           → treat as hard stop (model started a new turn)
  • <final>/<final>\n in think      → jump straight to final mode
  • </final>/<|im_end|> in think    → treat preceding text as answer, stop

Cross-chunk safety: _emit_safe holds back the longest possible partial tag prefix
so a tag split across two feed() calls is never missed.
"""

# All terminal/noise tags the splitter watches for
_STOP_TAGS   = ("</final>", "<|im_end|>", "<|im_start|>")
_NOISE_FINAL = ("</think>", "<think>")          # strip silently in final mode
_FINAL_OPEN  = ("<final>\n", "<final>")          # longest first

# Watch list for _emit_safe in each mode (must cover every tag we might split on)
_WATCH_THINK  = ["</think>", "<think>", "<final>", "</final>", "<|im_end|>", "<|im_start|>"]
_WATCH_FINAL  = ["</final>", "<|im_end|>", "<|im_start|>", "</think>", "<think>"]
_WATCH_HEAD   = ["<think>", "<final>", "</final>", "<|im_end|>", "<|im_start|>"]
_WATCH_BETWEEN = ["<final>", "</final>", "<|im_end|>", "<|im_start|>"]


class CotStreamSplitter:
    """
    Feed incremental text; receive a list of typed (kind, content) events.

    Usage:
        splitter = CotStreamSplitter(reasoning=True)
        for chunk in decoder:
            for kind, content in splitter.feed(chunk):
                handle(kind, content)
        for kind, content in splitter.flush():
            handle(kind, content)
    """

    def __init__(self, reasoning: bool = True) -> None:
        self.mode: str = "think" if reasoning else "head"
        self._buf: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Append text to buffer and process; return list of (kind, content)."""
        self._buf += text
        events: list[tuple[str, str]] = []
        self._run(events)
        return events

    def flush(self) -> list[tuple[str, str]]:
        """Emit whatever remains in the buffer (end-of-generation)."""
        events: list[tuple[str, str]] = []
        if self._buf and self.mode in ("final", "head"):
            events.append(("token", self._buf))
        elif self._buf and self.mode == "think":
            events.append(("reasoning", self._buf))
        # "between" and "done": discard silently
        self._buf = ""
        return events

    def force_done(self) -> None:
        """Watchdog / budget override — drop buffer and mark done."""
        self._buf = ""
        self.mode = "done"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self, events: list) -> None:
        while True:
            if   self.mode == "think":   did = self._proc_think(events)
            elif self.mode == "between": did = self._proc_between(events)
            elif self.mode == "final":   did = self._proc_final(events)
            elif self.mode == "head":    did = self._proc_head(events)
            else:                        self._buf = ""; return   # done
            if not did:
                break

    def _stop(self, before: str, after: str, events: list,
              kind: str = "token") -> bool:
        """Emit `before` as `kind`, then emit stop and transition to done."""
        if before:
            events.append((kind, before))
        self._buf = after
        self.mode = "done"
        events.append(("stop", ""))
        return True

    # ── Per-mode processors ────────────────────────────────────────────────────

    def _proc_think(self, events: list) -> bool:
        # Normal exit: </think>
        before, after = self._split_on("</think>")
        if before is not None:
            if before:
                events.append(("reasoning", before))
            self._buf = after
            self.mode = "between"
            return True

        # Model re-opened <think> while already in think mode — skip silently
        before, after = self._split_on("<think>")
        if before is not None:
            if before:
                events.append(("reasoning", before))
            self._buf = after           # discard the redundant <think>
            return True

        # Model jumped straight to <final> without </think>
        for tag in _FINAL_OPEN:
            before, after = self._split_on(tag)
            if before is not None:
                if before.strip():
                    events.append(("reasoning", before))
                self._buf = after
                self.mode = "final"
                return True

        # Hard-stop tags: </final>, <|im_end|>, <|im_start|>
        for tag in _STOP_TAGS:
            before, after = self._split_on(tag)
            if before is not None:
                return self._stop(before.strip(), after, events, kind="token")

        return self._emit_safe("reasoning", _WATCH_THINK, events)

    def _proc_between(self, events: list) -> bool:
        # Normal: <final> (with or without trailing \n)
        for tag in _FINAL_OPEN:
            before, after = self._split_on(tag)
            if before is not None:
                self._buf = after
                self.mode = "final"
                return True

        # Hard-stop tags
        for tag in _STOP_TAGS:
            before, after = self._split_on(tag)
            if before is not None:
                return self._stop(before.strip(), after, events)

        return self._emit_safe("token", _WATCH_BETWEEN, events)

    def _proc_final(self, events: list) -> bool:
        # Normal exit: </final> or <|im_end|> or <|im_start|>
        for tag in _STOP_TAGS:
            before, after = self._split_on(tag)
            if before is not None:
                return self._stop(before, after, events)

        # Strip stray CoT tags that leak into the answer (model artifact)
        for tag in _NOISE_FINAL:
            before, after = self._split_on(tag)
            if before is not None:
                if before:
                    events.append(("token", before))
                self._buf = after       # discard the noise tag, continue
                return True

        return self._emit_safe("token", _WATCH_FINAL, events)

    def _proc_head(self, events: list) -> bool:
        # Hard-stop tags
        for tag in _STOP_TAGS:
            before, after = self._split_on(tag)
            if before is not None:
                return self._stop(before, after, events)

        # Transition to final on <final>
        for tag in _FINAL_OPEN:
            before, after = self._split_on(tag)
            if before is not None:
                if before:
                    events.append(("token", before))
                self._buf = after
                self.mode = "final"
                return True

        # Transition to think on <think>
        before, after = self._split_on("<think>")
        if before is not None:
            if before:
                events.append(("token", before))
            self._buf = after
            self.mode = "think"
            return True

        return self._emit_safe("token", _WATCH_HEAD, events)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _split_on(self, tag: str) -> tuple[str | None, str]:
        """Return (before, after) if tag is found, else (None, '')."""
        idx = self._buf.find(tag)
        if idx == -1:
            return None, ""
        return self._buf[:idx], self._buf[idx + len(tag):]

    def _emit_safe(self, kind: str, watch_tags: list[str], events: list) -> bool:
        """Emit buffer content up to the safe point, holding back partial tag prefixes."""
        if not self._buf:
            return False
        hold = 0
        for tag in watch_tags:
            for plen in range(1, len(tag)):
                if self._buf.endswith(tag[:plen]):
                    hold = max(hold, plen)
        emit_len = len(self._buf) - hold
        if emit_len > 0:
            events.append((kind, self._buf[:emit_len]))
            self._buf = self._buf[emit_len:]
            return True
        return False
