# -*- coding: utf-8 -*-
"""
Instruction 品質過濾（SFT 用）：依 ChatML user 區塊過濾弱社交／空泛起手式。

用法：
  - off：不過濾
  - drop_weak：丟問候、Tell me something 等保留其餘
  - ins_strict：偏指令／問答型（What/Why/How/Explain…；含 Alpaca ### Instruction）

僅對「第一段 user」為準；複雜對話可自行改規則。
"""
from __future__ import annotations

import re

VALID_INSTRUCTION_FILTER_MODES: tuple[str, ...] = ("off", "drop_weak", "ins_strict")
_VALID_MODES = frozenset(VALID_INSTRUCTION_FILTER_MODES)


def normalize_instruction_filter_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in _VALID_MODES:
        raise ValueError(
            f"instruction filter 須為 {sorted(_VALID_MODES)}，目前為 {mode!r}"
        )
    return m


_FIRST_USER_BLOCK_RE = re.compile(
    r"<\|im_start\|>user\n([\s\S]*?)(?:<\|redacted_im_end\|>|<\|im_end\|>)",
)

# 全 user 區塊幾乎只含社交起手（整段判定）
_WEAK_CHAT_RE = re.compile(
    r"""
    ^\s*(
        hi\b | hello\b | hey\b | yo\b | sup\b | hiya\b | morning\b
        | hi\s+there\b | hey\s+there\b
        | how\s+are\s+you(\s+today|\s+doing)?\b
        | how\s+'?s\s+it\s+going\b | how\s+have\s+you\s+been\b
        | what'?s\s+up\b | wassup\b | whats\s+good\b
        |tell\s+me\s+something\b | say\s+something\b
        |tell\s+me\s+anything\b | say\s+more\b | tell\s+me\s+more\b
        |good\s+(morning|afternoon|evening|night)\b
        | thanks?\b | thank\s+you\b | thx\b | ty\b
        | bye\b | goodbye\b | cya\b | see\s+you\b
        | nice\s+to\s+meet\s+you\b
        | how\s+do\s+you\s+do\b
        | you\s+okay\b | you\s+alright\b
    )\s*[!?.…]*\s*$
    """,
    re.I | re.VERBOSE | re.MULTILINE,
)

_WEAK_TAIL_RE = re.compile(
    r"^\s*(tell|give)\s+me\s+(something|anything|more)\s*[!?.…]*\s*$",
    re.I,
)


def _alpaca_instruction_core(user_block: str) -> str | None:
    if "### Instruction:" not in user_block:
        return None
    _, _, tail = user_block.partition("### Instruction:")
    tail = tail.split("### Input:")[0].split("### Response:")[0].strip()
    return tail if tail else None


# 強指令／問句式開頭（ins_strict）
_INSTRUCTION_OPEN_RE = re.compile(
    r"""
    ^\s*(
        what\b | why\b | how\b | when\b | where\b | which\b | who\b | whose\b
        | explain\b | describe\b | list\b | outline\b | summar(?:ize|ise|y)\b
        | compare\b | contrast\b | define\b | identify\b | analy(?:se|ze)\b
        | discuss\b | evaluate\b | justify\b | predict\b | recommend\b
        | calculate\b | compute\b | solve\b | prove\b | derive\b | simplify\b
        | write\b | compose\b | draft\b | rewrite\b | translate\b | convert\b
        | formulate\b | state\b | name\b | give\b | show\b | find\b | determine\b
        | is\b | are\b | was\b | were\b | can\b | could\b | would\b | should\b
        | does\b | do\b | did\b | has\b | have\b | had\b | will\b
        |(?: could|can)\s+you\s+(?:please\s+)?(?:explain|describe|list|summar(?:ize|ise)|tell|help|expand)
        | please\s+(?:explain|describe|list|summar(?:ize|ise)|write|analyze|answer|provide|show)
        | define\s+whether\b | in\s+detail\b | step\s+by\s+step\b
    )[\s\S]*""",
    re.I | re.VERBOSE | re.DOTALL,
)


def extract_first_chatml_user_block(chatml_example: str) -> str:
    m = _FIRST_USER_BLOCK_RE.search(chatml_example.strip())
    return m.group(1).strip() if m else ""


def instruction_filter_keeps_chatml(chatml_example: str, mode: str) -> bool:
    mode = normalize_instruction_filter_mode(mode)
    if mode == "off":
        return True
    user_blk = extract_first_chatml_user_block(chatml_example)
    if not user_blk.strip():
        return False

    # Alpaca／指令模板：視為優質 supervised，交給 ins_strict／drop_weak 均保留骨架
    acore = _alpaca_instruction_core(user_blk)
    if acore:
        plain = acore.strip()
        weak_on_core = (
            _WEAK_CHAT_RE.match(plain.strip())
            or _WEAK_TAIL_RE.match(_collapse_ws(plain))
        )
        if mode == "drop_weak":
            return not weak_on_core
        # ins_strict：模板內真正有字就留；若連 instruction 區都過短再接 regex
        if len(plain) >= 16:
            return True
        if _INSTRUCTION_OPEN_RE.match(plain):
            return True
        return plain.count(" ") >= 3 or len(plain) >= 8

    # 一般對話／UltraChat raw user
    first_line = user_blk.strip().splitlines()[0].strip()
    collapsed = _collapse_ws(user_blk)

    if mode == "drop_weak":
        if _WEAK_CHAT_RE.match(_collapse_ws(first_line)):
            return False
        if _WEAK_TAIL_RE.match(_collapse_ws(first_line)):
            return False
        if _WEAK_CHAT_RE.match(collapsed.strip()):
            return False
        if _weak_ultra_short_filler(collapsed):
            return False
        return True

    # ins_strict
    if _INSTRUCTION_OPEN_RE.match(user_blk.strip()):
        return True
    if len(user_blk.strip()) >= 120 and ("?" in user_blk or "```" in user_blk):
        return True
    return False


def _collapse_ws(s: str) -> str:
    return " ".join(s.split()).strip()


def _weak_ultra_short_filler(s: str) -> bool:
    if len(s) > 56:
        return False
    w = len(s.split())
    if len(s) <= 26 and w <= 6:
        fillers = frozenset(
            {
                "go",
                "on",
                "ok",
                "okay",
                "cool",
                "nice",
                "really",
                "sure",
                "yeah",
                "yep",
                "wow",
                "interesting",
                "continue",
                "more",
                "else",
                "right",
                "so",
                "and",
                "then",
                "what",
                "hmm",
                "uh",
                "um",
                "lol",
                "haha",
            }
        )
        words = tuple(x.strip(".,?!'\"…").lower() for x in s.split())
        if all(x in fillers for x in words) or (w <= 2 and set(words) <= fillers):
            return True
    vague = ("tell me", "give me", "say something")
    low = s.lower()
    if len(s) < 72 and low.startswith(("tell ", "give ")) and "something" in low:
        return True
    return False
