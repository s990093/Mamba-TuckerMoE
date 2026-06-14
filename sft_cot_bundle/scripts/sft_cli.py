#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil

# 設定與訓練環境一致的 CUDA / Triton 環境變數
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".triton", "cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "torch", "inductor"))

# 尋找 ptxas 路徑供 Triton 使用
if not os.environ.get("TRITON_PTXAS_PATH"):
    ptxas_bin = shutil.which("ptxas")
    if ptxas_bin:
        os.environ["TRITON_PTXAS_PATH"] = ptxas_bin

import argparse
from pathlib import Path

# 確保能夠找到根目錄與 scripts 目錄中的模組
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(SCRIPT_DIR))

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
import random
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

from model import Mamba3LanguageModel
from lima_to_bin import SPECIAL_TOKENS
from stf_cot_sysprompt import DEFAULT_SYS_PROMPTS_BY_BUCKET
from train_sft import (
    _sft_sample_test_reply, 
    _load_pretrain_file, 
    _mamba3_config_from_ckpt, 
    unwrap_model,
    _next_token_from_logits
)

DEFAULT_SYSTEM_PROMPT = DEFAULT_SYS_PROMPTS_BY_BUCKET["daily_conversation"]


STOP_STRINGS = ["</final>", "<|im_end|>"]
# Secondary stop: truncate the reply at </final> or <|im_end|> in post-processing.
# Note: </think> is NOT a stop string because we may inject it ourselves.

# Token IDs resolved once at import time (tokenizer is loaded later per-run;
# these are set after tokenizer load — see _resolve_cot_token_ids).
_COT_THINK_END_ID: int = -1
_COT_FINAL_START_ID: int = -1
_COT_NEWLINE_ID: int = -1


def _resolve_cot_token_ids(tok) -> None:
    """Populate module-level CoT token IDs from the loaded tokenizer."""
    global _COT_THINK_END_ID, _COT_FINAL_START_ID, _COT_NEWLINE_ID

    def _single(s: str) -> int:
        ids = tok.encode(s, add_special_tokens=False)
        return int(ids[0]) if ids else -1

    _COT_THINK_END_ID = _single("</think>")
    _COT_FINAL_START_ID = _single("<final>")
    _COT_NEWLINE_ID = _single("\n")

def _format_cot_prompt(
    user_turn: str,
    system: str | None = None,
    reasoning: bool = False,
) -> str:
    """
    格式化為 ChatML prompt。
    reasoning=True 時在 assistant 後強制注入 <think>，
    迫使 550M 小模型進入 CoT reasoning distribution，避免格式不穩。
    """
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system.strip()}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_turn.strip()}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if reasoning:
        parts.append("<think>\n")
    return "".join(parts)

KAGGLE_BASE_CKPT = os.path.expanduser(
    "~/.cache/kagglehub/datasets/s990093/sft-step-27510-model-only/versions/3"
    "/checkpoint_sft_s53280_model_only.pt"
)

def get_latest_checkpoint(ckpt_path):
    p = Path(ckpt_path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        ckpts = list(p.glob("checkpoint_sft_s*.pt"))
        if ckpts:
            def extract_step(f):
                m = re.search(r'_s(\d+)\.pt$', f.name)
                return int(m.group(1)) if m else -1
            latest = max(ckpts, key=extract_step)
            return str(latest)
    if Path(KAGGLE_BASE_CKPT).is_file():
        print(f"ℹ️  使用 Kaggle base checkpoint: {KAGGLE_BASE_CKPT}")
        return KAGGLE_BASE_CKPT
    return None

def get_random_prompt_from_sft_dataset(dataset_dir, sample_seed=42):
    try:
        ds = load_from_disk(dataset_dir)
        rng = random.Random(sample_seed)
        idx = rng.sample(range(len(ds)), 1)[0]
        text = ds[idx]['text']
        m = re.search(
            r"<\|im_start\|>user\n(.*?)(?=<\|redacted_im_end\|>|<\|im_end\|>)",
            text,
            flags=re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        else:
            return "Explain what is quantum computing in simple terms."
    except Exception as e:
        print(f"[!] 讀取 SFT dataset 失敗: {e}")
        return "Explain what is quantum computing in simple terms."

def parse_args():
    parser = argparse.ArgumentParser(description="Mamba3 SFT 模型 CLI 互動與推理工具")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=str(ROOT_DIR / "output"),
        help="模型權重檔案路徑或目錄。若為目錄將自動讀取最新步數之權重"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default=str(ROOT_DIR / "dataset" / "tokenizer"),
        help="Tokenizer 目錄路徑"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default=None,
        help="單次測試的 Prompt。若不提供將隨機從 SFT dataset 抽樣一題（取第一輪 user）。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT_DIR / "dataset" / "stf_cot_hf"),
        help="隨機抽樣 prompt 用的 dataset 路徑（預設使用 SFT-CoT 資料集）",
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="進入互動對話模式"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt。可填 bucket 名（emotion/self_awareness/daily_conversation/...）或自訂文字。預設 daily_conversation。",
    )
    parser.add_argument(
        "--reasoning", action="store_true",
        help="強制進入 CoT reasoning mode：在 assistant 後注入 <think>，穩定小模型推理格式",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大生成 Token 數量（normal 模式預設 128；reasoning 模式由 max_think_tokens+max_answer_tokens 決定）")
    parser.add_argument("--max_think_tokens", type=int, default=256, help="reasoning 模式下 <think> 區域最大生成 token 數；超限自動 force-inject </think><final>（預設 256）")
    parser.add_argument("--max_answer_tokens", type=int, default=160, help="reasoning 模式下 <final> 區域（回答）最大生成 token 數（預設 160）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成溫度")
    parser.add_argument("--top_p", type=float, default=0.85, help="Top-P 採樣")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K 採樣")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重複懲罰系數")
    parser.add_argument("--sample_seed", type=int, default=42, help="隨機抽樣 prompt 的 seed（與 check_mask.py 一致）")
    parser.add_argument("--compile", action="store_true", help="啟用 torch.compile 進行圖編譯與融合優化（CUDA 建議開啟）")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile 模式")
    parser.add_argument("--compile_warmup_tokens", type=int, default=16, help="啟用 compile 時的 warmup 生成 token 數，用於預先建 cache")
    parser.add_argument("--inductor_cache_dir", type=str, default=os.environ.get("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "torch", "inductor")), help="TorchInductor 編譯快取目錄")
    
    return parser.parse_args()

@torch.inference_mode()
def _stream_sft_reply(
    model,
    tok,
    user_turn,
    device,
    global_step,
    max_new,
    seq_len_cap,
    amp_dtype,
    temperature=0.0,
    top_p=None,
    top_k=50,
    repetition_penalty=1.15,
    sample_seed=0,
    stream_output=True,
    system_prompt=None,
    reasoning=False,
    max_think_tokens: int = 256,
    max_answer_tokens: int = 160,
):
    """
    Generate a reply, streaming tokens to stdout.

    reasoning=True activates a two-phase budget:
      Phase 1 \u2014 think:  up to max_think_tokens model-generated tokens.
                        If </think> is not produced within the budget,
                        the tokens ``</think>\\n<final>\\n`` are **force-injected**
                        (appended without a forward pass) so the format is
                        always well-formed.
      Phase 2 \u2014 answer: up to max_answer_tokens tokens until a stop token.

    reasoning=False uses max_new as a flat budget (original behaviour).
    """
    text0 = _format_cot_prompt(user_turn.strip(), system=system_prompt, reasoning=reasoning)
    ids = tok.encode(text0, add_special_tokens=False)
    if not ids:
        return ""
    n0 = len(ids)

    eos_id = int(tok.eos_token_id) if getattr(tok, "eos_token_id", None) is not None else 2
    stop_ids = {eos_id}
    if len(SPECIAL_TOKENS) > 1:
        try:
            raw = tok.convert_tokens_to_ids(SPECIAL_TOKENS[1])
            _ie = int(raw[0] if isinstance(raw, (list, tuple)) and raw else raw)
            if _ie >= 0:
                stop_ids.add(_ie)
        except (TypeError, ValueError, IndexError):
            pass

    out = list(ids)
    m = unwrap_model(model)
    was_training = model.training
    model.eval()

    gen = torch.Generator()
    gen.manual_seed(int(sample_seed) + int(global_step) * 1_000_003)

    decoded_so_far = ""

    # -----------------------------------------------------------------------
    # Two-phase CoT budget \u2014 active in BOTH reasoning=True (injected <think>)
    # and reasoning=False (model naturally generates <think>).
    #
    # State machine:
    #   PRE_THINK  \u2192 waiting for model to generate <think> (only when reasoning=False)
    #   IN_THINK   \u2192 counting think tokens; force-close at max_think_tokens
    #   IN_ANSWER  \u2192 counting answer tokens; stop at max_answer_tokens
    #   FLAT       \u2192 no <think> seen at all; plain flat budget (max_new)
    # -----------------------------------------------------------------------
    _THINK_START_ID = _COT_THINK_END_ID  # reuse module-level; resolve below
    _THINK_START_ID = -1
    if _COT_THINK_END_ID >= 0:           # derive <think> id from tokenizer
        _ts = tok.encode("<think>", add_special_tokens=False)
        _THINK_START_ID = int(_ts[0]) if len(_ts) == 1 else -1

    _in_think      = reasoning           # True if already inside <think>
    _think_seen    = reasoning           # True once <think> was ever generated/injected
    _think_generated = 0
    _think_forced  = False
    _answer_generated = 0
    # Budget: if reasoning, expand right away; otherwise start flat and extend
    # dynamically when <think> is detected.
    total_budget   = (max_think_tokens + max_answer_tokens) if reasoning else max_new

    def _stream_tokens(token_ids: list[int]) -> None:
        """Append token_ids to out and flush any new printable text."""
        nonlocal decoded_so_far
        for tid in token_ids:
            out.append(tid)
        if stream_output:
            new_text = tok.decode(out[n0:], skip_special_tokens=False)
            if not new_text.endswith("\ufffd"):
                print_text = new_text[len(decoded_so_far):]
                if print_text:
                    print(print_text, end="", flush=True)
                decoded_so_far = new_text

    def _force_close_think() -> None:
        """Inject </think>\n<final>\n and update state."""
        nonlocal _in_think, _think_forced
        nl = _COT_NEWLINE_ID if _COT_NEWLINE_ID >= 0 else \
            tok.encode("\n", add_special_tokens=False)[0]
        force_seq: list[int] = []
        if _COT_THINK_END_ID >= 0:
            force_seq.append(_COT_THINK_END_ID)
        force_seq.append(nl)
        if _COT_FINAL_START_ID >= 0:
            force_seq.append(_COT_FINAL_START_ID)
        force_seq.append(nl)
        _stream_tokens(force_seq)
        _in_think = False
        _think_forced = True

    for _step in range(total_budget):
        # --- Phase 1 budget exhausted: force-inject </think>\n<final>\n ---
        if _in_think and not _think_forced and _think_generated >= max_think_tokens:
            _force_close_think()
            cur_text = tok.decode(out[n0:], skip_special_tokens=False)
            if any(s in cur_text for s in STOP_STRINGS):
                break
            continue

        # --- Phase 2 budget exhausted: stop ---
        if _think_seen and not _in_think and _answer_generated >= max_answer_tokens:
            break

        # --- Normal model forward ---
        use = out[-seq_len_cap:] if len(out) > seq_len_cap else out
        t = torch.tensor([use], device=device, dtype=torch.long)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = m(t, labels=None, step=global_step)
        else:
            logits = m(t, labels=None, step=global_step)

        last = logits[0, -1, :]
        nxt = _next_token_from_logits(
            logits_last=last,
            temperature=temperature,
            top_p=top_p,
            top_k=(50 if top_k is None else top_k),
            repetition_penalty=float(repetition_penalty),
            generated_ids=out,
            gen=gen,
        )

        _stream_tokens([nxt])
        new_text = tok.decode(out[n0:], skip_special_tokens=False)

        # --- Update phase state ---
        if not _think_seen:
            # reasoning=False: detect natural <think> generation
            if _THINK_START_ID >= 0 and nxt == _THINK_START_ID:
                _think_seen = True
                _in_think = True
                # Switch to expanded budget from this step forward
                remaining = (max_think_tokens + max_answer_tokens) - _step - 1
                total_budget = _step + 1 + max(0, remaining)
        elif _in_think:
            _think_generated += 1
            if nxt == _COT_THINK_END_ID:
                _in_think = False
        else:
            # In answer phase: ignore stray </think> tokens that appear after
            # force-injection (model may re-enter think pattern in its context).
            _answer_generated += 1

        if nxt in stop_ids:
            break
        if any(s in new_text for s in STOP_STRINGS):
            break

    if was_training:
        model.train()

    full_reply = tok.decode(out[n0:], skip_special_tokens=False) if out[n0:] else ""
    for ss in STOP_STRINGS:
        idx = full_reply.find(ss)
        if idx != -1:
            full_reply = full_reply[: idx + len(ss)]
            break
    return full_reply


def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    print(f"[*] 正在載入 Tokenizer: {tokenizer_path}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    print(f"[*] 正在載入權重檔: {checkpoint_path}...")
    if not Path(checkpoint_path).exists():
        print(f"[!] 錯誤: 找不到權重檔案 {checkpoint_path}")
        sys.exit(1)
        
    ckpt = _load_pretrain_file(checkpoint_path)
    
    # 預設模型設定 (Fallback)
    fb = {
        "D_MODEL": 768,
        "D_STATE": 64,
        "D_HEAD": 64,
        "EXPAND": 2,
        "NUM_LAYERS": 6,
        "CHUNK_SIZE": 64,
        "KMOE_NUM_EXPERTS": 8,
        "KMOE_TOP_K": 2,
        "KMOE_R1": 4,
        "KMOE_R2": 1024,
        "KMOE_R3": 256,
        "FFN_EXPAND": 6,
        "MIMO_RANK": 4,
        "NUM_KV_HEADS": 4,
    }
    config = _mamba3_config_from_ckpt(ckpt.get("config"), fb)
    
    print("[*] 初始化模型架構...")
    vocab_size = 32007  # 預設詞表大小
    model = Mamba3LanguageModel(config, vocab_size)
    
    print("[*] 載入權重至模型...")
    unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    
    return model, tok

def maybe_compile_model(model, args, device):
    if not args.compile:
        return model
    if device.type != "cuda":
        console.print("[bold yellow][!] 非 CUDA 裝置，跳過 torch.compile。[/]")
        return model

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.inductor_cache_dir
    cache_dir = Path(args.inductor_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[*] torch.compile mode={args.compile_mode}, cache={cache_dir}")
    return torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=True)

def print_rich_full_conversation(full_text):
    tokens_to_highlight = [
        "<|im_start|>", "<|im_end|>", 
        "<think>", "</think>", 
        "<final>", "</final>",
        "[PAD]"
    ]
    # 建立用來切分字串的 pattern
    pattern = "(" + "|".join([re.escape(t) for t in tokens_to_highlight]) + ")"
    
    # 建立格式化文字
    ast_text = Text()
    parts = re.split(pattern, full_text)
    for part in parts:
        if part in tokens_to_highlight:
            ast_text.append(part, style="bold black on bright_yellow")
        else:
            ast_text.append(part, style="white")
            
    # 印出整體對話
    console.print("\n")
    console.print(Panel(ast_text, title="[bold bright_magenta]完整對話紀錄 (含特殊 Token)[/]", border_style="bright_magenta"))

def main():
    args = parse_args()

    # Flat budget for non-reasoning; reasoning uses phase budgets.
    if args.max_new_tokens is None:
        args.max_new_tokens = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    actual_ckpt = get_latest_checkpoint(args.checkpoint)
    if not actual_ckpt:
        console.print(f"[bold red][!] 錯誤: 找不到權重檔案或該目錄下無 checkpoint_sft_s*.pt 檔案: {args.checkpoint}[/]")
        sys.exit(1)

    model, tok = load_model_and_tokenizer(actual_ckpt, args.tokenizer, device)
    _resolve_cot_token_ids(tok)   # populate </think>/<final>/\n token IDs
    model = maybe_compile_model(model, args, device)

    sys_text = args.system
    if sys_text and sys_text in DEFAULT_SYS_PROMPTS_BY_BUCKET:
        sys_text = DEFAULT_SYS_PROMPTS_BY_BUCKET[sys_text]
    elif not sys_text:
        sys_text = DEFAULT_SYSTEM_PROMPT

    mode_label = "[bold yellow]reasoning[/]" if args.reasoning else "normal"
    if args.reasoning:
        budget_info = f"think≤{args.max_think_tokens} + answer≤{args.max_answer_tokens}"
    else:
        budget_info = f"max_new={args.max_new_tokens}"
    console.print(f"[dim]System: {sys_text[:80]}{'…' if len(sys_text) > 80 else ''}[/]")
    console.print(f"[dim]Mode: {mode_label}  {budget_info}[/]")

    if args.compile and args.compile_warmup_tokens > 0:
        console.print(f"[*] 執行 compile warmup（{args.compile_warmup_tokens} tokens）...")
        _ = _stream_sft_reply(
            model=model,
            tok=tok,
            user_turn="Hello",
            device=device,
            global_step=200,
            max_new=args.compile_warmup_tokens,
            seq_len_cap=768,
            amp_dtype=amp_dtype,
            temperature=0.0,
            top_p=None,
            top_k=1,
            repetition_penalty=1.0,
            sample_seed=args.sample_seed,
            stream_output=False,
            system_prompt=sys_text,
        )
    
    if args.interactive:
        # 互動對話模式
        console.print(Panel("進入 Mamba3 SFT 互動對話模式\n(輸入 'exit' 或 'quit' 退出)", style="bold green", expand=False))
        
        while True:
            try:
                user_input = console.input("[bold cyan]User > [/]").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[bold yellow]再見！[/]")
                    break
                
                print("Assistant > ", end="", flush=True)
                response = _stream_sft_reply(
                    model=model,
                    tok=tok,
                    user_turn=user_input,
                    device=device,
                    global_step=200,
                    max_new=args.max_new_tokens,
                    seq_len_cap=768,
                    amp_dtype=amp_dtype,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    sample_seed=args.sample_seed,
                    stream_output=True,
                    system_prompt=sys_text,
                    reasoning=args.reasoning,
                    max_think_tokens=args.max_think_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                )
                print("\n")
                full_conversation = _format_cot_prompt(user_input, system=sys_text, reasoning=args.reasoning) + response
                print_rich_full_conversation(full_conversation)
                
            except KeyboardInterrupt:
                console.print("\n[bold yellow]再見！[/]")
                break
    else:
        # 單次對話模式 (指定 prompt 或隨機從 SFT dataset 抽樣)
        user_prompt = args.prompt
        if not user_prompt:
            console.print(f"[bold green][*] 未提供 prompt，從 dataset 隨機抽樣（第一輪 user）: {args.dataset}[/]")
            user_prompt = get_random_prompt_from_sft_dataset(args.dataset, sample_seed=args.sample_seed)
            
        console.print(f"\n[bold cyan][User][/]:\n{user_prompt}\n")
        console.print("[bold magenta][Assistant][/]:", end=" ")
        response = _stream_sft_reply(
            model=model,
            tok=tok,
            user_turn=user_prompt,
            device=device,
            global_step=200,
            max_new=args.max_new_tokens,
            seq_len_cap=768,
            amp_dtype=amp_dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            sample_seed=args.sample_seed,
            stream_output=True,
            system_prompt=sys_text,
            reasoning=args.reasoning,
            max_think_tokens=args.max_think_tokens,
            max_answer_tokens=args.max_answer_tokens,
        )
        print("\n")
        full_conversation = _format_cot_prompt(user_prompt, system=sys_text, reasoning=args.reasoning) + response
        print_rich_full_conversation(full_conversation)

if __name__ == "__main__":
    main()
