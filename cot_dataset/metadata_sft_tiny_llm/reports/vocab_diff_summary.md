# Tiny-LLM Vocabulary Diff Summary

Result: `PASS`

`arnir0/Tiny-LLM` and the local project tokenizer share the same 32,000 base tokens with the same IDs. The local tokenizer has exactly 7 additional tokens, all matching the project-defined metadata / ChatML / CoT tokens.

| Metric | Value |
| --- | ---: |
| Tiny-LLM vocab length | 32000 |
| Local tokenizer vocab length | 32007 |
| Tokens only in local tokenizer | 7 |
| Tokens only in Tiny-LLM | 0 |
| Shared tokens with different IDs | 0 |

| Added token | ID |
| --- | ---: |
| `<|im_start|>` | 32000 |
| `<|im_end|>` | 32001 |
| `<think>` | 32002 |
| `</think>` | 32003 |
| `<final>` | 32004 |
| `</final>` | 32005 |
| `[PAD]` | 32006 |

The full machine-readable report is in `metadata_sft_tiny_llm/reports/vocab_diff.json`.
