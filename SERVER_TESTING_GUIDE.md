# Server Testing Guide — CoT Generation Quality Verification

## Starting the Server

```bash
# Terminal 1: Start the server
python -m mamba3_mlx.server --port 8000

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Loading tokenizer ...
# INFO:     Loading model weights from ...
# INFO:     Warming up ...
# INFO:     Format guard: enabled, ban=[...], close_bias=[...]
```

---

## Test 1: API Status Check

```bash
# Terminal 2: Verify server is ready
curl -s http://localhost:8000/api/status | jq .
```

**Expected response:**
```json
{
  "ready": true,
  "loading": false,
  "model_config": {
    "d_model": 1152,
    "num_layers": 24,
    "kmoe_num_experts": 8,
    "kmoe_top_k": 2,
    "quantize": 0,
    "dtype": "bf16",
    "max_new_tokens": 2048
  },
  "load_timings": {
    "total_ms": 5000,
    "checkpoint": "latest_sft_cot_model.npz"
  }
}
```

---

## Test 2: Format Guard Status

```bash
# Check the middleware initialization
curl -s http://localhost:8000/api/status | jq '.mw_deps'
```

**Expected output should show:**
```
Format guard: enabled
  Ban IDs: [2, 32000, 32001] (<|im_end|>, </s>, <|im_start|>)
  Close bias targets: {
    think: 32003 `</think>`,
    between: 32004 `<final>`,
    final: 32005 `</final>`
  }
```

**⚠️ If you see ID 829 instead of 32003-32005, the vocab_size fix did not apply.**

---

## Test 3: WebSocket Chat with Self-Awareness System Prompt

### Setup
```bash
# In a test script (Python)
import asyncio
import json
import websockets

async def test_chat():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Send chat request with self_awareness mode
        msg = {
            "action": "chat",
            "prompt": "Who are you?",
            "category_key": "self_awareness",
            "reasoning": True,
            "max_tokens": 300
        }
        
        reasoning_blocks = []
        final_blocks = []
        tokens_received = 0
        
        await websocket.send(json.dumps(msg))
        
        async for response_str in websocket:
            response = json.loads(response_str)
            msg_type = response.get("type")
            
            if msg_type == "reasoning":
                print(f"[REASONING] {response['markdown'][:100]}...")
                reasoning_blocks.append(response['markdown'])
                
            elif msg_type == "assistant_split":
                print("[FSM TRANSITION] Reasoning → Final Answer")
                
            elif msg_type == "token":
                tokens_received += 1
                if tokens_received <= 5:
                    print(f"[TOKEN {tokens_received}] {response['text']}")
                    
            elif msg_type == "done":
                final_blocks.append(response.get('text', ''))
                print(f"\n[DONE] Generated {response['total_tokens']} tokens in {response['total_ms']}ms")
                break
            
            elif msg_type == "error":
                print(f"[ERROR] {response['message']}")
                break
        
        # Quality verification
        print("\n" + "=" * 80)
        print("QUALITY VERIFICATION")
        print("=" * 80)
        
        has_reasoning = len(reasoning_blocks) > 0 and len(''.join(reasoning_blocks)) > 10
        has_final = tokens_received > 20
        
        print(f"✓ Has reasoning block: {has_reasoning}")
        print(f"✓ Has final answer: {has_final}")
        print(f"✓ Tokens generated: {tokens_received}")
        
        if has_reasoning and has_final:
            print("\n✅ PASS: CoT generation working correctly")
            print("   - Reasoning block properly identified")
            print("   - Final answer properly generated")
            print("   - FSM state transitions correct")
        else:
            print("\n⚠️ FAIL: CoT generation incomplete")
            if not has_reasoning:
                print("   - No reasoning block detected")
            if not has_final:
                print("   - No final answer generated")

asyncio.run(test_chat())
```

### Expected Output

```
[REASONING] <think>
Let me think about my identity and capabilities...
I'm a state space model (Mamba) combined with Tucker decomposition...
</think>...

[FSM TRANSITION] Reasoning → Final Answer

[TOKEN 1] I
[TOKEN 2] am
[TOKEN 3] Mamba
[TOKEN 4] ,
[TOKEN 5] a

...

[DONE] Generated 180 tokens in 2450ms

════════════════════════════════════════════════════════════════════════════════
QUALITY VERIFICATION
════════════════════════════════════════════════════════════════════════════════
✓ Has reasoning block: True
✓ Has final answer: True
✓ Tokens generated: 180

✅ PASS: CoT generation working correctly
   - Reasoning block properly identified
   - Final answer properly generated
   - FSM state transitions correct
```

---

## Test 4: Compare Before/After Fix

If you have an older version without the vocab_size fix, you'd see:

### ❌ BEFORE FIX (with vocab_size=32000)
```
Format guard: enabled
  Ban IDs: [2, 32000, 32001]
  Close bias targets: {
    think: 829 `</`,      ❌ WRONG!
    between: 829 `</`,    ❌ WRONG!
    final: 829 `</`       ❌ WRONG!
  }

[REASONING] (empty or very short)
[NO FSM TRANSITION] ❌
[TOKEN 1] <
[TOKEN 2] /
[TOKEN 3] (repeated generation of `</`)

⚠️ FAIL: Close bias applied to wrong tokens, model stuck
```

### ✅ AFTER FIX (with vocab_size=32007)
```
Format guard: enabled
  Ban IDs: [2, 32000, 32001]
  Close bias targets: {
    think: 32003 `</think>`,    ✓ CORRECT
    between: 32004 `<final>`,   ✓ CORRECT
    final: 32005 `</final>`     ✓ CORRECT
  }

[REASONING] Let me think about my identity...
[FSM TRANSITION] Reasoning → Final Answer ✓
[TOKEN ...] I am Mamba...

✅ PASS: All targets correct, generation quality restored
```

---

## Quick cURL WebSocket Test

For a quick test without Python WebSocket library:

```bash
# Using websocat (if installed)
echo '{"action":"chat","prompt":"Who are you?","category_key":"self_awareness","max_tokens":300}' | \
  websocat ws://localhost:8000/ws | head -20

# Or test the HTTP status endpoint
curl -s http://localhost:8000/api/demo-config | jq '.system_prompts'
```

---

## Troubleshooting

### Issue: Server shows `Format guard: disabled`
- The CotMiddlewareDeps.build() failed silently
- Check tokenizer path is correct
- Verify vocab_size detection is working

### Issue: Close bias targets show ID 829
- The vocab_size=32000 issue is still present
- Verify server.py has the fix (lines 209-217)
- Check git status: `git diff HEAD~1 mamba3_mlx/server.py`

### Issue: FSM doesn't transition to final mode
- Model is being biased toward `</` instead of `</think>`
- Check format guard close_bias values in status endpoint
- Verify inference is using CotMiddleware (check logs)

### Issue: Reasoning block is empty
- Model might not be entering think mode at all
- Verify prompt includes `<think>\n` injection
- Check reasoning=True in request payload

---

## Performance Expectations

- **Prefill** (system + prompt): 2-5 seconds
- **First reasoning token TTFT**: 200-500ms
- **Decode throughput**: 40-100 tokens/second
- **Total "Who are you?" response**: 5-10 seconds

---

## Success Criteria

✅ **Generation quality is restored when:**

1. Format guard shows correct CoT token IDs (32003-32005)
2. FSM properly transitions between reasoning/final modes
3. Output includes both reasoning and answer blocks
4. No stuck loops on `</` token
5. Consistent and coherent generation

---

**Status:** Ready to test on your Mac with Apple Silicon

Last updated: 2026-05-19
