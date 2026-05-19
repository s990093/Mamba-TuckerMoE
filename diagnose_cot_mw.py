#!/usr/bin/env python3
"""
CoT Middleware 诊断工具
快速对比启用/禁用不同功能对输出质量的影响

用法:
  python diagnose_cot_mw.py --test scenario --prompt "who are you?" --runs 3
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent
SERVER_PY = REPO_ROOT / "mamba3_mlx" / "server.py"

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def patch_server(patch_name, original_content):
    """Apply diagnostic patches to server.py"""
    content = original_content

    if patch_name == "disable_injection":
        # 禁用 </final> 注入
        content = content.replace(
            "if did_inject:",
            "if False and did_inject:  # DIAGNOSTIC: disabled injection"
        )
        return content

    elif patch_name == "disable_transform_on_inject":
        # 禁用注入后的 transform_logits 二次调用
        content = content.replace(
            """        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            inj_logits = None
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(
                model, prev_tok, mamba_states, kv_caches, step=decode_step_n)
            _eval_states(mamba_states, kv_caches)
            logits_1d = logits[0]

            # Apply format guard (ban mask + dynamic close-bias) before sampling
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")""",
            """        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            inj_logits = None
            # DIAGNOSTIC: Skip transform for injected logits
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(
                model, prev_tok, mamba_states, kv_caches, step=decode_step_n)
            _eval_states(mamba_states, kv_caches)
            logits_1d = logits[0]

            # Apply format guard (ban mask + dynamic close-bias) before sampling
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")"""
        )
        return content

    elif patch_name == "disable_cot_mw_entirely":
        # 完全禁用 CoT MW
        content = content.replace(
            'enable_cot_mw = msg.get("enable_cot_middleware", True)',
            'enable_cot_mw = False  # DIAGNOSTIC: disabled entire CoT MW'
        )
        return content

    return content

def run_chat(prompt, num_runs=1, scenario="baseline"):
    """Run chat_precise.sh with given prompt and collect output"""
    results = []

    for run_idx in range(num_runs):
        print(f"\n  [Run {run_idx+1}/{num_runs}]", end=" ", file=sys.stderr)
        sys.stderr.flush()

        try:
            result = subprocess.run(
                [str(REPO_ROOT / "chat_precise.sh"), prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout + result.stderr
            results.append({
                "run": run_idx + 1,
                "scenario": scenario,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            })

            # Extract key metrics from output
            if "[benchmark]" in output:
                print(f"✓", file=sys.stderr)
            else:
                print(f"⚠ no benchmark line", file=sys.stderr)

        except subprocess.TimeoutExpired:
            print(f"✗ timeout", file=sys.stderr)
            results.append({
                "run": run_idx + 1,
                "scenario": scenario,
                "error": "timeout"
            })
        except Exception as e:
            print(f"✗ error: {e}", file=sys.stderr)
            results.append({
                "run": run_idx + 1,
                "scenario": scenario,
                "error": str(e)
            })

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose CoT Middleware instability"
    )
    parser.add_argument(
        "--test",
        choices=["baseline", "no-injection", "no-transform", "no-mw", "all"],
        default="all",
        help="Which test scenario to run"
    )
    parser.add_argument(
        "--prompt",
        default="who are you?",
        help="Test prompt"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per scenario"
    )
    parser.add_argument(
        "--keep-patch",
        action="store_true",
        help="Keep patched server.py for manual inspection"
    )

    args = parser.parse_args()

    original_content = read_file(SERVER_PY)

    scenarios = []
    if args.test == "all":
        scenarios = [
            ("baseline", None),
            ("no_injection", "disable_injection"),
            ("no_transform_on_inject", "disable_transform_on_inject"),
            ("no_mw", "disable_cot_mw_entirely"),
        ]
    else:
        scenario_map = {
            "baseline": (None,),
            "no-injection": ("disable_injection",),
            "no-transform": ("disable_transform_on_inject",),
            "no-mw": ("disable_cot_mw_entirely",),
        }
        scenarios = [(args.test, scenario_map[args.test][0])]

    all_results = {}

    try:
        for scenario_name, patch_name in scenarios:
            print(f"\n{'='*70}")
            print(f"📊 Scenario: {scenario_name}")
            print(f"{'='*70}")

            if patch_name:
                print(f"  Applying patch: {patch_name}")
                patched = patch_server(patch_name, original_content)
                write_file(SERVER_PY, patched)
            else:
                write_file(SERVER_PY, original_content)

            print(f"  Prompt: {args.prompt}")
            print(f"  Runs: {args.runs}")
            print(f"  Running...", file=sys.stderr)

            results = run_chat(args.prompt, num_runs=args.runs, scenario=scenario_name)
            all_results[scenario_name] = results

            # Print summary
            successful = sum(1 for r in results if r.get("returncode") == 0)
            print(f"\n  ✓ {successful}/{args.runs} successful")

            if results:
                first_output = results[0].get("stdout", "")[:500]
                print(f"  Sample output (first 500 chars):")
                print("    " + "\n    ".join(first_output.split("\n")[:5]))

    finally:
        # Restore original
        write_file(SERVER_PY, original_content)
        print(f"\n✅ Restored original server.py")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"📈 Summary")
    print(f"{'='*70}")

    for scenario_name, results in all_results.items():
        successful = sum(1 for r in results if r.get("returncode") == 0)
        status = "✓" if successful == len(results) else "⚠"
        print(f"  {status} {scenario_name:25} {successful}/{len(results)} successful")

    print(f"\n💾 Results saved to: diagnose_results.json")
    with open("diagnose_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
