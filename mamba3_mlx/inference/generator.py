"""
Generator: Handles prefill/decode pipeline with state management.
"""

import time
import mlx.core as mx
from .sampler import TextSampler, greedy_sample


class AutoregressiveGenerator:
    """
    Manages prefill and decode stages of generation.

    Prefill: Process full prompt and cache Mamba state.
    Decode: Generate tokens one-by-one using state.
    """

    def __init__(self, model, tokenizer, max_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt_text,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        frequency_penalty=0.02,
        repeat_last_n=64,
        greedy=False,
        eos_token_id=None,
        verbose=False,
    ):
        """
        Generate text from a prompt.

        Args:
            prompt_text: Input prompt string
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            min_p: Min-p filtering
            repetition_penalty: Repetition penalty coefficient
            presence_penalty: Presence penalty
            frequency_penalty: Frequency penalty
            repeat_last_n: Window for penalties
            greedy: Use greedy sampling instead of temperature sampling
            eos_token_id: Token ID for end-of-sequence
            verbose: Print timing information

        Returns:
            Dict with:
                - text: Generated text
                - token_ids: Token IDs generated (excluding prompt)
                - prefill_latency: Latency of prefill stage (seconds)
                - decode_latency: Latency of decode stage (seconds)
                - decode_throughput: Tokens per second during decode
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt_text)
        generated_ids = list(prompt_ids)

        # ===== PREFILL STAGE =====
        if verbose:
            print(f"[Prefill] Processing {len(prompt_ids)} tokens...")

        prefill_start = time.time()

        # Convert to tensor
        prompt_tensor = mx.array([prompt_ids])  # (1, prompt_len)

        # Model forward
        logits, states, _ = self.model(prompt_tensor)

        prefill_latency = time.time() - prefill_start

        if verbose:
            print(f"[Prefill] Done in {prefill_latency:.3f}s")

        # Sample first token from prefill output
        next_logits = logits[0, -1, :]  # (vocab_size,)
        if greedy:
            next_token = greedy_sample(next_logits)
        else:
            sampler = TextSampler(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_last_n=repeat_last_n,
            )
            next_token = sampler(next_logits, [])

        generated_ids.append(next_token)

        # ===== DECODE STAGE =====
        if verbose:
            print(f"[Decode] Starting generation...")

        decode_start = time.time()
        decode_tokens_count = 1

        # Continue decoding
        for step in range(1, self.max_tokens):
            # Check stopping conditions
            if eos_token_id is not None and next_token == eos_token_id:
                break

            # Create single-token input
            token_tensor = mx.array([[next_token]])  # (1, 1)

            # Model forward with state
            logits, states, _ = self.model(token_tensor, states=states)

            next_logits = logits[0, 0, :]  # (vocab_size,)

            # Sample next token
            if greedy:
                next_token = greedy_sample(next_logits)
            else:
                sampler = TextSampler(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repeat_last_n=repeat_last_n,
                )
                next_token = sampler(next_logits, generated_ids[-repeat_last_n:])

            generated_ids.append(next_token)
            decode_tokens_count += 1

            if verbose and (step + 1) % 10 == 0:
                elapsed = time.time() - decode_start
                throughput = decode_tokens_count / elapsed
                print(f"[Decode] {step+1} tokens, {throughput:.1f} tok/s")

        decode_latency = time.time() - decode_start
        decode_throughput = decode_tokens_count / decode_latency

        if verbose:
            print(f"[Decode] Done in {decode_latency:.3f}s ({decode_throughput:.1f} tok/s)")

        # Decode tokens to text
        generated_text = self.tokenizer.decode(generated_ids[len(prompt_ids) :])

        return {
            "text": generated_text,
            "token_ids": generated_ids[len(prompt_ids) :],
            "prefill_latency": prefill_latency,
            "decode_latency": decode_latency,
            "decode_throughput": decode_throughput,
            "total_tokens": len(generated_ids) - len(prompt_ids),
        }

    def benchmark(
        self,
        prompt_text,
        num_generate=100,
        temperature=0.8,
        verbose=False,
    ):
        """
        Benchmark generation speed.

        Args:
            prompt_text: Input prompt
            num_generate: Number of tokens to generate
            temperature: Sampling temperature
            verbose: Print progress

        Returns:
            Dict with benchmark results
        """
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text)

        # Prefill
        if verbose:
            print(f"[Benchmark] Prefill with {len(prompt_ids)} tokens...")

        prefill_start = time.time()
        prompt_tensor = mx.array([prompt_ids])
        logits, states, _ = self.model(prompt_tensor)
        prefill_latency = time.time() - prefill_start

        # Decode loop
        if verbose:
            print(f"[Benchmark] Decode {num_generate} tokens...")

        next_token = mx.argmax(logits[0, -1, :]).item()

        decode_start = time.time()
        for step in range(num_generate):
            token_tensor = mx.array([[next_token]])
            with mx.no_grad():
                logits, states, _ = self.model(token_tensor, states=states)
            next_token = mx.argmax(logits[0, 0, :]).item()

            if verbose and (step + 1) % 20 == 0:
                elapsed = time.time() - decode_start
                throughput = (step + 1) / elapsed
                print(f"[Benchmark] {step+1}/{num_generate} tokens ({throughput:.1f} tok/s)")

        decode_latency = time.time() - decode_start
        decode_throughput = num_generate / decode_latency

        return {
            "prefill_latency": prefill_latency,
            "decode_latency": decode_latency,
            "decode_throughput": decode_throughput,
            "total_latency": prefill_latency + decode_latency,
        }
