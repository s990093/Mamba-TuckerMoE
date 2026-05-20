"""
Speculative decoding: draft + verify pipeline for faster generation.
"""

import mlx.core as mx
import numpy as np


class SpeculativeGenerator:
    """
    Wrapper for speculative decoding with draft and target models.

    Pipeline:
    1. Draft model generates K tokens sequentially
    2. Target model evaluates all K tokens in parallel (one forward pass)
    3. Accept/reject comparison: sample from target vs draft acceptance threshold
    4. Return accepted tokens and continue
    """

    def __init__(self, target_model, draft_model, tokenizer, k=5):
        """
        Initialize speculative generator.

        Args:
            target_model: Full model (Mamba3LanguageModel)
            draft_model: Lightweight draft model (can be smaller Transformer or Mamba)
            tokenizer: Tokenizer for encoding/decoding
            k: Number of draft tokens to verify per round
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.k = k

    def verify_draft_tokens(self, prefix_ids, draft_ids, draft_probs, states):
        """
        Verify draft tokens using target model.

        Args:
            prefix_ids: Original prefix token IDs
            draft_ids: K draft token IDs [t1, t2, ..., tK]
            draft_probs: K draft probabilities [p1, p2, ..., pK]
            states: Target model states from prefix

        Returns:
            accepted_ids: List of accepted token IDs
            new_states: Updated target model states
            accept_rate: Fraction of tokens accepted
        """
        # Prepare sequence: [prefix + draft tokens]
        full_seq = prefix_ids + draft_ids

        # Run target model on full sequence in one forward pass
        seq_tensor = mx.array([full_seq])
        with mx.no_grad():
            target_logits, new_states, _ = self.target_model(seq_tensor)

        # Extract logits for draft token positions
        # target_logits shape: (1, len(full_seq), vocab_size)
        target_logits_draft = target_logits[0, len(prefix_ids) : len(prefix_ids) + len(draft_ids), :]

        # Convert logits to probabilities
        target_probs = mx.softmax(target_logits_draft, axis=-1)

        # Accept/reject decision for each draft token
        accepted_ids = []
        for i, draft_id in enumerate(draft_ids):
            target_prob = target_probs[i, draft_id].item()
            draft_prob = draft_probs[i]

            # Standard acceptance: accept if random < target_prob / draft_prob
            acceptance_threshold = min(1.0, target_prob / max(draft_prob, 1e-6))

            if np.random.random() < acceptance_threshold:
                accepted_ids.append(draft_id)
            else:
                # Reject: sample from target distribution
                # (In practice, could also reject and resample)
                break

        return accepted_ids, new_states

    def generate_draft(self, last_token_id, draft_states, num_tokens):
        """
        Generate draft tokens using draft model.

        Args:
            last_token_id: Last accepted token ID
            draft_states: Draft model states
            num_tokens: Number of tokens to generate

        Returns:
            draft_ids: Generated token IDs
            draft_probs: Corresponding probabilities
            new_draft_states: Updated draft states
        """
        draft_ids = []
        draft_probs = []
        current_token = last_token_id

        for _ in range(num_tokens):
            token_tensor = mx.array([[current_token]])

            with mx.no_grad():
                logits, draft_states, _ = self.draft_model(token_tensor, states=draft_states)

            # Convert to probabilities and sample
            logits_last = logits[0, 0, :]
            probs = mx.softmax(logits_last, axis=-1)

            # Sample token
            token = mx.random.categorical(probs)
            token_id = token.item()

            # Record
            draft_ids.append(token_id)
            draft_probs.append(probs[token_id].item())

            current_token = token_id

        return draft_ids, draft_probs, draft_states

    def generate(
        self,
        prompt_text,
        max_tokens=256,
        temperature=0.8,
        verbose=False,
    ):
        """
        Generate with speculative decoding.

        Args:
            prompt_text: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for target model
            verbose: Print progress

        Returns:
            Generated text and statistics
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt_text)
        generated_ids = list(prompt_ids)

        # Prefill target model
        prompt_tensor = mx.array([prompt_ids])
        with mx.no_grad():
            _, target_states, _ = self.target_model(prompt_tensor)

        # Initialize draft model states
        draft_states = [None] * len(self.draft_model.layers) if hasattr(
            self.draft_model, 'layers'
        ) else None

        # Generation loop
        total_tokens = 0
        verified_tokens = 0
        draft_tokens = 0

        while total_tokens < max_tokens:
            # Generate K draft tokens
            last_token = generated_ids[-1]
            draft_ids, draft_probs, draft_states = self.generate_draft(
                last_token, draft_states, self.k
            )

            if not draft_ids:
                break

            draft_tokens += len(draft_ids)

            # Verify draft tokens
            accepted_ids, target_states = self.verify_draft_tokens(
                generated_ids, draft_ids, draft_probs, target_states
            )

            if accepted_ids:
                generated_ids.extend(accepted_ids)
                verified_tokens += len(accepted_ids)
                total_tokens += len(accepted_ids)
            else:
                # If no tokens accepted, sample from target
                last_tensor = mx.array([[generated_ids[-1]]])
                with mx.no_grad():
                    logits, target_states, _ = self.target_model(last_tensor, states=target_states)

                logits_last = logits[0, 0, :]
                probs = mx.softmax(logits_last / temperature, axis=-1)
                token = mx.random.categorical(probs).item()

                generated_ids.append(token)
                total_tokens += 1
                verified_tokens += 1

            if verbose and (total_tokens % 50 == 0):
                accept_rate = verified_tokens / (verified_tokens + draft_tokens)
                print(f"Generated {total_tokens} tokens, accept rate: {accept_rate:.1%}")

        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids[len(prompt_ids) :])

        return {
            "text": generated_text,
            "token_ids": generated_ids[len(prompt_ids) :],
            "total_tokens": total_tokens,
            "verified_tokens": verified_tokens,
            "draft_tokens": draft_tokens,
        }
