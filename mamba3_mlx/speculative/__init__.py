"""Jacobi (speculative) decoding — independent add-on for mamba3_mlx.

This package is intentionally **isolated** from the existing mamba3_mlx
runtime:

* No existing file under ``mamba3_mlx/`` is modified.
* It only **imports** the model and helpers it needs
  (``Mamba3LanguageModel``, ``prefill``, etc.).
* Public entry points:
    - :func:`mamba3_mlx.speculative.jacobi.jacobi_decode`
    - ``python -m mamba3_mlx.speculative.run_jacobi``  (CLI runner)
    - ``python -m mamba3_mlx.speculative.verify``      (correctness harness)
"""

from .jacobi import JacobiResult, jacobi_decode  # noqa: F401
from .jacobi_sampling import (  # noqa: F401
    JacobiSamplingResult,
    jacobi_decode_sampling,
)
from .ngram_cache import NGramCache  # noqa: F401

__all__ = [
    "JacobiResult", "jacobi_decode",
    "JacobiSamplingResult", "jacobi_decode_sampling",
    "NGramCache",
]
