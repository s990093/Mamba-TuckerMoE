"""Profiler server defaults — edit here instead of argparse wiring."""

HOST = "0.0.0.0"
PORT = 8765

# Seconds between WebSocket ticks.
INTERVAL_S = 1.0

# Rolling buffer length (samples).
BUFFER_SECONDS = 60

# powermetrics sample interval (ms). Must match GPU regex parsing cadence.
GPU_SAMPLE_MS = 1000

# Model label shown in snapshot.state when chat_demo publishes telemetry.
DEFAULT_MODEL_NAME = "Mamba3-XR"
