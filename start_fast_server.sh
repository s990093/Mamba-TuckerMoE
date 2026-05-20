#!/bin/bash
# Fast Server Startup — Phase 2C Optimized (681 tok/s)
#
# Usage:
#   ./start_fast_server.sh
#   MAMBA_BLOCK_MODE=fused ./start_fast_server.sh
#   CHECKPOINT=/path/to/model.npz ./start_fast_server.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Mamba3-XR Fast Server Startup${NC}"
echo -e "${BLUE}Phase 2C Optimized (681 tok/s)${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
CHECKPOINT="${CHECKPOINT:-checkpoints/latest_sft_cot_model.npz}"
TOKENIZER="${TOKENIZER:-checkpoints/tokenizer}"
PORT="${PORT:-8000}"
MAMBA_BLOCK_MODE="${MAMBA_BLOCK_MODE:-baseline}"

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}✗ Error: Checkpoint not found: $CHECKPOINT${NC}"
    echo ""
    echo "Please provide a checkpoint via:"
    echo "  export CHECKPOINT=/path/to/model.npz"
    echo "  ./start_fast_server.sh"
    exit 1
fi

# Validate tokenizer
if [ ! -d "$TOKENIZER" ]; then
    echo -e "${RED}✗ Error: Tokenizer not found: $TOKENIZER${NC}"
    exit 1
fi

# Validate block mode
case "$MAMBA_BLOCK_MODE" in
    baseline|metal|fused)
        echo -e "${GREEN}✓${NC} Block mode: ${BLUE}$MAMBA_BLOCK_MODE${NC}"
        ;;
    *)
        echo -e "${RED}✗ Invalid MAMBA_BLOCK_MODE: $MAMBA_BLOCK_MODE${NC}"
        echo "Valid options: baseline, metal, fused"
        exit 1
        ;;
esac

# Show configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Checkpoint:     $CHECKPOINT"
echo "  Tokenizer:      $TOKENIZER"
echo "  Block Mode:     $MAMBA_BLOCK_MODE"
echo "  Port:           $PORT"
echo ""

# Performance expectations
case "$MAMBA_BLOCK_MODE" in
    baseline)
        echo -e "${GREEN}Expected Performance:${NC}"
        echo "  Decode:     681 tok/s (11.3× target)"
        echo "  Latency:    1.47 ms/token"
        echo "  Prefill:    7,622 tok/s"
        echo ""
        ;;
    fused)
        echo -e "${BLUE}Expected Performance (Experimental):${NC}"
        echo "  Decode:     672 tok/s"
        echo "  Latency:    1.49 ms/token"
        echo "  Prefill:    2,653 tok/s (slower)"
        echo "  Note:       Prefill performance lower with fused mode"
        echo ""
        ;;
esac

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Error: python3 not found${NC}"
    exit 1
fi

# Check MLX availability
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "import mlx.core" 2>/dev/null || {
    echo -e "${RED}✗ Error: MLX not installed${NC}"
    exit 1
}
echo -e "${GREEN}✓${NC} MLX available"

python3 -c "import transformers" 2>/dev/null || {
    echo -e "${RED}✗ Error: transformers not installed${NC}"
    exit 1
}
echo -e "${GREEN}✓${NC} transformers available"

python3 -c "import fastapi" 2>/dev/null || {
    echo -e "${RED}✗ Error: fastapi not installed${NC}"
    exit 1
}
echo -e "${GREEN}✓${NC} fastapi available"

echo ""
echo -e "${GREEN}✓${NC} All dependencies available"
echo ""

# Set environment variables
export CHECKPOINT
export TOKENIZER
export MAMBA_BLOCK_MODE
export PORT

# Start server
echo -e "${BLUE}═════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting Mamba server...${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Server will be available at:${NC}"
echo -e "${BLUE}  http://localhost:$PORT${NC}"
echo ""
echo -e "${BLUE}WebSocket endpoint:${NC}"
echo -e "${BLUE}  ws://localhost:$PORT/ws${NC}"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop${NC}"
echo ""

# Start server with optimized settings
python3 -m mamba3_mlx.server \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --port "$PORT" \
    2>&1 | while IFS= read -r line; do
        # Highlight key metrics
        if echo "$line" | grep -q "tok/s"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "Error"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "Using.*Mamba blocks"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "ready\|loaded\|started"; then
            echo -e "${GREEN}$line${NC}"
        else
            echo "$line"
        fi
    done
