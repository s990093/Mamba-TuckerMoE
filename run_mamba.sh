#!/bin/bash
# Mamba3 MLX 推理脚本 - 快速生成文本
#
# 使用方式:
#   ./run_mamba.sh "你的提示词"
#   ./run_mamba.sh --prompt "你的提示词" --max-tokens 512
#   ./run_mamba.sh --help

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Python 环境
if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="python3"
fi

# 默认参数
PROMPT="Hello, how are you?"
MAX_TOKENS=256
TEMPERATURE=0.7
QUANTIZE=""
DTYPE="bf16"
STREAM=""
BENCHMARK=""
VERBOSE=""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║           Mamba3 MLX Inference - Quick Start                  ║
╚════════════════════════════════════════════════════════════════╝

使用方式:
  ./run_mamba.sh "你的提示词"
  ./run_mamba.sh --prompt "你的提示词" --max-tokens 512
  ./run_mamba.sh --stream --prompt "你的提示词"

选项:
  --prompt TEXT           输入提示词 (默认: "Hello, how are you?")
  --max-tokens N          生成的最大 token 数 (默认: 256)
  --temperature T         采样温度 0-1 (默认: 0.7)
  --quantize BITS         量化位数 (4 或 8)
  --dtype TYPE            数据类型 bf16|fp32|fp16 (默认: bf16)
  --stream                流式输出模式
  --benchmark             运行基准测试 (256 tokens)
  -v, --verbose           详细输出
  -h, --help              显示此帮助

示例:
  # 基本使用
  ./run_mamba.sh "Explain quantum computing"

  # 生成更长的文本
  ./run_mamba.sh --prompt "Write a story about" --max-tokens 512

  # 流式输出（实时显示生成过程）
  ./run_mamba.sh --stream --prompt "Hello"

  # 使用 8-bit 量化（更快）
  ./run_mamba.sh --quantize 8 --prompt "Tell me about AI"

  # 基准测试
  ./run_mamba.sh --benchmark --prompt "Test"

  # 详细输出
  ./run_mamba.sh -v --prompt "Verbose mode"

EOF
}

# 解析命令行参数
parse_args() {
    # 如果第一个参数不以 - 开头，把它作为 prompt
    if [ $# -gt 0 ] && [[ ! "$1" =~ ^- ]]; then
        PROMPT="$1"
        shift
    fi

    while [ $# -gt 0 ]; do
        case "$1" in
            --prompt)
                PROMPT="$2"
                shift 2
                ;;
            --max-tokens)
                MAX_TOKENS="$2"
                shift 2
                ;;
            --temperature)
                TEMPERATURE="$2"
                shift 2
                ;;
            --quantize)
                QUANTIZE="--quantize $2"
                shift 2
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --stream)
                STREAM="--stream"
                shift
                ;;
            --benchmark)
                BENCHMARK="--benchmark"
                shift
                ;;
            -v|--verbose)
                VERBOSE="-v"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查依赖
check_dependencies() {
    if ! command -v "$PYTHON" &> /dev/null; then
        echo -e "${RED}❌ Python not found. Install Python 3.10+${NC}"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT/mamba3_mlx" ]; then
        echo -e "${RED}❌ Mamba3 directory not found at $PROJECT_ROOT${NC}"
        exit 1
    fi

    if [ ! -f "$PROJECT_ROOT/mamba3_mlx/inference_api.py" ]; then
        echo -e "${RED}❌ inference_api.py not found${NC}"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT/checkpoints" ]; then
        echo -e "${RED}❌ checkpoints directory not found${NC}"
        echo -e "${YELLOW}   Please download model checkpoint first${NC}"
        exit 1
    fi
}

# 显示配置信息
show_config() {
    if [ -n "$VERBOSE" ]; then
        echo -e "${BLUE}Configuration:${NC}"
        echo "  Prompt:      $PROMPT"
        echo "  Max tokens:  $MAX_TOKENS"
        echo "  Temperature: $TEMPERATURE"
        echo "  Dtype:       $DTYPE"
        [ -n "$QUANTIZE" ] && echo "  Quantize:    $QUANTIZE"
        [ -n "$STREAM" ] && echo "  Mode:        Stream"
        [ -n "$BENCHMARK" ] && echo "  Mode:        Benchmark"
        echo ""
    fi
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"

    # 检查依赖
    check_dependencies

    # 显示配置
    show_config

    # 显示启动信息
    if [ -z "$BENCHMARK" ]; then
        echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║ 🚀 Mamba3 MLX Inference${NC}"
        echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    fi

    # 运行推理
    cd "$PROJECT_ROOT"

    $PYTHON -m mamba3_mlx.inference_api \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --dtype "$DTYPE" \
        $QUANTIZE \
        $STREAM \
        $BENCHMARK \
        $VERBOSE

    # 显示完成信息
    if [ -z "$BENCHMARK" ] && [ -z "$STREAM" ]; then
        echo ""
        echo -e "${GREEN}✓ Done!${NC}"
    fi
}

# 如果脚本被直接执行
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi
