#!/usr/bin/env bash
# run.sh — build (if needed) and launch the Mamba desktop pet on macOS.
#
# The pet talks to the REAL model served by `make chat` (chat_demo, port 7860):
# that server hosts both the /eyes page and the chat /ws socket. Start it first:
#
#   make chat          # in one terminal (loads the model, serves :7860)
#   make pet           # in another — the pet connects to :7860 automatically
#
#   PORT=7860 ./run.sh           # override chat port
#   ./run.sh --width 300 --height 360
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-7860}"   # chat_demo / make chat

# 1. The chat server must already be running (model load is heavy — we don't
#    boot it here; the user starts `make chat` themselves).
if ! curl -sf "http://127.0.0.1:${PORT}/eyes" -o /dev/null; then
  echo "[pet] No chat server on :${PORT}." >&2
  echo "[pet] Start it first:  make chat   (then re-run: make pet)" >&2
  exit 1
fi
echo "[pet] chat server up on :${PORT} — connecting"

# 2. Build the Swift wrapper if the binary is stale. The target is split across
#    several .swift files (compiled together). The Info.plist is embedded into
#    the binary (-sectcreate __TEXT __info_plist) so macOS TCC will prompt for
#    microphone access even though this is a plain executable, not a .app.
BIN="$HERE/pet"
PLIST="$HERE/Info.plist"
# Rebuild if any source (.swift or the plist) is newer than the binary.
stale=0
[ -x "$BIN" ] || stale=1
for f in "$HERE"/*.swift "$PLIST"; do [ "$f" -nt "$BIN" ] && stale=1; done
if [ "$stale" = 1 ]; then
  echo "[pet] compiling $(ls "$HERE"/*.swift | wc -l | tr -d ' ') swift files ..."
  swiftc "$HERE"/*.swift -o "$BIN" \
    -framework Cocoa -framework WebKit -framework AVFoundation \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$PLIST"
fi

# 3. Launch
echo "[pet] launching (drag to move · menu-bar 🐍 or on-pet gear to configure)"
exec "$BIN" --url "http://127.0.0.1:${PORT}/eyes?pet=1" "$@"
