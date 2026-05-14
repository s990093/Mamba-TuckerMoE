#!/usr/bin/env bash
# 從本腳本所在目錄的「上一層」（hybrid-mamba-15min）啟動靜態伺服器，讓
# presentation.html 內的 ../assets/… 圖檔能正確載入（根目錄若只設在
# presentation/ 則 /assets/… 會 404）。
# 預設埠 18886；要換埠：PORT=9000 ./serve_presentation.sh
#
# 請用「子行程」執行：  ./serve_presentation.sh  或  bash serve_presentation.sh
# 勿用：  source serve_presentation.sh  或  . ./serve_presentation.sh
#
# 路徑：若已 cd 到 presentation/，用 ./serve_presentation.sh
# 若在 repo 根目錄：  bash paper/hybrid-mamba-15min/presentation/serve_presentation.sh

if [[ -n "${BASH_VERSION:-}" && "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "錯誤：偵測到以 source / . 載入本腳本。" >&2
  echo "請改為：  ./serve_presentation.sh   或   bash \"$(basename "${BASH_SOURCE[0]}")\"" >&2
  return 2>/dev/null || exit 1
fi

set -eu
cd "$(dirname "$0")/.."
PORT="${PORT:-18886}"

echo "Serving from: $(pwd)  (含 presentation/ 與 assets/)"
echo "Open one of:"
echo "  http://127.0.0.1:${PORT}/presentation/"
echo "  http://127.0.0.1:${PORT}/presentation/presentation.html"
echo "(Stop with Ctrl+C)"
echo ""
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
