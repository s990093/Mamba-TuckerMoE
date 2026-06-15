from PIL import Image
import os

# ── 參數 ──────────────────────────────────────────
SRC    = os.path.join(os.path.dirname(__file__), "raw.png")
OUT    = os.path.dirname(__file__)   # 輸出到同一個資料夾
COLS   = 4
ROWS   = 4
OFF_X  = 0    # 左邊緣偏移（px）
OFF_Y  = 0    # 上邊緣偏移（px）
GAP_X  = 0    # 格子間水平間距
GAP_Y  = 0    # 格子間垂直間距
# CELL_W / CELL_H 為 None → 自動平均分割
CELL_W = None
CELL_H = None
# ──────────────────────────────────────────────────

img = Image.open(SRC)
W, H = img.size
print(f"原圖: {W} × {H} px")

cell_w = CELL_W or (W - OFF_X - GAP_X * (COLS - 1)) // COLS
cell_h = CELL_H or (H - OFF_Y - GAP_Y * (ROWS - 1)) // ROWS
print(f"每格: {cell_w} × {cell_h} px  (gap {GAP_X},{GAP_Y}  offset {OFF_X},{OFF_Y})")

n = 0
for row in range(ROWS):
    for col in range(COLS):
        n += 1
        x = OFF_X + col * (cell_w + GAP_X)
        y = OFF_Y + row * (cell_h + GAP_Y)
        crop = img.crop((x, y, x + cell_w, y + cell_h))
        path = os.path.join(OUT, f"{n}.png")
        crop.save(path)
        print(f"  [{n:02d}] ({x},{y}) → {path}")

print(f"\n完成！共輸出 {n} 張到 {OUT}")
