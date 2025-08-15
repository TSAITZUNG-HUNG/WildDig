# -*- coding: utf-8 -*-
"""
Verification 2：100注生存驗證（含「必輸」判斷）── 單檔輸出指定版型
- 每 100 注為一單位，模擬 10,000 單位 → 共 1,000,000 注
- 先以 p_lose = 1 - TARGET_RTP 判斷是否必輸（必輸者派彩=0，不做獎項分配）
- 非必輸者用 1~55 權重抽獎（仍可能抽到 0/X）
- 以全表相對權重計算 E_full，將獎金等比縮放 scale = 1/E_full，使 E[payout | 非必輸] = 1
  → 整體期望 = (1 - p_lose)*1 = TARGET_RTP；本驗證預設 TARGET_RTP=1.0
- 以 0.05 為一分佈區間，從「≥2」往下到「<0.05, ≥ 0.00」
- 輸出單一工作表，欄位順序與格式如下：
  RTP範圍 | 數量 | 佔比例
  ……（分佈各列）
  （空白列）
  總押
  總贏
  RTP
"""

import numpy as np
import pandas as pd

# ===================== 參數 =====================
TARGET_RTP = 0.93               # 本驗證僅需 RTP=1；如需測試別值可調整
RANDOM_SEED = 43
UNITS = 10_000                 # 單位數（每單位=100注）
BETS_PER_UNIT = 100
TOTAL_BETS = UNITS * BETS_PER_UNIT
TICKET_PRICE = 1.0
BIN_SIZE = 0.05
TOP_BIN_EDGE = 2.0             # 頂層分佈邊界；最上層為 [2.00, +inf)
OUTPUT_XLSX = "WildDig_100注生存驗證.xlsx"
SHEET_NAME = "100注生存驗證"

# ===================== 獎項表（55 組） =====================
LABELS = [
    "0 / X1","0 / X2","0 / X3","0 / X4",
    "10000 / X1","5000 / X2","2000 / X4","2000 / X3",
    "5000 / X1","2000 / X2","1000 / X4","1000 / X3",
    "2000 / X1","1000 / X2","500 / X4","500 / X3",
    "1000 / X1","500 / X2","200 / X4","200 / X3",
    "500 / X1","200 / X2","100 / X4","100 / X3",
    "200 / X1","100 / X2","50 / X4","50 / X3",
    "100 / X1","50 / X2","25 / X4","20 / X4",
    "25 / X3","20 / X3","50 / X1","25 / X2",
    "20 / X2","10 / X4","10 / X3","25 / X1",
    "20 / X1","10 / X2","5 / X4","5 / X3",
    "10 / X1","5 / X2","2 / X4","2 / X3",
    "5 / X1","2 / X2","1 / X4","1 / X3","2 / X1","1 / X2","1 / X1"
]

def bonus_from_label(lbl: str) -> int:
    return int(lbl.split("X")[-1])

BASE_MULT = np.array([
    0,0,0,0,
    10000,5000,2000,2000,
    5000,2000,1000,1000,
    2000,1000,500,500,
    1000,500,200,200,
    500,200,100,100,
    200,100,50,50,
    100,50,25,20,
    25,20,50,25,
    20,10,10,25,
    20,10,5,5,
    10,5,2,2,
    5,2,1,1,2,1,1
], dtype=np.float64)

WEIGHTS = np.array([
    462217,136615,56923,27323,
    1,2,2,2,
    2,2,2,4,
    1,1,2,5,
    4,4,10,10,
    20,15,15,40,
    10,13,20,50,
    15,30,80,150,
    150,200,250,250,
    300,500,650,1000,
    300,810,1000,2200,
    1500,1300,2900,3000,
    8500,4500,7200,14000,51000,74500,140400
], dtype=np.int64)

# ===================== 準備機率與縮放 =====================
assert len(LABELS) == 55 == len(BASE_MULT) == len(WEIGHTS)

BONUS_MULT = np.array([bonus_from_label(lbl) for lbl in LABELS], dtype=np.float64)
PAYOUT_BASE = BASE_MULT * BONUS_MULT                           # 原始派彩（倍數×X）
PROBS_FULL = WEIGHTS / WEIGHTS.sum()                           # 1~55 全表相對機率

# 以「全表相對機率」計算單注期望派彩（含 0 獎）
E_full = float((PAYOUT_BASE * PROBS_FULL).sum())
# 讓「非必輸」時的期望派彩 = 1 → 整體期望 = (1 - p_lose) * 1 = TARGET_RTP
scale = 1.0 / E_full if E_full > 0 else 1.0
PAYOUT = PAYOUT_BASE * scale

# 必輸機率
p_lose = 1.0 - TARGET_RTP

# ===================== 模擬 =====================
rng = np.random.default_rng(RANDOM_SEED)
indices = np.arange(55, dtype=np.int64)

# 1) 必輸判斷（True=必輸，不抽獎；派彩=0）
lose_flags = rng.random(TOTAL_BETS) < p_lose

# 2) 非必輸者抽獎（1~55 全權重；仍可能抽到 0/X）
draw_idx = np.full(TOTAL_BETS, -1, dtype=np.int64)            # -1 代表必輸未抽
num_pass = int((~lose_flags).sum())
if num_pass > 0:
    draw_idx[~lose_flags] = rng.choice(indices, size=num_pass, replace=True, p=PROBS_FULL)

# 3) 建立派彩向量
payouts = np.zeros(TOTAL_BETS, dtype=np.float64)
if num_pass > 0:
    valid = draw_idx >= 0
    payouts[valid] = PAYOUT[draw_idx[valid]]

# ===================== 以 100 注為一單位計算 RTP 並分佈 =====================
unit_wins = payouts.reshape(UNITS, BETS_PER_UNIT).sum(axis=1)
unit_rtp = unit_wins / (TICKET_PRICE * BETS_PER_UNIT)         # 每 100 注的 RTP

# 建立 0.05 的分佈格；頂層為 ≥ 2.00
num_regular_bins = int(TOP_BIN_EDGE / BIN_SIZE)               # 2.00 / 0.05 = 40
counts = np.zeros(num_regular_bins + 1, dtype=int)            # 最後一格為頂層 [2.00, +inf)

regular_mask = unit_rtp < TOP_BIN_EDGE
regular_indices = np.minimum((unit_rtp[regular_mask] / BIN_SIZE).astype(int), num_regular_bins - 1)
np.add.at(counts, regular_indices, 1)
counts[-1] = np.count_nonzero(~regular_mask)                  # 頂層數量

# 依指定格式產生標籤（自上而下）
labels_bins = ["≥2"]
for k in range(num_regular_bins - 1, -1, -1):                 # 從 [1.95,2.00) 到 [0.00,0.05)
    lower = k * BIN_SIZE
    upper = lower + BIN_SIZE
    labels_bins.append(f"<{upper:.2f}, ≥ {lower:.2f}")

# 依相同順序整理 counts
ordered_counts = [int(counts[-1])] + [int(counts[k]) for k in range(num_regular_bins - 1, -1, -1)]
percentages = (np.array(ordered_counts, dtype=np.float64) / UNITS) * 100.0

# ===================== 組成最終表格並輸出 =====================
dist_df = pd.DataFrame({
    "RTP範圍": labels_bins,
    "數量": ordered_counts,
    "佔比例": np.round(percentages, 4),
})

# 總計
total_pay = float(TOTAL_BETS * TICKET_PRICE)
total_win = float(unit_wins.sum())
rtp_overall = total_win / total_pay if total_pay else float('nan')

# 在尾端附上空白列與三列總計（數值放在「數量」欄；「佔比例」留空）
totals_df = pd.DataFrame({
    "RTP範圍": ["", "總押", "總贏", "RTP"],
    "數量": ["", int(total_pay), total_win, rtp_overall],
    "佔比例": ["", "", "", ""],
})

final_df = pd.concat([dist_df, totals_df], ignore_index=True)

with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    final_df.to_excel(writer, index=False, sheet_name=SHEET_NAME)

print(f"完成：{OUTPUT_XLSX} | UNITS={UNITS}, TOTAL_BETS={TOTAL_BETS}, TARGET_RTP={TARGET_RTP:.3f}, p_lose={p_lose:.3f}")
