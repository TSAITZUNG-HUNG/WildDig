# -*- coding: utf-8 -*-
"""
WildDig 瘋狂挖金 - 刮刮樂模擬程式（先判斷必輸；不分配至 0/X；不輸出「必輸」欄）
- 先以 p_lose = 1 - TARGET_RTP 判斷必輸；必輸者派彩=0，且不分配到 0/X 欄位。
- 非必輸者用 1~55 權重抽獎（仍可能抽到 0/X）。
- 為使整體期望 RTP = TARGET_RTP：
  * 以全表相對機率計算 E_full = Σ(payout_base * probs_full)
  * 將所有獎金等比例縮放 scale = 1 / E_full，使「非必輸」情況下 E[payout]=1
  * 整體期望 = (1 - p_lose) * 1 = TARGET_RTP
- 停止條件：以「累積 RTP」為準，連續 CONSEC_REQUIRED 批落在 [TARGET_RTP ± TOL]
- 批內「RTP 的標準差」：切成 SUBCHUNKS 段，對各段 RTP 做樣本標準差
"""

import numpy as np
import pandas as pd

# ===================== 可調參數 =====================
TARGET_RTP = 0.93            # 目標 RTP（例：0.965）
TOL = 0.001                   # 容忍度 ±0.1%
BATCH_SIZE = 1_000_000        # 每批張數
CONSEC_REQUIRED = 10          # 連續達標批數（以「累積RTP」為準）
TICKET_PRICE = 1.0            # 單張票價
RANDOM_SEED = 42              # 重現性
MAX_BATCHES = 100_000         # 安全上限（避免無限循環）
SUBCHUNKS = 100               # 批內分段數（用於計算「該批RTP的標準差」）

OUTPUT_XLSX = "WildDig瘋狂挖金統計驗證.xlsx"
SHEET_NAME = "Sheet1"         # 不特別指定可用預設

# ===================== 獎項表定義（55 組） =====================
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
    462217,136615,56923,27323,   # 0 獎項（4個）
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

# ===================== 準備機率與獎金縮放 =====================
assert len(LABELS) == 55 == len(BASE_MULT) == len(WEIGHTS)

BONUS_MULT = np.array([bonus_from_label(lbl) for lbl in LABELS], dtype=np.float64)
PAYOUT_BASE = BASE_MULT * BONUS_MULT               # 原始派彩（倍數×X）

indices = np.arange(55, dtype=np.int64)
probs_full = WEIGHTS / WEIGHTS.sum()               # 1~55 全表相對機率（非必輸用）

# 以「全表相對機率」的期望派彩（含 0 獎）
E_full = float((PAYOUT_BASE * probs_full).sum())
# 讓非必輸時的期望派彩=1 → 整體期望 = TARGET_RTP
scale = 1.0 / E_full if E_full > 0 else 1.0
PAYOUT_TABLE = PAYOUT_BASE * scale

# ===================== 主流程 =====================
rng = np.random.default_rng(RANDOM_SEED)

rows = []
consec_ok = 0
batch_idx = 0
rtp_batch_history = []

p_lose = 1.0 - TARGET_RTP

print(
    f"開始模擬 WildDig 瘋狂挖金（先判斷必輸；不分配至 0/X）\n"
    f"- 目標RTP={TARGET_RTP*100:.3f}% | 必輸機率={p_lose*100:.3f}% | 非必輸機率={(1-p_lose)*100:.3f}%\n"
    f"- 全表期望 E_full={E_full:.6f} → 獎金縮放係數 scale={scale:.9f}\n"
    f"- 容忍±{TOL*100:.3f}% | 批次大小={BATCH_SIZE:,} | 需連續達標 {CONSEC_REQUIRED}\n"
)

while batch_idx < MAX_BATCHES and consec_ok < CONSEC_REQUIRED:
    batch_idx += 1

    # 1) 必輸判斷（必輸者不做獎項分配）
    lose_flags = rng.random(BATCH_SIZE) < p_lose
    num_pass = BATCH_SIZE - int(lose_flags.sum())

    # 2) 僅對「非必輸」者抽獎（1~55 全權重；仍可能抽到 0 獎）
    draw_idx = np.full(BATCH_SIZE, -1, dtype=np.int64)  # -1 代表必輸未抽獎
    if num_pass > 0:
        draw_idx[~lose_flags] = rng.choice(indices, size=num_pass, replace=True, p=probs_full)

    # 3) 建立整批派彩（必輸=0；非必輸依抽獎結果）
    payouts = np.zeros(BATCH_SIZE, dtype=np.float64)
    if num_pass > 0:
        valid = draw_idx >= 0
        payouts[valid] = PAYOUT_TABLE[draw_idx[valid]]

    # ---- 批次統計 ----
    # 只統計「非必輸抽獎」的結果（必輸不計入 55 欄）
    hit_counts = np.bincount(draw_idx[draw_idx >= 0], minlength=55) if num_pass > 0 else np.zeros(55, dtype=np.int64)

    total_pay = TICKET_PRICE * BATCH_SIZE
    total_win = float(payouts.sum())
    rtp_batch = total_win / total_pay

    # 批內 RTP 標準差（分段法）
    if BATCH_SIZE % SUBCHUNKS != 0:
        raise ValueError("BATCH_SIZE 必須能整除 SUBCHUNKS")
    seg_size = BATCH_SIZE // SUBCHUNKS
    rtp_segments = []
    start = 0
    for _ in range(SUBCHUNKS):
        end = start + seg_size
        seg_win = float(payouts[start:end].sum())
        rtp_segments.append(seg_win / (TICKET_PRICE * seg_size))
        start = end
    rtp_segments = np.array(rtp_segments, dtype=np.float64)
    std_batch = float(rtp_segments.std(ddof=1)) if SUBCHUNKS > 1 else float("nan")

    # 累積（以「各批RTP」計）
    rtp_batch_history.append(rtp_batch)
    cum_rtp = float(np.mean(rtp_batch_history))
    cum_std = float(np.std(rtp_batch_history, ddof=1)) if len(rtp_batch_history) > 1 else float("nan")

    # 中獎資訊（>0 派彩；必輸自帶 0，非必輸抽到 0 也算 0）
    win_cards = int((payouts > 0).sum())
    win_rate = win_cards / BATCH_SIZE

    # 停止條件（以「累積RTP」）
    if (TARGET_RTP - TOL) <= cum_rtp <= (TARGET_RTP + TOL):
        consec_ok += 1
    else:
        consec_ok = 0

    # 記錄一列（不輸出「必輸」欄）
    row = {
        "驗證次數": batch_idx,
        "Total Pay": total_pay,
        "Total Win": total_win,
        "RTP": rtp_batch,
        "標準差": std_batch,       # 該批RTP標準差（分段）
        "累積標準差": cum_std,      # 迄今各批RTP的標準差
        "中獎卡數": win_cards,
        "中獎率": win_rate,
    }
    for i in range(55):
        row[str(i+1)] = int(hit_counts[i])  # 只包含非必輸抽獎的結果
    rows.append(row)

    # 進度提示（不印「必輸」數）
    print(f"批次 {batch_idx:>4} 完成 | 批次RTP={rtp_batch:.6f} | 累積RTP={cum_rtp:.6f} | 連續達標={consec_ok}/{CONSEC_REQUIRED}")

# ===================== 匯出結果 =====================
df = pd.DataFrame(rows)

if len(df) > 0:
    total_pay_all = df["Total Pay"].sum()
    total_win_all = df["Total Win"].sum()
    rtp_all = total_win_all / total_pay_all if total_pay_all else np.nan

    total_row = {
        "驗證次數": "總計",
        "Total Pay": total_pay_all,
        "Total Win": total_win_all,
        "RTP": rtp_all,
        "標準差": np.nan,
        "累積標準差": float(np.std(df["RTP"].to_numpy(dtype=np.float64), ddof=1)) if len(df) > 1 else np.nan,
        "中獎卡數": int(df["中獎卡數"].sum()),
        "中獎率": float(df["中獎卡數"].sum() / (BATCH_SIZE * len(df))),
    }
    for i in range(55):
        total_row[str(i+1)] = int(df[str(i+1)].sum())
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

# 欄位順序（無「必輸」欄）
summary_cols = ["驗證次數","Total Pay","Total Win","RTP","標準差","累積標準差","中獎卡數","中獎率"]
prize_cols = [str(i) for i in range(1, 56)]
df = df[summary_cols + prize_cols]

with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name=SHEET_NAME)

print(f"\n模擬完成！共輸出 {len(df)-1 if len(df)>0 else 0} 批 + 總計列，檔案已儲存為：{OUTPUT_XLSX}")
