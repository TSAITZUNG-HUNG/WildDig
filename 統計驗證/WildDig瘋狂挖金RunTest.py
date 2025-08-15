# -*- coding: utf-8 -*-
"""
runtest（一次跑 4 個 RTP；含「必輸」邏輯，並在 Excel 報告中加入「P值」與「是否均勻」欄位）

功能：
- 針對 4 組 RTP（100% / 96.5% / 95% / 93%）各自模擬 1,000,000 局（含必輸判斷）
- 僅檢查「派彩 > 10」的獎項集合，是否依全權重「均勻分配」在非必輸的樣本中
- 產出每個獎項的期望命中、實際命中、百分比誤差、Z 分數，以及「P值」與「是否均勻」
  - P值：以常態近似 Z 轉換（雙尾）p = erfc(|z| / sqrt(2))
  - 是否均勻：以 P值 >= α 視為「是」，否則「否」（預設 α=0.05）
- 另保留主控台輸出一個以百分比誤差門檻（預設 1%）的 PASS/FAIL 摘要
- 匯出 Excel：每個 RTP 一張工作表（檔名預設：runtest_4RTP_results.xlsx）

使用：
- 直接執行本檔：python runtest_4rtp_with_pvalue.py
- 或在其他程式中：from runtest_4rtp_with_pvalue import run_tests; run_tests()
"""

import numpy as np
import pandas as pd
import math

# ===================== 基礎資料（55 組） =====================
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

# ===================== 參數 =====================
RTP_LIST = [1.00, 0.965, 0.95, 0.93]
N_DRAWS = 1_000_000            # 每個 RTP 的模擬局數
TOL_PCT = 1.0                  # 文字摘要的誤差門檻（|百分比誤差| ≤ 1.0%）
ALPHA = 0.001                   # P 值門檻（P >= α → 視為「是：均勻」）
BASE_SEED = 42               # 基礎隨機種子
OUTPUT_XLSX = "WildDig瘋狂挖金RunTest.xlsx"

# 僅檢查「派彩 > 10」的獎項集合（用原始派彩基準：基礎倍數×X）
BONUS_MULT = np.array([bonus_from_label(lbl) for lbl in LABELS], dtype=np.float64)
PAYOUT_BASE = BASE_MULT * BONUS_MULT
MASK_GT10 = PAYOUT_BASE > 10.0

# 55 組全權重機率
PROBS_FULL = WEIGHTS / WEIGHTS.sum()
INDICES = np.arange(55, dtype=np.int64)

def z_to_p_two_tailed(z: np.ndarray) -> np.ndarray:
    """Z 分數轉雙尾 P 值：p = erfc(|z| / sqrt(2))。支援 NaN。"""
    z_abs = np.abs(z.astype(float))
    with np.errstate(over='ignore'):
        p = np.array([math.erfc(v / math.sqrt(2.0)) if np.isfinite(v) else np.nan for v in z_abs])
    return p

def run_one_rtp_test(target_rtp: float, n_draws: int, seed: int,
                     tol_pct: float, alpha: float) -> pd.DataFrame:
    """
    對單一 RTP 做 1,000,000 局的均勻性檢查（含必輸判斷）
    回傳：包含「派彩>10」各獎項的明細 DataFrame，含 P值 與 是否均勻 欄位
    """
    rng = np.random.default_rng(seed)
    p_lose = 1.0 - target_rtp

    # 步驟1：必輸判斷（True=必輸）
    lose_flags = rng.random(n_draws) < p_lose
    n_pass = int((~lose_flags).sum())

    # 步驟2：非必輸者依 55 權重抽獎
    if n_pass > 0:
        draw = rng.choice(INDICES, size=n_pass, replace=True, p=PROBS_FULL)
        hits_all = np.bincount(draw, minlength=55)
    else:
        hits_all = np.zeros(55, dtype=np.int64)

    # 只看派彩 > 10 的獎項
    probs_sel = PROBS_FULL[MASK_GT10]
    hits_sel = hits_all[MASK_GT10]
    exp_sel = n_pass * probs_sel

    # 誤差、Z 分數、P 值（常態近似）
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_err = np.where(exp_sel > 0, (hits_sel - exp_sel) / exp_sel * 100.0, np.nan)
        var_sel = n_pass * probs_sel * (1.0 - probs_sel)
        z = np.where(var_sel > 0, (hits_sel - exp_sel) / np.sqrt(var_sel), np.nan)
    p_vals = z_to_p_two_tailed(z)

    # 「是否均勻」：以 P 值門檻判定（P >= α → 是）
    is_uniform = np.where(p_vals >= alpha, "是", "否")

    df = pd.DataFrame({
        "idx": np.where(MASK_GT10)[0] + 1,         # 1-based 索引（對齊你原表的序號）
        "獎項": np.array(LABELS)[MASK_GT10],
        "派彩": PAYOUT_BASE[MASK_GT10],
        "權重": WEIGHTS[MASK_GT10],
        "機率": probs_sel,
        "N_pass": n_pass,                          # 非必輸樣本量（條件母體）
        "期望命中": exp_sel,
        "實際命中": hits_sel.astype(np.int64),
        "誤差": hits_sel - exp_sel,
        "誤差(%)": pct_err,
        "Z分數": z,
        "P值": p_vals,
        "是否均勻": is_uniform,                    # 以 P 值門檻判定
    }).sort_values(by="idx", ignore_index=True)

    # 主控台摘要（維持原本的百分比誤差門檻統計）
    abs_pct_err = np.abs(df["誤差(%)"].to_numpy(dtype=float))
    max_abs_pct_err = np.nanmax(abs_pct_err)
    mean_abs_pct_err = np.nanmean(abs_pct_err)
    pass_mask = abs_pct_err <= tol_pct
    all_pass = bool(np.all(pass_mask[~np.isnan(abs_pct_err)]))

    print(f"\n=== RTP={target_rtp:.3f} 的均勻性檢查（N={n_draws:,}, N_pass={n_pass:,}, p_lose={p_lose:.3f}） ===")
    print(f"- 百分比誤差門檻：|誤差(%)| ≤ {tol_pct:.3f}% → {'PASS ✅' if all_pass else 'FAIL ❌'}")
    print(f"- 最大|誤差(%)| = {max_abs_pct_err:.4f}% | 平均|誤差(%)| = {mean_abs_pct_err:.4f}%")
    # 也給一個 P 值層面的整體觀感：列出 P 值 < α 的筆數
    num_p_fail = int(np.sum(df["P值"].to_numpy(dtype=float) < alpha))
    print(f"- 以 P 值（α={alpha:.3f}）判定：不均勻（P<α）筆數 = {num_p_fail}")

    return df

def run_tests(rtp_list=None, n_draws=N_DRAWS, tol_pct=TOL_PCT, alpha=ALPHA,
              base_seed=BASE_SEED, to_excel="WildDig瘋狂挖金RunTest.xlsx") -> dict:
    """
    一次跑多個 RTP 的 runtest。
    回傳：{rtp_value: DataFrame}；同時輸出多 Sheet 的 Excel，且每張表包含「P值」與「是否均勻」欄位。
    """
    if rtp_list is None:
        rtp_list = RTP_LIST

    results = {}
    with pd.ExcelWriter(to_excel, engine="xlsxwriter") as writer:
        for i, rtp in enumerate(rtp_list):
            seed = base_seed + i * 9973
            df = run_one_rtp_test(target_rtp=rtp, n_draws=n_draws, seed=seed,
                                  tol_pct=tol_pct, alpha=alpha)
            results[rtp] = df
            sheet = f"RTP={rtp:.3f}"
            # 欄位順序微調（把 P值 / 是否均勻 放到最後較醒目）
            ordered_cols = ["idx","獎項","派彩","權重","機率","N_pass",
                            "期望命中","實際命中","誤差","誤差(%)","Z分數","P值","是否均勻"]
            df[ordered_cols].to_excel(writer, index=False, sheet_name=sheet)
    print(f"\n已輸出 Excel：{to_excel}")
    return results

# 直接執行
if __name__ == "__main__":
    run_tests()
