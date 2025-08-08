import torch
import pandas as pd
from collections import defaultdict

# ----------- 模擬參數 -------------
must_lose_rate = 0.07        # 設定必輸率 (0.01 = 1%)
batch_size = 10_000_000        # 每批模擬數
max_total = 100_000_000_000  # 最多模擬數：1 兆
rtp_tolerance = 0.001          # RTP 誤差容許範圍
success_streak_required = 10   # 連續幾批落在範圍內就提前停止

# ----------- 賠率與權重資料（RTP 100% 設定）-------------
multipliers = [
    0, 0, 0, 0,
    10000, 10000, 8000, 6000, 5000, 4000, 4000, 3000,
    2000, 2000, 2000, 1500, 1000, 1000, 800, 600, 500,
    400, 400, 300, 200, 200, 200, 150, 100, 100, 100,
    80, 75, 60, 50, 50, 40, 40, 30, 25, 20, 20, 20,
    15, 10, 10, 8, 6, 5, 4, 4, 3, 2, 2, 1
]
weights = [
    462217, 136615, 56923, 27323,
    1, 2, 2, 2, 2, 2, 2, 4,
    1, 1, 2, 5, 4, 4, 10, 10, 20,
    15, 15, 40, 10, 13, 20, 50, 15, 30, 80,
    150, 150, 200, 250, 250, 300, 500, 650, 1000, 300,
    810, 1000, 2200, 1500, 1300, 2900, 3000, 8500, 4500, 7200,
    14000, 51000, 74500, 140400
]

# ----------- 初始化張量與理論表 -------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
multipliers_tensor = torch.tensor(multipliers, dtype=torch.float32, device=device)  # MPS 限制只能用 float32
weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
probabilities = weights_tensor / weights_tensor.sum()

df_theoretical = pd.DataFrame({
    "倍數": multipliers,
    "理論權重": weights
})
df_theoretical["理論出現佔比 (%)"] = df_theoretical["理論權重"] / weights_tensor.sum().item() * 100 * (1-must_lose_rate)
df_theoretical = df_theoretical.groupby("倍數", as_index=False).sum().sort_values(by="倍數", ascending=False)

# ----------- 模擬主迴圈 (GPU 運算 + CPU 精度累加) -------------
total_return = torch.tensor(0.0, dtype=torch.float64)  # CPU 累加用 float64
total_wins = torch.tensor(0.0, dtype=torch.float64)    # CPU 累加用 float64
count_dict = defaultdict(int)
num_simulated = torch.tensor(0.0, dtype=torch.float64)

rtp_target = 1.0 - must_lose_rate
rtp_success_streak = 0
max_batches = max_total // batch_size

for i in range(max_batches):
    samples = torch.multinomial(probabilities, batch_size, replacement=True)
    drawn = multipliers_tensor[samples]

    # 強制不中獎
    if must_lose_rate > 0:
        mask = (torch.rand(batch_size, device=device) < must_lose_rate) & (drawn > 0)
        drawn[mask] = 0.0

    # 先在 GPU sum，再轉 CPU float64 累加
    total_return += drawn.sum().cpu().to(torch.float64)
    total_wins += (drawn > 0).sum().cpu().to(torch.float64)
    num_simulated += batch_size

    # 統計每個倍數出現次數（CPU 處理）
    for m in drawn.cpu().numpy():
        count_dict[float(m)] += 1

    # 計算累積 RTP（float64 精度）
    current_rtp = (total_return / num_simulated).item()
    if rtp_target - rtp_tolerance <= current_rtp <= rtp_target + rtp_tolerance:
        rtp_success_streak += 1
    else:
        rtp_success_streak = 0

    print(f"第 {i+1} 批｜模擬數: {int(num_simulated):,}｜RTP: {current_rtp:.5f}｜中獎率: {(total_wins / num_simulated)*100:.2f}%｜穩定連續批數: {rtp_success_streak}")

    if rtp_success_streak >= success_streak_required:
        print("✅ 提前完成：已達連續穩定條件")
        break

# ----------- 匯出結果 -------------
df_actual = pd.DataFrame([
    {"倍數": k, "實際出現次數": v, "實際出現佔比 (%)": (v / num_simulated.item()) * 100}
    for k, v in count_dict.items()
])
df_actual = df_actual.sort_values(by="倍數", ascending=False)

df_result = pd.merge(df_theoretical, df_actual, on="倍數", how="outer").fillna(0)
df_result = df_result.sort_values(by="倍數", ascending=False)

df_result.to_excel("模擬結果_賠率統計.xlsx", index=False)

print("\n✅ Excel 已匯出：模擬結果_賠率統計.xlsx")
print(f"🎯 模擬總次數：{int(num_simulated):,}")
print(f"📈 最終 RTP：{(total_return / num_simulated).item():.5f}")
print(f"🥇 中獎率：{(total_wins / num_simulated).item() * 100:.2f}%")
