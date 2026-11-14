"""Анализ результатов скана: топ по DSR."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

df = pd.read_csv("scan_20261114_042229.csv")
print("Всего строк:", len(df))
print()
print("=== ТОП-15 по DSR (честный ранжинг с поправкой на перебор) ===")
top = df.sort_values("dsr", ascending=False).head(15)
for _, r in top.iterrows():
    print(f"{r['symbol']:7s} {r['indicator']:20s} | DSR={r['dsr']:.3f} [{r['dsr_verdict']}] "
          f"| ret={r['return_pct']:8.2f}% n={r['n_trades']:4d} PF={r['profit_factor']:5.2f} "
          f"DD={r['max_dd_pct']:5.2f}%")

print()
print("=== Распределение вердиктов ===")
print(df["dsr_verdict"].value_counts().to_string())

print()
print("=== Лучший DSR по символу ===")
for sym in ["XAUUSD", "GBPUSD", "USDJPY", "US100", "BTCUSD"]:
    sub = df[df["symbol"] == sym]
    if sub.empty:
        continue
    best = sub.loc[sub["dsr"].idxmax()]
    print(f"{sym}: {best['indicator']:20s} DSR={best['dsr']:.3f} "
          f"ret={best['return_pct']:7.2f}% n={best['n_trades']:4d} PF={best['profit_factor']:.2f}")

print()
print("=== Сводка: DSR-пороги ===")
n_sig = (df["dsr"] >= 0.95).sum()
n_marg = ((df["dsr"] >= 0.80) & (df["dsr"] < 0.95)).sum()
n_over = (df["dsr"] < 0.80).sum()
print(f"SIGNIFICANT (>=0.95): {n_sig}")
print(f"MARGINAL (0.80-0.95): {n_marg}")
print(f"OVERFIT (<0.80): {n_over}")
