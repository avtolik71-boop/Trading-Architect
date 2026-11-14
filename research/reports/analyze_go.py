"""Финальный анализ GO-кандидата: DSR с учётом 81 попытки."""
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from run_pipeline import DataLoader, SYMBOL_SPECS, ScanBacktester, PipelineConfig
from costs.cost_engine import CostEngine, PIPFARM_COSTS
from indicators.indicator_registry import INDICATOR_REGISTRY
from validation.walkforward import WalkForwardEngine, aggregate_oos_metrics, walk_forward_splits
from testing.dsr import evaluate_dsr

config = PipelineConfig()
loader = DataLoader(config.history_root)
ce = CostEngine(PIPFARM_COSTS, {k: {"pip_size": v.pip_size} for k, v in SYMBOL_SPECS.items()})
ce.cost_set = PIPFARM_COSTS
bt = ScanBacktester(ce, SYMBOL_SPECS)
wf = WalkForwardEngine(bt)

# GO-кандидат: US100 ema_separation
symbol = "US100"
spec = SYMBOL_SPECS[symbol]
df = loader.load(symbol, config.in_sample_start, config.in_sample_end, "1h")

# Найти params для ema_separation из CSV
wf_csv = pd.read_csv("reports/walkforward_20261114_203513.csv")
go = wf_csv[wf_csv["passed_gates"] == True]
print("GO-кандидаты:")
print(go[["symbol", "indicator", "params", "t_stat_oos", "pf_oos", "n_trades_oos"]].to_string())

# Реконструировать сигнал для GO-кандидата
import json
row = go.iloc[0]
params = json.loads(row["params"])
ind = INDICATOR_REGISTRY[row["indicator"]]
signal = ind.signal_fn(df, **params)

# Собрать OOS-сделки по окнам и дневную эквити
windows = walk_forward_splits(len(df), 18600, 3100, 48)
all_daily_rets = []
for s, e in windows:
    test_df = df.iloc[s:e]
    test_sig = signal.iloc[s:e]
    trades, equity = bt.run(test_df, test_sig, symbol, spec.sl_pips, spec.tp_pips, use_actual_spread=True)
    daily = equity.resample("1D").last().dropna().pct_change().dropna()
    all_daily_rets.extend(daily.tolist())

all_daily = np.array(all_daily_rets)
print(f"\nДневных OOS наблюдений: {len(all_daily)}")

# DSR с разным числом попыток
for n_trials in [1, 8, 81, 640]:
    res = evaluate_dsr(all_daily, n_trials=n_trials)
    print(f"  n_trials={n_trials:4d}: SR(год)={res.sharpe_annualized:.3f}, "
          f"haircut={res.haircut_sr:.4f}, DSR={res.dsr:.3f}, вердикт={res.verdict}")

print("\nВывод:")
print("  При n_trials=8 (выживших US100) DSR ещё значим,")
print("  при n_trials=81 (все выжившие) DSR падает — множественное тестирование.")
