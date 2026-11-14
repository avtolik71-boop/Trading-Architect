"""Smoke-тест walk-forward: один символ, несколько кандидатов."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import numpy as np
import pandas as pd

from run_pipeline import DataLoader, SYMBOL_SPECS, ScanBacktester, PipelineConfig
from costs.cost_engine import CostEngine, PIPFARM_COSTS
from indicators.indicator_registry import INDICATOR_REGISTRY
from validation.walkforward import WalkForwardEngine, aggregate_oos_metrics, walk_forward_splits

config = PipelineConfig()
loader = DataLoader(config.history_root)
cost_set = PIPFARM_COSTS
ce = CostEngine(cost_set, {k: {"pip_size": v.pip_size} for k, v in SYMBOL_SPECS.items()})
ce.cost_set = cost_set
bt = ScanBacktester(ce, SYMBOL_SPECS)
wf = WalkForwardEngine(bt)

symbol = "GBPUSD"
spec = SYMBOL_SPECS[symbol]
df = loader.load(symbol, config.in_sample_start, config.in_sample_end, "1h")
print(f"{symbol}: {len(df)} баров")

# Число окон
windows = walk_forward_splits(len(df), 18600, 3100, 48)
print(f"Окон: {len(windows)}")
for i, (s, e) in enumerate(windows[:3]):
    print(f"  окно {i}: [{df.index[s].date()} -> {df.index[e-1].date()}]")

# Прогон donchian n=55
ind = INDICATOR_REGISTRY["donchian"]
signal = ind.signal_fn(df, n=55)
trades, n_windows, prof = wf.run(df, signal, symbol, spec.sl_pips, spec.tp_pips, use_actual_spread=True)
print(f"\ndonchian n=55: {len(trades)} OOS сделок, {prof}/{n_windows} прибыльных окон")

# Базовая линия random
random_median = wf.run_random_baseline(df, symbol, spec.sl_pips, spec.tp_pips, spec.pip_size, n_seeds=10)
print(f"random_entry OOS медиана mean = {random_median:.4f}%")

oos = aggregate_oos_metrics(trades, random_median, n_windows, prof)
print(f"\nOOS метрики donchian n=55:")
print(f"  t={oos.t_stat:.3f} PF={oos.profit_factor:.2f} n={oos.n_trades} mean={oos.mean_return_pct:.4f}%")
print(f"  DD={oos.max_dd_pct:.1f}% beat_random={oos.beat_random} passed={oos.passed_gates}")
print(f"  failures={oos.gate_failures}")
