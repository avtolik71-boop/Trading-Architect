"""
Walk-Forward runner — честная OOS-валидация
===========================================
1. Загрузить in-sample результаты скана (scan JSON)
2. Отфильтровать выживших (n>=30, PF>=1.0, max_dd<=20%)
3. Walk-forward каждого выжившего (15 окон, train 18600 / test 3100 / embargo 48)
4. Прогнать через OOS-ворота: t>=2, PF>=1.1, mean>0, n>=50, beat random
5. Отчёт: GO-кандидаты + DSR

Запуск:
  python run_walkforward.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from indicators.indicator_registry import INDICATOR_REGISTRY
from costs.cost_engine import CostEngine, PIPFARM_COSTS, TICKMILL_COSTS
from run_pipeline import (
    DataLoader, SYMBOL_SPECS, ScanBacktester, PipelineConfig,
)
from validation.walkforward import (
    WalkForwardEngine, aggregate_oos_metrics, OOSResult,
)


# =============================================================================
# Загрузка выживших из скана
# =============================================================================

def load_survivors(scan_json_path: str, min_n: int = 30, min_pf: float = 1.0,
                   max_dd: float = 20.0) -> dict[str, list[dict]]:
    """Загрузить выживших кандидатов из JSON-отчёта скана."""
    with open(scan_json_path, "r", encoding="utf-8") as f:
        scan = json.load(f)

    survivors = {}
    for symbol, res in scan.items():
        if not isinstance(res, dict) or "results" not in res:
            continue
        surv = []
        for r in res["results"]:
            if (r.get("n_trades", 0) >= min_n
                    and r.get("profit_factor", 0) >= min_pf
                    and r.get("max_dd_pct", 0) <= max_dd):
                surv.append(r)
        survivors[symbol] = surv
        print(f"  {symbol}: {len(res['results'])} кандидатов -> {len(surv)} выживших "
              f"(n>={min_n}, PF>={min_pf}, DD<={max_dd}%)")
    return survivors


# =============================================================================
# Walk-forward прогон
# =============================================================================

def run_walkforward():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  Walk-Forward валидация (OOS) — train 18600 / test 3100 / embargo 48")
    print("  Ворота: t>=2, PF>=1.1, mean>0, n>=50, beat random_entry OOS")
    print("=" * 70)

    scan_dir = Path(__file__).resolve().parent / "reports"
    scan_jsons = sorted(scan_dir.glob("scan_*.json"))
    if not scan_jsons:
        print("  Нет скана. Сначала запустите run_pipeline.py")
        return
    scan_json = scan_jsons[-1]  # последний
    print(f"  Скан: {scan_json.name}")

    survivors = load_survivors(str(scan_json))

    config = PipelineConfig()
    loader = DataLoader(config.history_root)
    cost_set = PIPFARM_COSTS if config.cost_set == "PIPFARM_CTRADER" else TICKMILL_COSTS
    cost_engine = CostEngine(cost_set, {k: {"pip_size": v.pip_size} for k, v in SYMBOL_SPECS.items()})
    cost_engine.cost_set = cost_set
    backtester = ScanBacktester(cost_engine, SYMBOL_SPECS)
    wf = WalkForwardEngine(backtester)

    all_go = []
    report_rows = []

    for symbol, surv_list in survivors.items():
        if not surv_list:
            continue
        spec = SYMBOL_SPECS[symbol]
        df = loader.load(symbol, config.in_sample_start, config.in_sample_end, config.timeframe)
        print(f"\n{'='*70}")
        print(f"  {symbol}: {len(surv_list)} выживших на walk-forward | {len(df)} баров")
        print(f"{'='*70}")

        # Базовая линия random_entry OOS
        t0 = time.time()
        random_median = wf.run_random_baseline(
            df, symbol, spec.sl_pips, spec.tp_pips, spec.pip_size,
            n_seeds=30, use_actual_spread=config.use_actual_spread,
        )
        print(f"  random_entry OOS медиана mean = {random_median:.4f}% (baseline)")

        for r in surv_list:
            ind_name = r["indicator"]
            params = r["params"]
            if ind_name not in INDICATOR_REGISTRY:
                continue
            ind = INDICATOR_REGISTRY[ind_name]

            try:
                signal = ind.signal_fn(df, **params)
            except Exception:
                continue

            trades, n_windows, prof_windows = wf.run(
                df, signal, symbol, spec.sl_pips, spec.tp_pips,
                use_actual_spread=config.use_actual_spread,
            )
            oos = aggregate_oos_metrics(
                trades, random_median, n_windows, prof_windows,
            )

            row = {
                "symbol": symbol,
                "indicator": ind_name,
                "params": json.dumps(params, ensure_ascii=False),
                "n_windows": oos.n_windows,
                "windows_profitable": oos.windows_profitable,
                "n_trades_oos": oos.n_trades,
                "win_rate_oos": oos.win_rate,
                "pf_oos": oos.profit_factor,
                "mean_return_oos": oos.mean_return_pct,
                "t_stat_oos": oos.t_stat,
                "total_return_oos": oos.total_return_pct,
                "max_dd_oos": oos.max_dd_pct,
                "random_median_mean": oos.random_median_mean_pct,
                "beat_random": oos.beat_random,
                "passed_gates": oos.passed_gates,
                "gate_failures": ",".join(oos.gate_failures),
            }
            report_rows.append(row)

            if oos.passed_gates:
                all_go.append(row)
                print(f"  [GO] {ind_name:22s} t={oos.t_stat:.2f} PF={oos.profit_factor:.2f} "
                      f"n={oos.n_trades} mean={oos.mean_return_pct:.4f}% "
                      f"win={oos.windows_profitable}/{oos.n_windows} DD={oos.max_dd_pct:.1f}%")

        print(f"  Время: {time.time()-t0:.1f}s")

    # =========================================================================
    # Отчёт
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  ИТОГ WALK-FORWARD")
    print(f"{'='*70}")

    if all_go:
        print(f"\n  GO-КАНДИДАТЫ (прошли все OOS-ворота): {len(all_go)}")
        for g in all_go:
            print(f"    {g['symbol']:7s} {g['indicator']:22s} | t={g['t_stat_oos']:.2f} "
                  f"PF={g['pf_oos']:.2f} n={g['n_trades_oos']} "
                  f"mean={g['mean_return_oos']:.4f}% DD={g['max_dd_oos']:.1f}%")
    else:
        print("\n  GO-КАНДИДАТОВ: 0 — ни одна стратегия не прошла все OOS-ворота.")
        print("  Это честный результат: фиксированные индикаторы + фикс. SL/TP")
        print("  редко дают устойчивый эдж без walk-forward подгонки.")

    # Сохранить CSV
    if report_rows:
        df_out = pd.DataFrame(report_rows)
        out_path = scan_dir / f"walkforward_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  Отчёт: {out_path}")

        # Топ по t-stat
        df_out["t_stat_oos"] = pd.to_numeric(df_out["t_stat_oos"], errors="coerce")
        top = df_out.sort_values("t_stat_oos", ascending=False).head(15)
        print(f"\n  ТОП-15 по t-статистике OOS:")
        for _, row in top.iterrows():
            print(f"    {row['symbol']:7s} {row['indicator']:22s} t={row['t_stat_oos']:.2f} "
                  f"PF={row['pf_oos']:.2f} n={row['n_trades_oos']} "
                  f"mean={row['mean_return_oos']:.4f}% win={row['windows_profitable']}/{row['n_windows']}")

    return all_go


if __name__ == "__main__":
    run_walkforward()
