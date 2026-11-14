"""
Главный оркестратор пайплайна перебора стратегий (v3)
======================================================
Полный цикл исследования с DSR-поправкой:
  1. Загрузить M1 данные из E:\ИСТОРИЯ, ресемплировать в H1
  2. Перебрать ВСЕ индикаторы и ВСЕ комбинации параметров
  3. Бэктест: сигнал[i-1] -> вход open[i], интрабарные SL/TP (high/low)
  4. Издержки ВСЕГДА: спред (из данных) + комиссия (конфиг) + своп (конфиг)
  5. Нуль-модели: random_entry (30 сидов) + buy&hold
  6. DSR (Deflated Sharpe Ratio) с n_trials = число кандидатов
  7. Риск-профили: PIPFARM / FREE / CLASSIC
  8. Отчёт: CSV + JSON + таблица топов по DSR

ДАННЫЕ: E:\ИСТОРИЯ (M1, индекс ts_utc, колонки open/high/low/close/volume/spread)
БРОКЕР: Pipfarm cTrader / Tickmill
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from indicators.indicator_registry import (
    INDICATOR_REGISTRY, enumerate_all_indicators, METHOD_MANIFEST,
)
from costs.cost_engine import CostEngine, PIPFARM_COSTS, TICKMILL_COSTS
from testing.null_models import NullModelEngine
from testing.dsr import evaluate_dsr
from risk_mgmt.risk_profiles import (
    ALL_PROFILES, profile_gate_pass, compute_total_drawdown,
    compute_max_daily_drawdown,
)


# =============================================================================
# Конфигурация символов
# =============================================================================

@dataclass(frozen=True)
class SymbolSpec:
    pip_size: float            # размер пипса в ценовых единицах
    dollar_per_pip: float      # $ P&L за 1 пипс на 1 лот
    contract_size: float       # единиц актива в 1 лоте
    sl_pips: float             # стоп-лосс по умолчанию (пипсы)
    tp_pips: float             # тейк-профит по умолчанию (пипсы)


SYMBOL_SPECS = {
    "XAUUSD": SymbolSpec(pip_size=0.1, dollar_per_pip=10.0, contract_size=100.0, sl_pips=500, tp_pips=400),
    "GBPUSD": SymbolSpec(pip_size=0.0001, dollar_per_pip=10.0, contract_size=100000.0, sl_pips=150, tp_pips=225),
    "USDJPY": SymbolSpec(pip_size=0.01, dollar_per_pip=6.5, contract_size=100000.0, sl_pips=100, tp_pips=150),
    "US100":  SymbolSpec(pip_size=1.0, dollar_per_pip=1.0, contract_size=1.0, sl_pips=200, tp_pips=300),
    "BTCUSD": SymbolSpec(pip_size=1.0, dollar_per_pip=1.0, contract_size=1.0, sl_pips=3000, tp_pips=4500),
}


# =============================================================================
# Конфигурация пайплайна
# =============================================================================

@dataclass
class PipelineConfig:
    history_root: str = r"E:\ИСТОРИЯ"
    symbols: list[str] = field(default_factory=lambda: ["XAUUSD", "GBPUSD", "USDJPY", "US100", "BTCUSD"])
    timeframe: str = "1h"                 # ресемплинг с M1
    in_sample_start: str = "2016-01-01"
    in_sample_end: str = "2026-12-31"     # полный диапазон (walk-forward отдельно)
    cost_set: str = "PIPFARM_CTRADER"     # или "TICKMILL"
    null_seeds: int = 30
    null_prob: float = 0.002
    filters: list[str] = field(default_factory=lambda: ["none"])
    exits: list[str] = field(default_factory=lambda: ["fixed_sl_tp"])
    min_trades: int = 30                  # порог для зачёта (анти-дрейф)
    max_candidates_per_indicator: Optional[int] = None  # None = все (для отладки можно ограничить)
    use_actual_spread: bool = True        # True = спред из колонки данных; False = конфиг


# =============================================================================
# Загрузчик данных (M1 -> ресемплинг)
# =============================================================================

class DataLoader:
    """Загрузка M1 Parquet и ресемплинг в целевой таймфрейм."""

    RESAMPLE_MAP = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "tick_volume": "sum", "spread": "mean",
    }

    def __init__(self, root: str = r"E:\ИСТОРИЯ"):
        self.root = Path(root)
        self._cache: dict[tuple, pd.DataFrame] = {}

    def available_symbols(self) -> list[str]:
        return [d.name for d in self.root.iterdir() if d.is_dir() and d.name != "__pycache__"]

    def load_m1(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Загрузить сырые M1 бары (индекс ts_utc уже datetime)."""
        cache_key = (symbol, start, end, "m1")
        if cache_key in self._cache:
            return self._cache[cache_key]

        symbol_dir = self.root / symbol
        if not symbol_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {symbol_dir}")

        frames = []
        for year in range(int(start[:4]), int(end[:4]) + 1):
            p = symbol_dir / f"{year}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                frames.append(df)

        if not frames:
            raise FileNotFoundError(f"Нет данных для {symbol} за {start}-{end}")

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()
        result = result.loc[start:end].copy()

        # Нормализовать колонки
        if "tick_volume" not in result.columns and "volume" in result.columns:
            result = result.rename(columns={"volume": "tick_volume"})

        self._cache[cache_key] = result
        return result

    def load(self, symbol: str, start: str, end: str,
             timeframe: str = "1h") -> pd.DataFrame:
        """Загрузить и ресемплировать данные в целевой таймфрейм."""
        cache_key = (symbol, start, end, timeframe)
        if cache_key in self._cache:
            return self._cache[cache_key]

        m1 = self.load_m1(symbol, start, end)
        tf_map = {"1h": "1h", "H1": "1h", "4h": "4h", "1d": "1D", "D1": "1D"}
        rule = tf_map.get(timeframe, timeframe)

        resampled = m1.resample(rule).agg(self.RESAMPLE_MAP).dropna()
        result = resampled.copy()

        self._cache[cache_key] = result
        return result


# =============================================================================
# Бэктестер (сигнал -> сделки, интрабарные SL/TP, издержки)
# =============================================================================

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int                     # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_usd: float
    return_pct: float             # % от номинала
    costs_usd: float
    reason: str                   # take_profit / stop_loss / reverse / end


class ScanBacktester:
    """
    Бэктестер: сигнал[i-1] -> вход по open[i].
    Интрабарные SL/TP: если бар касается обоих уровней, срабатывает SL первым
    (консервативно). Издержки вычитаются всегда.
    """

    def __init__(self, cost_engine: CostEngine, symbol_specs: dict):
        self.cost_engine = cost_engine
        self.specs = symbol_specs

    def run(self, df: pd.DataFrame, signal: pd.Series, symbol: str,
            sl_pips: float, tp_pips: float,
            use_actual_spread: bool = True) -> tuple[list[Trade], pd.Series]:
        """
        Прогнать сигнал через бэктестер.
        Возвращает (список сделок, эквити-кривая в % от стартового капитала).
        """
        spec = self.specs[symbol]
        pip = spec.pip_size
        dpip = spec.dollar_per_pip
        csize = spec.contract_size

        n = len(df)
        c = df["close"].to_numpy()
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        spread = df["spread"].to_numpy() if use_actual_spread else None
        ts = df.index

        # Издержки на вход (в пипсах): комиссия
        cost_cfg = None
        if hasattr(self.cost_engine, "cost_set"):
            try:
                cost_cfg = self.cost_engine.cost_set.get(symbol)
            except KeyError:
                cost_cfg = None  # символ без конфига издержек — комиссия/своп = 0
        commission_usd = cost_cfg.commission_per_lot if cost_cfg else 0.0
        commission_pips = commission_usd / dpip if dpip > 0 else 0.0

        trades: list[Trade] = []
        equity = np.ones(n)          # мультипликативная кривая (1.0 = старт)
        pos = 0
        entry = 0.0
        entry_i = 0
        entry_notional = 1.0

        sig = signal.to_numpy() if isinstance(signal, pd.Series) else np.asarray(signal)

        for i in range(1, n):
            if pos == 0:
                # Ищем вход по сигналу предыдущего бара
                if sig[i - 1] != 0:
                    pos = 1 if sig[i - 1] > 0 else -1
                    entry = o[i]
                    entry_i = i
                    entry_notional = entry * csize
                equity[i] = equity[i - 1]
                continue

            # Держим позицию — интрабарная проверка SL/TP
            sl_price = entry - pos * (sl_pips * pip)   # для long: entry - sl; short: entry + sl
            tp_price = entry + pos * (tp_pips * pip)

            exit_price = None
            reason = None

            if pos == 1:  # long
                if l[i] <= sl_price:
                    exit_price = sl_price; reason = "stop_loss"
                elif h[i] >= tp_price:
                    exit_price = tp_price; reason = "take_profit"
            else:  # short
                if h[i] >= sl_price:
                    exit_price = sl_price; reason = "stop_loss"
                elif l[i] <= tp_price:
                    exit_price = tp_price; reason = "take_profit"

            if exit_price is None:
                equity[i] = equity[i - 1]
                continue

            # Закрытие сделки
            pnl_pips = (exit_price - entry) / pip * pos
            pnl_usd = pnl_pips * dpip

            # Издержки: спред (на входе, в ценовых единицах) + комиссия + своп
            spread_cost_usd = 0.0
            if use_actual_spread and spread is not None:
                spread_price = float(spread[entry_i])          # спред бара входа (ценовые единицы)
                spread_pips = spread_price / pip
                spread_cost_usd = spread_pips * dpip

            commission_total_usd = commission_usd  # round-trip (Pipfarm: всё при открытии)

            # Своп (ночи удержания)
            nights = max(0, (ts[i] - ts[entry_i]).days)
            swap_usd = 0.0
            if cost_cfg is not None and nights > 0:
                swap_points = cost_cfg.swap_long_points if pos == 1 else cost_cfg.swap_short_points
                # тройной своп в среду (упрощённо: средняя множитель ~1.286 за ночь)
                swap_usd = swap_points * nights * dpip * 0.01  # points -> $ (приблизительно)

            costs_usd = spread_cost_usd + commission_total_usd + swap_usd
            net_usd = pnl_usd - costs_usd
            ret_pct = net_usd / entry_notional if entry_notional > 0 else 0.0

            trades.append(Trade(
                entry_time=ts[entry_i], exit_time=ts[i], side=pos,
                entry_price=entry, exit_price=exit_price,
                pnl_pips=pnl_pips, pnl_usd=pnl_usd,
                return_pct=ret_pct, costs_usd=costs_usd, reason=reason,
            ))
            equity[i] = equity[i - 1] * (1.0 + ret_pct)
            pos = 0

        equity_series = pd.Series(equity, index=df.index, name="equity")
        return trades, equity_series


# =============================================================================
# Метрики
# =============================================================================

def trades_to_base_metrics(trades: list[Trade], equity: pd.Series) -> dict:
    """Метрики, НЕ зависящие от n_trials (без DSR)."""
    if not trades:
        return {"n_trades": 0, "return_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_dd_pct": 0.0, "sharpe": 0.0, "avg_trade_return_pct": 0.0,
                "avg_trade_cost_usd": 0.0, "daily_returns": None}

    rets = np.array([t.return_pct for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    total_return = (equity.iloc[-1] - 1.0) * 100
    win_rate = len(wins) / len(rets) * 100
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    max_dd = compute_total_drawdown(equity)

    daily = equity.resample("1D").last().dropna()
    daily_rets = daily.pct_change().dropna()
    sharpe = daily_rets.mean() / (daily_rets.std() + 1e-12) * np.sqrt(252) if len(daily_rets) > 5 else 0.0

    return {
        "n_trades": len(trades),
        "return_pct": round(total_return, 3),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else 999.0,
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(float(sharpe), 3),
        "avg_trade_return_pct": round(float(rets.mean()), 4),
        "avg_trade_cost_usd": round(float(np.mean([t.costs_usd for t in trades])), 3),
        "daily_returns": daily_rets.to_numpy(),
    }


def apply_dsr(metrics_row: dict, n_trials: int) -> dict:
    """Дополнить строку метрик DSR с правильным n_trials."""
    dr = metrics_row.get("daily_returns")
    if dr is not None and len(dr) > 30:
        dsr_result = evaluate_dsr(np.asarray(dr, dtype=float), n_trials=n_trials)
        metrics_row["dsr"] = round(dsr_result.dsr, 4)
        metrics_row["dsr_verdict"] = dsr_result.verdict
        metrics_row["dsr_haircut"] = round(dsr_result.haircut_sr, 5)
    else:
        metrics_row["dsr"] = 0.0
        metrics_row["dsr_verdict"] = "NO_TRADES"
        metrics_row["dsr_haircut"] = 0.0
    metrics_row.pop("daily_returns", None)
    return metrics_row


# =============================================================================
# Оркестратор
# =============================================================================

class ResearchPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.loader = DataLoader(self.config.history_root)
        cost_set = PIPFARM_COSTS if self.config.cost_set == "PIPFARM_CTRADER" else TICKMILL_COSTS
        self.cost_engine = CostEngine(cost_set, {k: {"pip_size": v.pip_size} for k, v in SYMBOL_SPECS.items()})
        self.cost_engine.cost_set = cost_set
        self.backtester = ScanBacktester(self.cost_engine, SYMBOL_SPECS)
        self.null_engine = NullModelEngine()

    def run_symbol(self, symbol: str, start: str, end: str) -> dict:
        t0 = time.time()
        print(f"\n{'='*64}")
        print(f"  {symbol} | {start} -> {end} | {self.config.cost_set} | TF={self.config.timeframe}")
        print(f"{'='*64}")

        spec = SYMBOL_SPECS[symbol]
        df = self.loader.load(symbol, start, end, self.config.timeframe)
        print(f"  Баров: {len(df)} ({df.index[0].date()} -> {df.index[-1].date()})")

        # Нуль-модели
        null_dist = self.null_engine.random_entry_multi_seed(
            df, prob=self.config.null_prob, n_seeds=self.config.null_seeds,
            sl_pips=spec.sl_pips, tp_pips=spec.tp_pips, pip_size=spec.pip_size,
        )
        null_median = (null_dist.iloc[-1].median() - 1.0) * 100
        bh_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100
        print(f"  Нуль-модель (медиана random_entry): {null_median:.2f}% | Buy&Hold: {bh_ret:.2f}%")

        # Перебор всех индикаторов
        all_candidates = enumerate_all_indicators(self.config.filters, self.config.exits)
        results = []
        total_tested = 0

        for ind_name, candidates in all_candidates.items():
            ind = INDICATOR_REGISTRY[ind_name]
            if self.config.max_candidates_per_indicator is not None:
                candidates = candidates[:self.config.max_candidates_per_indicator]

            for cfg in candidates:
                total_tested += 1
                try:
                    signal = ind.signal_fn(df, **cfg.params)
                    trades, equity = self.backtester.run(
                        df, signal, symbol,
                        sl_pips=spec.sl_pips, tp_pips=spec.tp_pips,
                        use_actual_spread=self.config.use_actual_spread,
                    )
                    if len(trades) < self.config.min_trades:
                        continue
                    metrics = trades_to_base_metrics(trades, equity)
                    metrics.update({
                        "symbol": symbol,
                        "indicator": ind_name,
                        "params": cfg.params,
                        "label": cfg.label,
                    })
                    results.append(metrics)
                except Exception:
                    pass  # тихо пропустить ошибочные конфигурации

        # Финальный DSR с правильным n_trials (все протестированные конфигурации)
        n_trials = max(1, total_tested)
        for r in results:
            r["n_trials"] = n_trials
            apply_dsr(r, n_trials)

        # Ранжирование по DSR
        results.sort(key=lambda x: -x["dsr"])

        elapsed = time.time() - t0
        print(f"  Протестировано: {total_tested} конфигов, валидных (n>={self.config.min_trades}): {len(results)}")
        print(f"  Время: {elapsed:.1f}s")

        return {
            "symbol": symbol,
            "period": f"{start} -> {end}",
            "n_bars": len(df),
            "null_median_return_pct": round(null_median, 3),
            "buy_hold_return_pct": round(bh_ret, 3),
            "n_trials": total_tested,
            "results": results,
        }

    def run_all(self, start: str = None, end: str = None) -> dict:
        start = start or self.config.in_sample_start
        end = end or self.config.in_sample_end
        all_results = {}
        for symbol in self.config.symbols:
            try:
                all_results[symbol] = self.run_symbol(symbol, start, end)
            except FileNotFoundError as e:
                print(f"  {symbol}: ДАННЫЕ НЕ НАЙДЕНЫ — {e}")
            except Exception as e:
                import traceback
                print(f"  {symbol}: ОШИБКА — {e}")
                traceback.print_exc()
        return all_results


# =============================================================================
# Отчёт
# =============================================================================

def save_report(results: dict, out_dir: str = "reports"):
    """Сохранить результаты в CSV и JSON + топ-таблицу."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows = []
    for symbol, res in results.items():
        for r in res["results"]:
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, f"scan_{stamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(out_dir, f"scan_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Топ-таблица
    top = df.sort_values("return_pct", ascending=False).head(20)
    print(f"\n{'='*80}")
    print(f"  ТОП-20 по доходности (без DSR-поправки — для ориентира)")
    print(f"{'='*80}")
    for _, row in top.iterrows():
        print(f"  {row['symbol']:7s} {row['indicator']:22s} | ret={row['return_pct']:8.2f}% "
              f"n={row['n_trades']:4d} PF={row['profit_factor']:5.2f} "
              f"DD={row['max_dd_pct']:5.2f}% DSR={row['dsr']:.3f}({row['dsr_verdict']})")

    print(f"\n  Отчёт сохранён: {csv_path}")
    print(f"  JSON: {json_path}")
    return csv_path, json_path


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 64)
    print("  DeepSeek Research Pipeline v3 — полный перебор + DSR")
    print("  Данные: E:\\ИСТОРИЯ (M1->H1) | Брокер: Pipfarm cTrader")
    print("=" * 64)

    print(f"\nЗАФИКСИРОВАНО МЕТОДОВ:")
    for cat, methods in METHOD_MANIFEST.items():
        print(f"  {cat}: {len(methods)}")

    print(f"\nИНДИКАТОРОВ: {len(INDICATOR_REGISTRY)}")

    # Собрать и прогнать
    config = PipelineConfig(
        max_candidates_per_indicator=None,   # все комбинации
        min_trades=30,
        use_actual_spread=True,
    )
    pipeline = ResearchPipeline(config)

    # Прогнать in-sample (весь диапазон) — для первого реального отчёта
    results = pipeline.run_all()

    save_report(results, out_dir=os.path.join(os.path.dirname(__file__), "reports"))
