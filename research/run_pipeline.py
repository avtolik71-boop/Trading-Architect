"""
Главный оркестратор пайплайна перебора стратегий
=================================================
Запуск полного цикла исследования:
  1. Загрузить данные из E:\ИСТОРИЯ (Parquet)
  2. Перебрать индикаторы из пополняемой библиотеки
  3. Применить издержки (спред + комиссия + свопы)
  4. Прогнать через нуль-модели (random_entry + buy&hold)
  5. Проверить риск-профили (PIPFARM + FREE)
  6. Сгенерировать отчёт

ДАННЫЕ: E:\ИСТОРИЯ (BTCUSD, GBPUSD, US100, USDJPY, XAUUSD)
БРОКЕР: Pipfarm cTrader / Tickmill
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Подключаем локальные модули
sys.path.insert(0, str(Path(__file__).resolve().parent))

from indicators.indicator_registry import (
    INDICATOR_REGISTRY, IndicatorEnumerator, enumerate_all_indicators,
    CandidateConfig, METHOD_MANIFEST,
)
from costs.cost_engine import CostEngine, ACTIVE_COST_SET, PIPFARM_COSTS, TICKMILL_COSTS
from testing.null_models import NullModelEngine
from risk_mgmt.risk_profiles import (
    RiskProfileEngine, PIPFARM_PROFILES, FREE_PROFILE, ALL_PROFILES,
    profile_gate_pass, compute_total_drawdown, compute_max_daily_drawdown,
)
from catalogs.idea_strategy_registry import (
    UnifiedRegistry, IdeasRegistry, StrategyRegistry,
    IdeaEntry, StrategyEntry, IdeaStatus,
)


# =============================================================================
# Конфигурация
# =============================================================================

class PipelineConfig:
    """Глобальная конфигурация пайплайна."""

    # Данные
    history_root: str = r"E:\ИСТОРИЯ"
    symbols: list[str] = None  # None = все доступные
    timeframes: list[str] = None

    # Периоды
    in_sample_start: str = "2016-01-01"
    in_sample_end: str = "2020-12-31"
    out_sample_start: str = "2021-01-01"
    out_sample_end: str = "2026-12-31"

    # Издержки
    cost_set: str = "PIPFARM_CTRADER"  # или "TICKMILL"

    # Профили
    active_profiles: list[str] = None  # None = все

    # Нуль-модели
    null_seeds: int = 30
    null_prob: float = 0.002

    # Фильтры и выходы
    filters: list[str] = None  # None = ["none"]
    exits: list[str] = None    # None = ["fixed_sl_tp"]

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["XAUUSD", "GBPUSD", "USDJPY", "US100", "BTCUSD"]
        if self.timeframes is None:
            self.timeframes = ["H1"]
        if self.active_profiles is None:
            self.active_profiles = list(ALL_PROFILES.keys())
        if self.filters is None:
            self.filters = ["none"]
        if self.exits is None:
            self.exits = ["fixed_sl_tp"]


# =============================================================================
# Загрузчик данных
# =============================================================================

class DataLoader:
    """Загрузка исторических данных из E:\ИСТОРИЯ (Parquet)."""

    def __init__(self, root: str = r"E:\ИСТОРИЯ"):
        self.root = Path(root)
        self._cache: dict[tuple, pd.DataFrame] = {}

    def available_symbols(self) -> list[str]:
        """Список доступных символов."""
        return [d.name for d in self.root.iterdir() if d.is_dir() and d.name != "__pycache__"]

    def load(self, symbol: str, start: str = "2016-01-01",
             end: str = "2026-12-31") -> pd.DataFrame:
        """
        Загрузить данные символа за период.

        Данные хранятся как {symbol}/{year}.parquet.
        """
        cache_key = (symbol, start, end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        symbol_dir = self.root / symbol
        if not symbol_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {symbol_dir}")

        frames = []
        start_year = int(start[:4])
        end_year = int(end[:4])

        for year in range(start_year, end_year + 1):
            parquet_path = symbol_dir / f"{year}.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                df = df.sort_index()
                # Колонки должны быть: open, high, low, close, tick_volume, spread
                frames.append(df)

        if not frames:
            raise FileNotFoundError(f"Нет данных для {symbol} за {start}-{end}")

        result = pd.concat(frames)
        result = result[~result.index.duplicated(keep="first")]
        result = result.loc[start:end].copy()

        self._cache[cache_key] = result
        return result

    def load_all(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Загрузить все символы."""
        return {s: self.load(s, start, end) for s in symbols}


# =============================================================================
# Простой бэктестер (сигнал по close, вход по open следующего бара)
# =============================================================================

class SimpleBacktester:
    """
    Бэктестер: сигнал по close[i-1] → вход по open[i].
    Это модель GENERATOR RUFP (constructor.py) — стандарт для генератора.
    """

    def __init__(self, cost_engine: CostEngine):
        self.cost_engine = cost_engine

    def run(self, df: pd.DataFrame, signal: pd.Series,
            sl_pips: Optional[float] = None, tp_pips: Optional[float] = None,
            pip_size: float = 0.0001, symbol: str = "", profile_engine=None) -> pd.DataFrame:
        """
        Прогнать сигнал через бэктестер.

        Возвращает DataFrame с колонками: equity, position, trade_open, trade_close.
        """
        n = len(df)
        c = df["close"].values
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values

        equity = np.ones(n) * 10000.0  # Стартовый капитал $10k
        position = np.zeros(n, dtype=int)
        entry_price = np.zeros(n)
        entry_idx = np.zeros(n, dtype=int)
        trades_log = []

        current_pos = 0
        current_entry = 0.0
        current_entry_i = 0

        for i in range(1, n):
            # Проверка риск-профиля (если активен)
            if profile_engine:
                ts = df.index[i]
                actions = profile_engine.check(ts, equity[i - 1])
                if any("FLATTEN" in a["action"] for a in actions):
                    if current_pos != 0:
                        # Закрыть позицию принудительно
                        pnl_pips = (c[i] - current_entry) / pip_size * current_pos
                        # Применить издержки выхода
                        exit_costs = self.cost_engine.apply_exit_costs(
                            symbol, "BUY" if current_pos > 0 else "SELL",
                            0.1, c[i], df.index[i],
                        )
                        equity[i] = equity[i - 1] * (1.0 + pnl_pips * pip_size / equity[i - 1] * 100)
                        trades_log.append({
                            "entry_time": df.index[current_entry_i],
                            "exit_time": df.index[i],
                            "side": "LONG" if current_pos > 0 else "SHORT",
                            "entry": current_entry,
                            "exit": c[i],
                            "pnl_pips": pnl_pips,
                            "reason": "profile_trigger",
                        })
                        current_pos = 0
                        continue

            # Сигнал с предыдущего бара → вход по open текущего
            sig = signal.iloc[i - 1]
            pos_prev = current_pos

            # Логика входа
            if current_pos == 0 and sig != 0 and (profile_engine is None or profile_engine.allow_entry()):
                current_pos = sig
                current_entry = o[i]
                current_entry_i = i
                position[i] = current_pos
                entry_price[i] = current_entry
                equity[i] = equity[i - 1]  # Вход по open — equity не меняется до закрытия
                continue

            # Удержание позиции — проверка SL/TP
            if current_pos != 0:
                pnl_pips = (c[i] - current_entry) / pip_size * current_pos
                close_reason = None

                if sl_pips and pnl_pips <= -sl_pips:
                    close_reason = "stop_loss"
                elif tp_pips and pnl_pips >= tp_pips:
                    close_reason = "take_profit"

                if close_reason:
                    # Применить издержки
                    cost_result = self.cost_engine.compute_total_trade_cost(
                        symbol, "BUY" if current_pos > 0 else "SELL",
                        0.1, current_entry, c[i],
                        df.index[current_entry_i], df.index[i],
                    )
                    ret = cost_result["net_pnl"] / equity[current_entry_i]
                    equity[i] = equity[i - 1] * (1.0 + ret)
                    trades_log.append({
                        "entry_time": df.index[current_entry_i],
                        "exit_time": df.index[i],
                        "side": "LONG" if current_pos > 0 else "SHORT",
                        "entry": current_entry,
                        "exit": c[i],
                        "pnl_pips": pnl_pips,
                        "net_pnl_usd": cost_result["net_pnl"],
                        "reason": close_reason,
                    })
                    current_pos = 0
                else:
                    equity[i] = equity[i - 1]  # Держим позицию
            else:
                equity[i] = equity[i - 1]

            position[i] = current_pos

        equity_series = pd.Series(equity, index=df.index, name="equity")
        return pd.DataFrame({
            "equity": equity_series,
            "position": pd.Series(position, index=df.index),
            "close": df["close"],
        }, index=df.index)


# =============================================================================
# Оркестратор полного пайплайна
# =============================================================================

class ResearchPipeline:
    """
    Полный пайплайн исследования:
    Загрузка → перебор индикаторов → бэктест → нуль-модели → риск-профили → отчёт.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.loader = DataLoader(self.config.history_root)

        cost_set = PIPFARM_COSTS if self.config.cost_set == "PIPFARM_CTRADER" else TICKMILL_COSTS
        # Instrument config from GENERATOR RUFP
        instruments = {
            "XAUUSD": {"pip_size": 0.1, "contract_size": 100, "lot_size": 100},
            "GBPUSD": {"pip_size": 0.0001, "contract_size": 100000, "lot_size": 100000},
            "USDJPY": {"pip_size": 0.01, "contract_size": 100000, "lot_size": 100000},
            "US100": {"pip_size": 1.0, "contract_size": 1, "lot_size": 1},
            "BTCUSD": {"pip_size": 1.0, "contract_size": 1, "lot_size": 1},
        }
        self.cost_engine = CostEngine(cost_set, instruments)
        self.backtester = SimpleBacktester(self.cost_engine)
        self.null_engine = NullModelEngine()

    def run_symbol(self, symbol: str, start: str, end: str) -> dict:
        """Запустить полный пайплайн для одного символа."""
        print(f"\n{'='*60}")
        print(f"  {symbol} | {start} → {end} | Брокер: {self.config.cost_set}")
        print(f"{'='*60}")

        # 1. Загрузка данных
        df = self.loader.load(symbol, start, end)
        print(f"  Загружено: {len(df)} баров, {df.index[0].date()} → {df.index[-1].date()}")

        # 2. Перебор индикаторов
        all_candidates = enumerate_all_indicators(self.config.filters, self.config.exits)
        total = sum(len(v) for v in all_candidates.values())
        print(f"  Кандидатов: {total} ({len(all_candidates)} индикаторов)")

        # 3. Нуль-модели
        pip_size = 0.0001
        if symbol == "XAUUSD":
            pip_size = 0.1
        elif symbol in ("US100", "BTCUSD"):
            pip_size = 1.0
        elif symbol == "USDJPY":
            pip_size = 0.01

        null_dist = self.null_engine.random_entry_multi_seed(
            df, prob=self.config.null_prob, n_seeds=self.config.null_seeds,
            sl_pips=100, tp_pips=200, pip_size=pip_size,
        )
        null_final_returns = null_dist.iloc[-1] - 1.0
        median_null = null_final_returns.median() * 100
        print(f"  Нуль-модель (медиана random_entry, {self.config.null_seeds} сидов): {median_null:.2f}%")

        buy_hold = self.null_engine.buy_and_hold_equity(df)
        bh_return = (buy_hold.iloc[-1] - 1.0) * 100
        print(f"  Buy & Hold: {bh_return:.2f}%")

        # 4. Прогон каждого индикатора (демо: только первый для скорости)
        results = []
        for ind_name, candidates in all_candidates.items():
            ind = INDICATOR_REGISTRY[ind_name]
            print(f"\n  Индикатор: {ind.display_name} ({len(candidates)} конфигов)")

            # Демо: прогоняем только первую комбинацию параметров
            for cfg in candidates[:1]:
                try:
                    signal = ind.signal_fn(df, **cfg.params)
                    bt_result = self.backtester.run(
                        df, signal, sl_pips=100, tp_pips=200,
                        pip_size=pip_size, symbol=symbol,
                    )
                    equity = bt_result["equity"]
                    final_return = (equity.iloc[-1] - 1.0) * 100

                    # Сравнение с нуль-моделью
                    comparison = self.null_engine.compare_strategy_to_null(equity, null_dist)

                    # Проверка риск-профилей
                    profile_results = {}
                    for pname, pcfg in ALL_PROFILES.items():
                        gate = profile_gate_pass(equity, pcfg)
                        profile_results[pname] = gate

                    results.append({
                        "indicator": ind_name,
                        "params": cfg.params,
                        "return_pct": final_return,
                        "vs_null": comparison["verdict"],
                        "profiles": profile_results,
                    })

                    print(f"    {cfg.label}: return={final_return:.2f}%, "
                          f"vs_null={comparison['verdict']}")

                except Exception as exc:
                    print(f"    {cfg.label}: ОШИБКА — {exc}")

        return {
            "symbol": symbol,
            "period": f"{start} → {end}",
            "n_bars": len(df),
            "null_median_return_pct": median_null,
            "buy_hold_return_pct": bh_return,
            "results": results,
        }

    def run_all(self):
        """Запустить пайплайн для всех символов."""
        all_results = {}
        for symbol in self.config.symbols:
            try:
                in_sample = self.run_symbol(
                    symbol, self.config.in_sample_start, self.config.in_sample_end,
                )
                all_results[f"{symbol}_IS"] = in_sample
            except FileNotFoundError as e:
                print(f"  {symbol}: ДАННЫЕ НЕ НАЙДЕНЫ — {e}")
            except Exception as e:
                print(f"  {symbol}: ОШИБКА — {e}")

        return all_results


# =============================================================================
# Точка входа
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DeepSeek Research Pipeline - Перебор стратегий")
    print("  Данные: E:\\ИСТОРИЯ | Брокер: Pipfarm cTrader")
    print("=" * 60)

    # Сводка существующих методов
    print("\nЗАФИКСИРОВАНО МЕТОДОВ (не дублировать):")
    for cat, methods in METHOD_MANIFEST.items():
        print(f"  {cat}: {len(methods)} - {', '.join(list(methods.keys())[:5])}")

    # Индикаторы в реестре
    print(f"\nИНДИКАТОРОВ В РЕЕСТРЕ: {len(INDICATOR_REGISTRY)}")
    for name, ind in INDICATOR_REGISTRY.items():
        combos = 1
        for v in ind.param_grid.values():
            combos *= len(v)
        print(f"  {name:30s} | {ind.category:15s} | {combos:4d} комбинаций | {ind.source}")

    # Реестр идей и стратегий
    reg = UnifiedRegistry()
    reg.seed()
    ideas_summary = reg.ideas.summary()
    strats_summary = reg.strategies.summary()
    print(f"\nРЕЕСТР ИДЕЙ: {ideas_summary['total']}")
    for status, count in ideas_summary['by_status'].items():
        if count:
            print(f"  {status}: {count}")
    print(f"\nРЕЕСТР СТРАТЕГИЙ: {strats_summary['total']}")
    print(f"  GO-кандидатов: {strats_summary['go_candidates']}")
    print(reg.go_candidates_report())

    # Запуск пайплайна
    config = PipelineConfig()
    pipeline = ResearchPipeline(config)

    try:
        results = pipeline.run_all()
        print(f"\nГОТОВО. Символов обработано: {len(results)}")
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")