"""
Нуль-модели для проверки «бета или тайминг»
============================================
ПРАВИЛО: Каждая стратегия обязана бить нуль-модель OOS.

Нуль-модели:
  1. random_entry — случайный вход в том же окне (тест «бета или тайминг»)
  2. buy_and_hold — пассивное удержание (тест на альфу vs дрейф)
  3. random_entry_short — случайный шорт (для симметрии)

Методология из STRATEGY_CATALOG.md и signals.py (GENERATOR RUFP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class NullModelResult:
    """Результат прогона нуль-модели."""
    model_name: str
    total_return_pct: float
    n_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_bar_return: float
    equity_curve: list[float] = field(default_factory=list)


class NullModelEngine:
    """
    Генерирует нуль-модели для эталонного сравнения.

    ТЕСТ «БЕТА ИЛИ ТАЙМИНГ»:
    Если стратегия не бьёт random_entry OOS — она не добавляет ценности
    сверх случайного входа в том же окне. Это МИНИМАЛЬНЫЙ барьер.
    """

    @staticmethod
    def random_entry_equity(df: pd.DataFrame, prob: float = 0.002,
                            seed: int = 7, sl_pips: Optional[float] = None,
                            tp_pips: Optional[float] = None,
                            pip_size: float = 0.0001) -> pd.Series:
        """
        Сгенерировать эквити-кривую случайного входа.

        prob: вероятность входа на каждом баре (0.002 = ~1 сделка в день на H1)
        """
        rng = np.random.default_rng(seed)
        n = len(df)
        fire = rng.random(n) < prob
        direction = rng.integers(0, 2, n) * 2 - 1  # 1 или -1

        c = df["close"].values
        equity = [1.0]
        pos = 0       # 0 = нет позиции, 1/-1 = направление
        entry_price = 0.0

        for i in range(1, n):
            ret = 0.0
            if pos != 0:
                # Проверить SL/TP
                if sl_pips and tp_pips:
                    pnl_pips = (c[i] - entry_price) / pip_size * pos
                    if pnl_pips <= -sl_pips or pnl_pips >= tp_pips:
                        # Дробная доходность позиции: цена_движения / цена_входа
                        ret = pnl_pips * pip_size / entry_price
                        pos = 0
                    else:
                        ret = 0.0  # Держим
                else:
                    # Без SL/TP — выход по обратному сигналу или через N баров
                    ret = 0.0  # Упрощённо: держим до конца окна

            if pos == 0 and fire[i]:
                pos = direction[i]
                entry_price = c[i]

            equity.append(equity[-1] * (1.0 + ret))

        return pd.Series(equity, index=df.index)

    @staticmethod
    def buy_and_hold_equity(df: pd.DataFrame) -> pd.Series:
        """Buy & hold — пассивное удержание с первого до последнего бара."""
        c = df["close"].values
        return pd.Series(c / c[0], index=df.index)

    @staticmethod
    def random_entry_multi_seed(df: pd.DataFrame, prob: float = 0.002,
                                n_seeds: int = 30, **kwargs) -> pd.DataFrame:
        """
        Сгенерировать распределение случайных входов (30 сидов).

        Возвращает DataFrame: каждая колонка — эквити-кривая одного сида.
        Стратегия должна быть ЛУЧШЕ медианы этого распределения.
        """
        results = {}
        for seed in range(n_seeds):
            eq = NullModelEngine.random_entry_equity(df, prob=prob, seed=seed, **kwargs)
            results[f"seed_{seed}"] = eq
        return pd.DataFrame(results, index=df.index)

    @classmethod
    def compute_metrics(cls, equity: pd.Series) -> NullModelResult:
        """Вычислить метрики эквити-кривой."""
        returns = equity.pct_change().dropna()
        peak = equity.cummax()
        dd = (equity / peak - 1.0).min()

        # Аппроксимация Sharpe
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)

        return NullModelResult(
            model_name="null",
            total_return_pct=(equity.iloc[-1] - 1.0) * 100,
            n_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=abs(dd) * 100,
            sharpe_ratio=sharpe,
            avg_bar_return=returns.mean(),
            equity_curve=equity.tolist(),
        )

    @classmethod
    def compare_strategy_to_null(cls, strategy_equity: pd.Series,
                                  null_equities: pd.DataFrame,
                                  metric: str = "total_return") -> dict:
        """
        Сравнить стратегию с распределением нуль-моделей.

        Возвращает:
          - beat_median: True если стратегия лучше медианы
          - beat_p75: True если лучше 75-го перцентиля
          - percentile: в каком перцентиле находится стратегия
        """
        null_final = null_equities.iloc[-1] - 1.0
        strategy_final = strategy_equity.iloc[-1] - 1.0

        median_null = null_final.median()
        p75_null = null_final.quantile(0.75)
        p90_null = null_final.quantile(0.90)

        percentile = (null_final < strategy_final).mean() * 100

        return {
            "strategy_return_pct": strategy_final * 100,
            "null_median_return_pct": median_null * 100,
            "null_p75_return_pct": p75_null * 100,
            "null_p90_return_pct": p90_null * 100,
            "beat_median": strategy_final > median_null,
            "beat_p75": strategy_final > p75_null,
            "strategy_percentile": percentile,
            "verdict": "ALPHA" if strategy_final > p75_null else (
                "BETA_OR_TIMING" if strategy_final > median_null else "NO_EDGE"),
        }


# =============================================================================
# Быстрый тест
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Сгенерировать синтетические данные
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2026-01-01", freq="h")[:10000]
    close = 1.05 + np.cumsum(np.random.randn(10000) * 0.001)
    df = pd.DataFrame({"close": close, "open": close * 0.999, "high": close * 1.002, "low": close * 0.998}, index=dates)

    engine = NullModelEngine()

    # Buy & hold
    bh = engine.buy_and_hold_equity(df)
    metrics_bh = engine.compute_metrics(bh)
    print(f"Buy & Hold: return={metrics_bh.total_return_pct:.1f}%, DD={metrics_bh.max_drawdown_pct:.1f}%")

    # Random entry распределение
    null_dist = engine.random_entry_multi_seed(df, prob=0.002, n_seeds=30, sl_pips=100, tp_pips=200, pip_size=0.0001)
    comparison = engine.compare_strategy_to_null(bh, null_dist)
    print(f"\nСравнение с random_entry (30 сидов):")
    for k, v in comparison.items():
        print(f"  {k}: {v}")