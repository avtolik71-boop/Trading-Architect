"""
Walk-Forward валидация (OOS)
=============================
Честная out-of-sample проверка стратегий через скользящие окна.

Методология (из GENERATOR RUFP / STRATEGY_CATALOG.md):
  - train 18600 баров (~3 года H1) — прогрев для rolling-индикаторов
  - test 3100 баров (~6 месяцев H1) — OOS-оценка
  - embargo 48 баров (~2 дня) — зазор между train и test
  - 15 окон на конфиг (скользящее окно, шаг = test_bars)

Ворота OOS (прохождение = GO-кандидат):
  - t-статистика ≥ 2.0
  - Profit Factor ≥ 1.1
  - средняя доходность сделки > 0
  - n_trades ≥ 50 (агрегированно по всем окнам)
  - средняя доходность > медиана random_entry OOS (проверка «бета или тайминг»)

Отличие от in-sample: сигнал вычисляется ОДИН раз на полном ряду
(rolling-индикаторы не имеют параметров для переобучения), затем
нарезается по OOS-окнам. Это корректно для фиксированных индикаторов —
«train» здесь выполняет роль прогрева, а не подгонки параметров.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd


# =============================================================================
# Разбиение на окна
# =============================================================================

def walk_forward_splits(n: int, train_bars: int, test_bars: int,
                        embargo_bars: int = 0) -> list[tuple[int, int]]:
    """
    Сгенерировать OOS-окна как пары (test_start, test_end) индексов.

    train = [i - train_bars, i) — прогрев
    test  = [i + embargo, i + embargo + test_bars) — оценка
    Окно сдвигается на test_bars каждый шаг.
    """
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars и test_bars должны быть положительными")
    windows = []
    i = train_bars
    while i + embargo_bars + test_bars <= n:
        windows.append((i + embargo_bars, i + embargo_bars + test_bars))
        i += test_bars
    return windows


# =============================================================================
# Метрики OOS
# =============================================================================

@dataclass
class OOSResult:
    """Результат walk-forward валидации одной стратегии."""
    n_windows: int
    n_trades: int
    win_rate: float
    profit_factor: float
    mean_return_pct: float        # средняя доходность сделки (%)
    std_return_pct: float
    t_stat: float                 # t = mean/std * sqrt(n)
    total_return_pct: float       # суммарная OOS доходность
    windows_profitable: int       # сколько окон в плюс
    max_dd_pct: float
    beat_random: bool             # средняя > медиана random_entry OOS
    random_median_mean_pct: float
    passed_gates: bool
    gate_failures: list[str] = field(default_factory=list)


def t_statistic(returns: np.ndarray) -> float:
    """t-статистика средней доходности сделки."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(len(r)))


def aggregate_oos_metrics(
    trades: list,
    random_median_mean_pct: float,
    n_windows: int = 0,
    profitable_windows: int = 0,
    min_n: int = 50,
    min_t: float = 2.0,
    min_pf: float = 1.1,
) -> OOSResult:
    """
    Вычислить OOS-метрики из агрегированных сделок и прогнать через ворота.
    """
    if not trades:
        return OOSResult(
            n_windows=n_windows, n_trades=0, win_rate=0.0, profit_factor=0.0,
            mean_return_pct=0.0, std_return_pct=0.0, t_stat=0.0,
            total_return_pct=0.0, windows_profitable=profitable_windows,
            max_dd_pct=0.0, beat_random=False,
            random_median_mean_pct=random_median_mean_pct,
            passed_gates=False, gate_failures=["no_trades"],
        )

    rets = np.array([t.return_pct for t in trades], dtype=float)  # % от номинала
    n = len(rets)
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    win_rate = len(wins) / n * 100
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    mean_r = float(rets.mean())
    std_r = float(rets.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(mean_r / std_r * np.sqrt(n)) if std_r > 0 else 0.0

    # Эквити-кривая из агрегированных сделок (по времени выхода)
    sorted_trades = sorted(trades, key=lambda tr: tr.exit_time)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for tr in sorted_trades:
        equity *= (1.0 + tr.return_pct / 100.0)
        peak = max(peak, equity)
        dd = (equity / peak - 1.0) * 100
        max_dd = min(max_dd, dd)

    total_return = (equity - 1.0) * 100

    # Проверка ворот
    failures = []
    if n < min_n:
        failures.append(f"n<{min_n}")
    if t_stat < min_t:
        failures.append(f"t<{min_t}")
    if pf < min_pf:
        failures.append(f"PF<{min_pf}")
    if mean_r <= 0:
        failures.append("mean<=0")
    if mean_r <= random_median_mean_pct:
        failures.append("mean<=random")

    passed = len(failures) == 0

    return OOSResult(
        n_windows=n_windows,
        n_trades=n,
        win_rate=round(win_rate, 2),
        profit_factor=round(pf, 3) if pf != float("inf") else 999.0,
        mean_return_pct=round(mean_r, 4),
        std_return_pct=round(std_r, 4),
        t_stat=round(t_stat, 3),
        total_return_pct=round(total_return, 2),
        windows_profitable=profitable_windows,
        max_dd_pct=round(abs(max_dd), 2),
        beat_random=mean_r > random_median_mean_pct,
        random_median_mean_pct=round(random_median_mean_pct, 4),
        passed_gates=passed,
        gate_failures=failures,
    )


# =============================================================================
# Walk-forward движок
# =============================================================================

class WalkForwardEngine:
    """
    Прогон стратегии через скользящие OOS-окна.

    Параметры (из RUFP):
      train_bars=18600, test_bars=3100, embargo_bars=48
    """

    def __init__(
        self,
        backtester,
        train_bars: int = 18600,
        test_bars: int = 3100,
        embargo_bars: int = 48,
    ):
        self.backtester = backtester
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.embargo_bars = embargo_bars

    def run(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        symbol: str,
        sl_pips: float,
        tp_pips: float,
        use_actual_spread: bool = True,
    ) -> tuple[list, int, int]:
        """
        Прогнать сигнал через все OOS-окна.
        Возвращает (все сделки, число окон, число прибыльных окон).

        Сигнал вычисляется на полном df заранее; здесь только нарезка окон.
        """
        n = len(df)
        windows = walk_forward_splits(n, self.train_bars, self.test_bars, self.embargo_bars)
        all_trades = []
        profitable_windows = 0
        for test_start, test_end in windows:
            test_df = df.iloc[test_start:test_end]
            test_signal = signal.iloc[test_start:test_end]
            trades, equity = self.backtester.run(
                test_df, test_signal, symbol,
                sl_pips=sl_pips, tp_pips=tp_pips,
                use_actual_spread=use_actual_spread,
            )
            all_trades.extend(trades)
            # Окно прибыльное, если эквити выросло
            if len(equity) > 0 and equity.iloc[-1] > 1.0:
                profitable_windows += 1
        return all_trades, len(windows), profitable_windows

    def run_random_baseline(
        self,
        df: pd.DataFrame,
        symbol: str,
        sl_pips: float,
        tp_pips: float,
        pip_size: float,
        n_seeds: int = 30,
        prob: float = 0.002,
        use_actual_spread: bool = True,
    ) -> float:
        """
        Медианная средняя OOS-доходность случайного входа (базовая линия).
        """
        rng = np.random.default_rng(7)
        means = []
        for seed in range(n_seeds):
            fire = rng.random(len(df)) < prob
            direction = rng.integers(0, 2, len(df)) * 2 - 1
            sig = pd.Series(np.where(fire, direction, 0), index=df.index, dtype=int)
            trades, n_windows, _ = self.run(df, sig, symbol, sl_pips, tp_pips, use_actual_spread)
            if trades:
                means.append(float(np.mean([t.return_pct for t in trades])))
        if not means:
            return 0.0
        return float(np.median(means))
