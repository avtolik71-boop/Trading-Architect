"""
Deflated Sharpe Ratio (DSR)
============================
Поправка на множественное тестирование и оверфиттинг.

Источник: Bailey, D. H. & López de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting, and Non-Normality".

Проблема: при переборе N конфигураций стратегий максимальный наблюдаемый
Sharpe ratio завышен пропорционально sqrt(2 ln N). Без поправки лучший
результат = оверфиттинг, а не эдж.

Решение (DSR):
  1. Вычислить haircut = ожидаемый максимум SR среди N независимых попыток
     при нулевой гипотезе (нулевая альфа).
  2. DSR = Ф( (SR - haircut) / SE(SR) ), где
     SE(SR) = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1) )
  3. DSR > 0.95 => значимо (вероятность оверфиттинга < 5%).

ВАЖНО (единицы): все величины — НА ПЕРИОД (не годовые). Годовая
доходность выводится отдельно как справочная метрика. Смешение
годового SR с per-period SE даёт неверный DSR.

Порог интерпретации (López de Prado):
  DSR >= 0.95  — стратегия статистически значима
  0.80–0.95    — пограничная
  < 0.80       — вероятный оверфиттинг
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# =============================================================================
# Базовые метрики (НА ПЕРИОД, не годовые)
# =============================================================================

def sharpe_per_period(returns: np.ndarray) -> float:
    """Sharpe ratio на один период (НЕ годовой)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd)


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Годовой Sharpe ratio (справочно)."""
    return sharpe_per_period(returns) * np.sqrt(periods_per_year)


def skewness(returns: np.ndarray) -> float:
    """Выборочная асимметрия (Fisher-Pearson, поправка на смещение)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return 0.0
    return float(stats.skew(r, bias=False))


def kurtosis(returns: np.ndarray) -> float:
    """Обычный эксцесс (нормальное распределение = 3)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 4:
        return 3.0
    # stats.kurtosis(fisher=True) даёт ИЗБЫТОЧНЫЙ эксцесс (норм = 0)
    return float(stats.kurtosis(r, fisher=True, bias=False)) + 3.0


# =============================================================================
# Expected Maximum Sharpe Ratio (haircut) — НА ПЕРИОД
# =============================================================================

def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """
    Ожидаемый максимум SR среди n_trials независимых попыток при нулевой
    гипотезе. Всё на период.

    Формула (Bailey & López de Prado, 2014):
      E[max SR_N] = sqrt(V[SR]) * [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    где V[SR] = 1/(T-1) при нулевой альфе (iid нормальные доходности).
    """
    if n_trials < 2:
        return 0.0
    if n_obs < 2:
        return 0.0
    gamma = EULER_MASCHERONI
    v_sr = 1.0 / (n_obs - 1)
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(v_sr) * ((1.0 - gamma) * q1 + gamma * q2))


# =============================================================================
# DSR (на период)
# =============================================================================

def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio — вероятность того, что наблюдаемый SR
    превышает haircut (ожидаемый максимум при переборе n_trials).

    Все вычисления на период; годовой SR выводится отдельно.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)

    if T < 2:
        return 0.0

    sr = sharpe_per_period(r)
    g3 = skewness(r)
    g4 = kurtosis(r)

    # Стандартная ошибка SR (per-period)
    se_sq = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / (T - 1)
    if se_sq <= 0:
        se_sq = abs(se_sq) + 1e-12
    se = np.sqrt(se_sq)

    # Haircut (per-period)
    haircut = expected_max_sharpe(n_trials, T)

    z = (sr - haircut) / se
    dsr = float(stats.norm.cdf(z))
    return max(0.0, min(1.0, dsr))


# =============================================================================
# Удобная обёртка для пайплайна
# =============================================================================

@dataclass
class DSRResult:
    """Полный результат DSR-анализа одной стратегии."""
    sharpe_per_period: float
    sharpe_annualized: float
    skew: float
    kurt: float
    n_obs: int
    n_trials: int
    haircut_sr: float       # per-period
    dsr: float
    verdict: str            # "SIGNIFICANT" / "MARGINAL" / "OVERFIT"
    probability_overfit: float  # 1 - DSR


def evaluate_dsr(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
    threshold: float = 0.95,
) -> DSRResult:
    """
    Полный DSR-анализ: вычисляет все метрики и вердикт.
    """
    r = np.asarray(returns, dtype=float)
    sr_pp = sharpe_per_period(r)
    sr_ann = sr_pp * np.sqrt(periods_per_year)
    g3 = skewness(r)
    g4 = kurtosis(r)
    T = len(r)

    haircut = expected_max_sharpe(n_trials, T) if n_trials >= 2 else 0.0
    dsr = deflated_sharpe_ratio(r, n_trials, periods_per_year)

    if dsr >= threshold:
        verdict = "SIGNIFICANT"
    elif dsr >= 0.80:
        verdict = "MARGINAL"
    else:
        verdict = "OVERFIT"

    return DSRResult(
        sharpe_per_period=sr_pp,
        sharpe_annualized=sr_ann,
        skew=g3,
        kurt=g4,
        n_obs=T,
        n_trials=n_trials,
        haircut_sr=haircut,
        dsr=dsr,
        verdict=verdict,
        probability_overfit=1.0 - dsr,
    )


# =============================================================================
# Демо / самопроверка
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    np.random.seed(42)

    print("=== Случай A: реальный эдж, 1 конфигурация ===")
    r_a = np.random.normal(0.001, 0.01, 500)
    res_a = evaluate_dsr(r_a, n_trials=1)
    print(f"  SR(год)={res_a.sharpe_annualized:.3f}, haircut={res_a.haircut_sr:.4f}, "
          f"DSR={res_a.dsr:.3f}, вердикт={res_a.verdict}")

    print("=== Случай B: нет эджа, 1000 конфигураций (оверфит) ===")
    # Средний DSR по многим нулевым стратегиям при N=1000
    dsrs = []
    for _ in range(200):
        r = np.random.normal(0, 0.01, 500)
        dsrs.append(evaluate_dsr(r, n_trials=1000).dsr)
    print(f"  Средний DSR нулевых при 1000 попыток: {np.mean(dsrs):.3f}")
    print(f"  Доля 'значимых' (ложные открытия): {np.mean([d > 0.95 for d in dsrs])*100:.1f}%")

    print("=== Случай C: реальный эдж, 1000 конфигураций ===")
    r_c = np.random.normal(0.001, 0.01, 500)
    res_c = evaluate_dsr(r_c, n_trials=1000)
    print(f"  SR(год)={res_c.sharpe_annualized:.3f}, haircut={res_c.haircut_sr:.4f}, "
          f"DSR={res_c.dsr:.3f}, вердикт={res_c.verdict}")

    print("=== Случай D: слабый эдж, 1000 конфигураций ===")
    r_d = np.random.normal(0.0002, 0.01, 500)
    res_d = evaluate_dsr(r_d, n_trials=1000)
    print(f"  SR(год)={res_d.sharpe_annualized:.3f}, haircut={res_d.haircut_sr:.4f}, "
          f"DSR={res_d.dsr:.3f}, вердикт={res_d.verdict}")

    print("\nИнтерпретация:")
    print("  A: DSR ~1.0 — сильный эдж БЕЗ перебора значим.")
    print("  B: 0% ложных открытий — haircut корректно штрафует перебор.")
    print("  C: эдж SR=1.6 годовых, но haircut при 1000 попытках = 2.3 годовых,")
    print("     значит найденный 'эдж' не превосходит лучший случайный — OVERFIT.")
    print("  D: слабый эдж + перебор = OVERFIT (честно).")
