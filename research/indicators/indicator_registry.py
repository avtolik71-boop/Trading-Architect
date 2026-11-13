"""
Реестр индикаторов — пополняемая библиотека БАЗА ЗНАНИЙ/indicators/
====================================================================
Каждый индикатор регистрируется один раз и автоматически разворачивается
в семейство кандидатов через пайплайн перебора.

Принцип: ОДИН индикатор → СЕТКА параметров → СЕМЕЙСТВО кандидатов.
Новые индикаторы добавляются в INDICATOR_REGISTRY — никакой другой
код менять не нужно. Пайплайн сам подхватит.

Архитектура заимствована из GENERATOR RUFP (signals.py TEMPLATE_FUNCS)
и расширена: индикатор → сигнал → фильтр → выход → профиль.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Базовые типы
# =============================================================================

class SignalDirection(int, Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class AssetClass(str, Enum):
    FX = "fx"
    METAL = "metal"
    INDEX = "index"
    CRYPTO = "crypto"
    COMMODITY = "commodity"


@dataclass
class IndicatorDef:
    """Определение одного индикатора — единица пополнения библиотеки."""
    name: str                                    # Уникальное имя: "rsi", "macd", "donchian"
    display_name: str                            # Человеческое: "RSI (Relative Strength Index)"
    category: str                                # "momentum", "trend", "volatility", "volume", "mean_reversion", "cycle"
    signal_fn: Callable[..., pd.Series]          # Функция: (df, **params) → pd.Series[int] (-1,0,1)
    param_grid: dict[str, list[Any]]             # Сетка параметров: {"period": [7,14,21], "threshold": [30,70]}
    compatible_asset_classes: list[AssetClass]    # На каких классах активов работает
    min_timeframe: str = "M5"                    # Минимальный ТФ
    source: str = ""                             # Откуда индикатор: "RUFP", "MQL5", "TradingView", "manual"
    requires_ohlc: bool = True                   # Требует OHLC или только close
    description: str = ""                        # Краткое описание логики


# =============================================================================
# СУЩЕСТВУЮЩИЕ МЕТОДЫ (зафиксировано — НЕ ДУБЛИРОВАТЬ)
# =============================================================================

EXISTING_SIGNAL_TEMPLATES = {
    # Из GENERATOR RUFP / generator/signals.py — ЗАФИКСИРОВАНО
    "donchian":           "Пробой N-дневного максимума/минимума (close > high_max / close < low_min)",
    "ema_cross":          "Пересечение двух EMA (fast > slow → long, fast < slow → short)",
    "rsi":                "RSI выход из зон перекупленности/перепроданности (oversold→long, overbought→short)",
    "squeeze":            "BB-сжатие → пробой волатильности (сжатие BB внутрь KC, выход → сигнал)",
    "vol_breakout":       "Пробой волатильности (close > open ± k×ATR), опц. NR7-фильтр",
    "adaptive_donchian":  "Адаптивный Дончиан: длина канала = const/ADX, clamp(1,40)",
    "random_entry":       "Случайный вход (50/50) — ЭТАЛОННАЯ БАЗА для проверки «бета или тайминг»",

    # Из каталога стратегий STRATEGY_CATALOG.md — ЗАФИКСИРОВАНО
    "bb_fade":            "Пробой BB(35,2) → возврат внутрь → fade (против пробоя)",
    "cci_zero":           "Пересечение CCI нуля → сигнал; выход при ±100",
    "macd_pullback_mtf":  "Мульти-ТФ MACD откат: D1 тренд, H4 откат, вход при развороте",
    "fractal_breakout":   "Фрактал (5 баров) выше/ниже Аллигатора → пробой на 1 пипс",
    "adx_di_cross":       "ADX растёт + DI пересекает порог → вход в сторону DI",
    "pivot_daily":        "Дневные пивоты: fade на R1/S1 или breakout с подтверждением",
    "elder_triple":       "Тройной экран Элдера: старший ТФ MACD, средний RSI",
    "trend_pullback_adx": "Holy Grail: ADX растёт, откат к EMA(20), закрытие в сторону тренда",
    "prior_day_breakout": "Свинг по пред. дню: 3-5 дней против тренда, пробой high/low",

    # Индикаторные функции, УЖЕ реализованные в signals.py
    "ema":                "EMA — вспомогательная (не сигнал, а строительный блок)",
    "atr_series":         "ATR — вспомогательная (не сигнал, а строительный блок)",
    "adx_series":         "ADX — вспомогательная (не сигнал, а строительный блок)",
    "rsi_series":         "RSI — вспомогательная (не сигнал, а строительный блок)",
}

EXISTING_FILTERS = {
    # Из GENERATOR RUFP / generator/filters.py — ЗАФИКСИРОВАНО
    "none":               "Без фильтра",
    "trend_ema200":       "Трендовый: цена > EMA(200) → только long, < EMA(200) → только short",
    "session":            "Сессионный: только в указанные часы (напр. 7-20 UTC)",
    "adx_min":            "Минимальный ADX: сигнал только при ADX > порога",
    "nr7":                "NR7: только если предыдущий бар — наименьший диапазон за 7",
    "volume_min":         "Минимальный объём: сигнал только при volume > порога",
    "day_of_week":        "День недели: исключить понедельник/пятницу",
}

EXISTING_EXITS = {
    # Из GENERATOR RUFP — ЗАФИКСИРОВАНО
    "fixed_sl_tp":        "Фиксированные SL (пипсы) и TP (пипсы)",
    "trailing":            "Трейлинг-стоп: k × ATR от экстремума",
    "chandelier":         "Люстра (chandelier): highest high − k×ATR",
    "time_exit":          "Выход через N баров независимо от цены",
    "reverse_signal":     "Выход по противоположному сигналу",
    "no_exit":            "Без выхода (до конца окна)",
}

EXISTING_RISK_PROFILES = {
    # Из GENERATOR RUFP / risk_engine/profiles.py — ЗАФИКСИРОВАНО
    "PIPFARM(target=0.3%)": "Дневной стоп 2%, дневная цель 0.3%, макс. общая DD 8%",
    "PIPFARM(target=0.5%)": "Дневной стоп 2%, дневная цель 0.5%, макс. общая DD 8%",
    "PIPFARM(target=0.8%)": "Дневной стоп 2%, дневная цель 0.8%, макс. общая DD 8%",
    "PIPFARM(target=1.0%)": "Дневной стоп 2%, дневная цель 1.0%, макс. общая DD 8%",
    "FREE":                 "Только общая просадка <= 15%, без дневных стопов",
    "CLASSIC":              "Дневной убыток <= 2% + общая DD <= 15%. Прибыль по правилам бота (без дневной цели).",
}

EXISTING_TESTING_METHODS = {
    # Методы тестирования — ЗАФИКСИРОВАНО
    "walk_forward":        "Walk-forward: train 18600 / test 3100 / embargo 48, 15 окон",
    "simple_split":        "Простое разделение: 2016–2020 in-sample, 2021–2026 out-of-sample",
    "monte_carlo":         "Монте-Карло: перемешивание сделок, 1000 симуляций",
    "random_entry_baseline":"Случайный вход как нуль-модель (random_entry сигнал)",
    "buy_and_hold":        "Buy & hold как базовая линия (дрейф инструмента)",
    "dsr_penalty":         "Deflated Sharpe Ratio — поправка на перебор гипотез",
}

EXISTING_COST_COMPONENTS = {
    # Из GENERATOR RUFP / costs/model.py — ЗАФИКСИРОВАНО
    "spread":              "Брокерский спред (из costs.csv, перезаписывает фид)",
    "commission_per_lot":  "Комиссия за лот (round-turn: true/false)",
    "swap_long":           "Своп лонг (пункты/день), triple в среду",
    "swap_short":          "Своп шорт (пункты/день), triple в среду",
    "slippage":            "Проскальзывание (фикс. пипсы или % от спреда)",
}

# =============================================================================
# СВОДНЫЙ МАНИФЕСТ СУЩЕСТВУЮЩИХ МЕТОДОВ
# =============================================================================

METHOD_MANIFEST = {
    "signals": EXISTING_SIGNAL_TEMPLATES,
    "filters": EXISTING_FILTERS,
    "exits": EXISTING_EXITS,
    "risk_profiles": EXISTING_RISK_PROFILES,
    "testing": EXISTING_TESTING_METHODS,
    "costs": EXISTING_COST_COMPONENTS,
}

# =============================================================================
# РЕЕСТР ИНДИКАТОРОВ — пополняемая библиотека
# =============================================================================

INDICATOR_REGISTRY: dict[str, IndicatorDef] = {}

def register_indicator(ind: IndicatorDef) -> None:
    """Зарегистрировать индикатор в библиотеке. Дубликаты отвергаются."""
    if ind.name in INDICATOR_REGISTRY:
        raise ValueError(f"Индикатор '{ind.name}' уже зарегистрирован. "
                         f"Существующий: {INDICATOR_REGISTRY[ind.name].display_name}")
    INDICATOR_REGISTRY[ind.name] = ind


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ИНДИКАТОРОВ
# =============================================================================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def _adx(df: pd.DataFrame, length: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1.0 / length, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1.0 / length, adjust=False).mean() / atr_s
    ndi = 100 * ndm.ewm(alpha=1.0 / length, adjust=False).mean() / atr_s
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / length, adjust=False).mean()

def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram}, index=close.index)

def _bb(close: pd.Series, period: int, n_std: float) -> pd.DataFrame:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return pd.DataFrame({"mid": ma, "upper": ma + n_std * sd, "lower": ma - n_std * sd}, index=close.index)

def _stoch(df: pd.DataFrame, k_period: int, d_period: int) -> pd.DataFrame:
    lo = df["low"].rolling(k_period).min()
    hi = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lo) / (hi - lo + 1e-9)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d}, index=df.index)

def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)


# =============================================================================
# СИГНАЛЬНЫЕ ФУНКЦИИ — каждая разворачивается в семейство кандидатов
# =============================================================================

def donchian_signal(df: pd.DataFrame, n: int) -> pd.Series:
    """Пробой Дончиан-канала: close > max_high(n) → long, close < min_low(n) → short."""
    c = df["close"]
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[c > hi] = 1
    sig[c < lo] = -1
    return sig


def ema_cross_signal(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Пересечение EMA: fast > slow → long, fast < slow → short."""
    f = _ema(df["close"], fast)
    s = _ema(df["close"], slow)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(f > s) & (f.shift(1) <= s.shift(1))] = 1
    sig[(f < s) & (f.shift(1) >= s.shift(1))] = -1
    return sig


def rsi_signal(df: pd.DataFrame, period: int, oversold: float, overbought: float) -> pd.Series:
    """RSI: выход из oversold → long, выход из overbought → short."""
    rsi = _rsi(df["close"], period)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
    sig[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
    return sig


def macd_signal(df: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.Series:
    """MACD: пересечение линии и сигнальной → сигнал."""
    m = _macd(df["close"], fast, slow, signal_period)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(m["macd"] > m["signal"]) & (m["macd"].shift(1) <= m["signal"].shift(1))] = 1
    sig[(m["macd"] < m["signal"]) & (m["macd"].shift(1) >= m["signal"].shift(1))] = -1
    return sig


def macd_histogram_signal(df: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.Series:
    """MACD: гистограмма пересекает ноль → сигнал."""
    m = _macd(df["close"], fast, slow, signal_period)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(m["histogram"] > 0) & (m["histogram"].shift(1) <= 0)] = 1
    sig[(m["histogram"] < 0) & (m["histogram"].shift(1) >= 0)] = -1
    return sig


def bb_fade_signal(df: pd.DataFrame, period: int, n_std: float) -> pd.Series:
    """Bollinger fade: пробой полосы → возврат внутрь → fade."""
    b = _bb(df["close"], period, n_std)
    c = df["close"]
    above = c > b["upper"]
    below = c < b["lower"]
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[below.shift(1) & (c > b["lower"])] = 1   # был ниже, вернулся → long (fade down)
    sig[above.shift(1) & (c < b["upper"])] = -1   # был выше, вернулся → short (fade up)
    return sig


def bb_squeeze_signal(df: pd.DataFrame, bb_len: int, kc_len: int, bb_mult: float, kc_mult: float) -> pd.Series:
    """BB Squeeze: сжатие → пробой волатильности."""
    c = df["close"]
    ma = c.rolling(bb_len).mean()
    sd = c.rolling(bb_len).std(ddof=0)
    atr = _atr(df, kc_len)
    kc_hi = ma + kc_mult * atr
    kc_lo = ma - kc_mult * atr
    bb_hi = ma + bb_mult * sd
    bb_lo = ma - bb_mult * sd
    squeeze = (bb_hi < kc_hi) & (bb_lo > kc_lo)
    hi_p = df["high"].rolling(bb_len).max().shift(1)
    lo_p = df["low"].rolling(bb_len).min().shift(1)
    up = (~squeeze) & squeeze.shift(1, fill_value=False) & (c > hi_p)
    dn = (~squeeze) & squeeze.shift(1, fill_value=False) & (c < lo_p)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[up] = 1
    sig[dn] = -1
    return sig


def cci_zero_signal(df: pd.DataFrame, period: int) -> pd.Series:
    """CCI пересекает ноль → сигнал."""
    cci = _cci(df, period)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(cci > 0) & (cci.shift(1) <= 0)] = 1
    sig[(cci < 0) & (cci.shift(1) >= 0)] = -1
    return sig


def stoch_cross_signal(df: pd.DataFrame, k_period: int, d_period: int, oversold: float, overbought: float) -> pd.Series:
    """Stochastic K/D пересечение + зоны."""
    s = _stoch(df, k_period, d_period)
    sig = pd.Series(0, index=df.index, dtype=int)
    # Long: K > D, оба ниже oversold → выход из перепроданности
    sig[(s["k"] > s["d"]) & (s["k"].shift(1) <= s["d"].shift(1)) & (s["k"] < 50)] = 1
    # Short: K < D, оба выше overbought → выход из перекупленности
    sig[(s["k"] < s["d"]) & (s["k"].shift(1) >= s["d"].shift(1)) & (s["k"] > 50)] = -1
    return sig


def adx_di_signal(df: pd.DataFrame, adx_len: int, adx_threshold: float) -> pd.Series:
    """ADX + DI: ADX растёт, DI+ > ADX → long, DI- > ADX → short."""
    h, l, c = df["high"], df["low"], df["close"]
    adx = _adx(df, adx_len)
    # DI+/DI- — упрощённая версия через rolling
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1.0 / adx_len, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1.0 / adx_len, adjust=False).mean() / atr_s
    ndi = 100 * ndm.ewm(alpha=1.0 / adx_len, adjust=False).mean() / atr_s
    sig = pd.Series(0, index=df.index, dtype=int)
    adx_rising = adx > adx.shift(1)
    sig[(pdi > adx_threshold) & adx_rising] = 1
    sig[(ndi > adx_threshold) & adx_rising] = -1
    return sig


def vol_breakout_signal(df: pd.DataFrame, k: float, atr_len: int, nr7: int = 0) -> pd.Series:
    """Волатильный пробой: close > open ± k×ATR, опц. NR7."""
    atr = _atr(df, atr_len)
    long_ok = df["close"] > (df["open"] + k * atr)
    short_ok = df["close"] < (df["open"] - k * atr)
    if nr7:
        bar_range = df["high"] - df["low"]
        nr7_flag = (bar_range == bar_range.rolling(7).min()).shift(1)
        long_ok &= nr7_flag.fillna(False)
        short_ok &= nr7_flag.fillna(False)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[long_ok & ~short_ok] = 1
    sig[short_ok & ~long_ok] = -1
    return sig


def adaptive_donchian_signal(df: pd.DataFrame, const: float, adx_len: int, min_x: int = 1, max_x: int = 40) -> pd.Series:
    """Адаптивный Дончиан: длина канала = const/ADX."""
    adx = _adx(df, adx_len)
    x = (const / adx.replace(0, np.nan)).clip(min_x, max_x).round().fillna(max_x).astype(int)
    c = df["close"]
    hi = pd.Series(np.nan, index=df.index)
    lo = pd.Series(np.nan, index=df.index)
    for w in range(min_x, max_x + 1):
        mask = x == w
        if not mask.any():
            continue
        hi[mask] = df["high"].rolling(w).max().shift(1)[mask]
        lo[mask] = df["low"].rolling(w).min().shift(1)[mask]
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[c > hi] = 1
    sig[c < lo] = -1
    return sig


def random_entry_signal(df: pd.DataFrame, prob: float, seed: int = 7) -> pd.Series:
    """Случайный вход — ЭТАЛОННАЯ БАЗА."""
    rng = np.random.default_rng(seed)
    n = len(df)
    fire = rng.random(n) < prob
    direction = rng.integers(0, 2, n) * 2 - 1
    return pd.Series(np.where(fire, direction, 0), index=df.index, dtype=int)


def mean_reversion_zscore_signal(df: pd.DataFrame, ma_period: int, entry_z: float) -> pd.Series:
    """Mean-reversion по Z-score: отклонение > порога → fade."""
    ma = df["close"].rolling(ma_period).mean()
    std = df["close"].rolling(ma_period).std()
    z = (df["close"] - ma) / (std + 1e-9)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[z < -entry_z] = 1    # Oversold → long
    sig[z > entry_z] = -1    # Overbought → short
    return sig


def fractal_breakout_signal(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """Фрактал Вильямса: пробой фрактала."""
    c = df["close"]
    # Up fractal: high[i] — максимум из n баров до и после
    is_high = df["high"] == df["high"].rolling(2 * n + 1, center=True).max()
    is_low = df["low"] == df["low"].rolling(2 * n + 1, center=True).min()
    sig = pd.Series(0, index=df.index, dtype=int)
    # Пробой верхнего фрактала → long (с задержкой 2 бара на формирование)
    up_fractal = is_high.shift(2).fillna(False)
    dn_fractal = is_low.shift(2).fillna(False)
    hi_break = df["high"] > df["high"].shift(1)
    lo_break = df["low"] < df["low"].shift(1)
    sig[up_fractal & hi_break] = 1
    sig[dn_fractal & lo_break] = -1
    return sig


def pivot_daily_signal(df: pd.DataFrame, mode: str = "fade") -> pd.Series:
    """Дневные пивоты: fade на R1/S1 или breakout."""
    daily = df.resample("D").agg({"high": "max", "low": "min", "close": "last", "open": "first"}).shift(1)
    # Присоединить пивоты обратно к intraday
    p = (daily["high"] + daily["low"] + daily["close"]) / 3
    r1 = 2 * p - daily["low"]
    s1 = 2 * p - daily["high"]
    sig = pd.Series(0, index=df.index, dtype=int)
    return sig  # Заглушка — требует intraday данных


def ema_separation_signal(df: pd.DataFrame, fast: int, slow: int, m: float) -> pd.Series:
    """MA separation: обе MA в одну сторону + расстояние ≥ m×ATR."""
    ema_f = _ema(df["close"], fast)
    ema_s = _ema(df["close"], slow)
    atr_val = _atr(df, slow)
    separation = (ema_f - ema_s).abs()
    slope_f = ema_f.diff(3)
    slope_s = ema_s.diff(3)
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[(slope_f > 0) & (slope_s > 0) & (separation >= m * atr_val)] = 1
    sig[(slope_f < 0) & (slope_s < 0) & (separation >= m * atr_val)] = -1
    return sig


# =============================================================================
# ПОПОЛНЯЕМЫЙ РЕЕСТР ИНДИКАТОРОВ
# =============================================================================

_INDICATORS = [
    IndicatorDef("donchian", "Дончиан-канал (пробой N-дневного диапазона)", "trend",
                 donchian_signal, {"n": [20, 55, 96]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("ema_cross", "Пересечение двух EMA", "trend",
                 ema_cross_signal, {"fast": [12, 21, 50], "slow": [50, 100, 200]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("rsi", "RSI — выход из зон перекупленности/перепроданности", "momentum",
                 rsi_signal, {"period": [7, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("macd", "MACD — пересечение линии и сигнальной", "trend",
                 macd_signal, {"fast": [5, 12], "slow": [13, 26], "signal_period": [8, 9]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("macd_histogram", "MACD — гистограмма пересекает ноль", "trend",
                 macd_histogram_signal, {"fast": [5, 12], "slow": [13, 26], "signal_period": [8, 9]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("bb_fade", "Bollinger Bands fade — возврат внутрь после пробоя", "mean_reversion",
                 bb_fade_signal, {"period": [20, 35], "n_std": [2.0, 2.5]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="RUFP", description="Пробой BB → возврат внутрь → вход против пробоя"),
    IndicatorDef("bb_squeeze", "Bollinger Squeeze — сжатие → пробой волатильности", "volatility",
                 bb_squeeze_signal, {"bb_len": [20, 30], "kc_len": [20, 30], "bb_mult": [2.0], "kc_mult": [1.5]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="RUFP"),
    IndicatorDef("cci_zero", "CCI — пересечение нулевой линии", "momentum",
                 cci_zero_signal, {"period": [14, 20, 50]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP", description="Пересечение CCI нуля → сигнал"),
    IndicatorDef("stoch_cross", "Stochastic — пересечение K/D в зонах", "momentum",
                 stoch_cross_signal, {"k_period": [5, 14], "d_period": [3, 5], "oversold": [20, 30], "overbought": [70, 80]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="manual"),
    IndicatorDef("adx_di", "ADX + DI — направленный вход по тренду", "trend",
                 adx_di_signal, {"adx_len": [14], "adx_threshold": [15, 20, 25]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="RUFP", description="ADX растёт + DI пересекает порог → вход в сторону DI"),
    IndicatorDef("vol_breakout", "Волатильный пробой (k×ATR + NR7)", "volatility",
                 vol_breakout_signal, {"k": [0.5, 1.0, 1.5], "atr_len": [24, 96], "nr7": [0, 1]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="RUFP"),
    IndicatorDef("adaptive_donchian", "Адаптивный Дончиан (const/ADX)", "trend",
                 adaptive_donchian_signal, {"const": [100, 150, 200], "adx_len": [14]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP"),
    IndicatorDef("random_entry", "Случайный вход — ЭТАЛОННАЯ БАЗА", "baseline",
                 random_entry_signal, {"prob": [0.001, 0.002], "seed": [7, 13, 42]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX, AssetClass.CRYPTO],
                 source="RUFP", description="Случайный вход с заданной вероятностью — проверка «бета или тайминг»"),
    IndicatorDef("mean_reversion_zscore", "Mean-Reversion по Z-score", "mean_reversion",
                 mean_reversion_zscore_signal, {"ma_period": [20, 50, 200], "entry_z": [2.0, 2.5, 3.0]},
                 [AssetClass.FX, AssetClass.METAL],
                 source="manual", description="Отклонение от MA > N сигм → fade"),
    IndicatorDef("fractal_breakout", "Фрактал Вильямса — пробой фрактала", "trend",
                 fractal_breakout_signal, {"n": [2, 3, 5]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="manual"),
    IndicatorDef("ema_separation", "MA separation — расхождение скользящих", "trend",
                 ema_separation_signal, {"fast": [10, 21], "slow": [30, 70], "m": [0.5, 1.0, 2.0]},
                 [AssetClass.FX, AssetClass.METAL, AssetClass.INDEX],
                 source="RUFP", description="Обе MA в одну сторону + расстояние ≥ m×ATR"),
]

for ind in _INDICATORS:
    register_indicator(ind)


# =============================================================================
# ПАЙПЛАЙН ПЕРЕБОРА: индикатор → семейство кандидатов
# =============================================================================

@dataclass
class CandidateConfig:
    """Одна конфигурация в семействе кандидатов."""
    indicator_name: str
    params: dict[str, Any]
    filter_name: str = "none"
    filter_params: dict[str, Any] = field(default_factory=dict)
    exit_name: str = "fixed_sl_tp"
    exit_params: dict[str, Any] = field(default_factory=dict)
    label: str = ""


class IndicatorEnumerator:
    """
    Разворачивает индикатор в полное семейство кандидатов.

    Алгоритм:
      1. Берём индикатор из реестра
      2. Генерируем декартово произведение param_grid
      3. Для каждой комбинации параметров применяем фильтры
      4. Для каждого фильтра применяем стратегии выхода
      5. Результат: список CandidateConfig — готов к прогону через бэктестер
    """

    def __init__(self, indicator_name: str):
        if indicator_name not in INDICATOR_REGISTRY:
            raise KeyError(f"Индикатор '{indicator_name}' не найден в реестре. "
                           f"Доступные: {list(INDICATOR_REGISTRY.keys())}")
        self.indicator = INDICATOR_REGISTRY[indicator_name]

    def enumerate(self,
                  filters: list[str] | None = None,
                  exits: list[str] | None = None) -> list[CandidateConfig]:
        """
        Развернуть индикатор в семейство кандидатов.

        Параметры:
          filters: список имён фильтров (если None — только "none")
          exits: список имён выходов (если None — только "fixed_sl_tp")
        """
        if filters is None:
            filters = ["none"]
        if exits is None:
            exits = ["fixed_sl_tp"]

        candidates = []
        param_combos = self._cartesian_product(self.indicator.param_grid)

        for params in param_combos:
            for filt in filters:
                for ex in exits:
                    cfg = CandidateConfig(
                        indicator_name=self.indicator.name,
                        params=params,
                        filter_name=filt,
                        exit_name=ex,
                        label=f"{self.indicator.name}({self._format_params(params)})|{filt}|{ex}",
                    )
                    candidates.append(cfg)

        return candidates

    def _cartesian_product(self, grid: dict[str, list]) -> list[dict]:
        """Декартово произведение сетки параметров."""
        import itertools
        keys = list(grid.keys())
        if not keys:
            return [{}]
        values = [grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _format_params(self, params: dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items())


def enumerate_all_indicators(filters: list[str] | None = None,
                              exits: list[str] | None = None) -> dict[str, list[CandidateConfig]]:
    """Развернуть ВСЕ зарегистрированные индикаторы в семейства кандидатов."""
    result = {}
    for name in INDICATOR_REGISTRY:
        enumerator = IndicatorEnumerator(name)
        result[name] = enumerator.enumerate(filters, exits)
    return result


def count_total_candidates() -> int:
    """Сколько всего кандидатов генерируется из текущего реестра."""
    all_cands = enumerate_all_indicators()
    return sum(len(v) for v in all_cands.values())


# =============================================================================
# Сводка
# =============================================================================

if __name__ == "__main__":
    print(f"Индикаторов в реестре: {len(INDICATOR_REGISTRY)}")
    for name, ind in INDICATOR_REGISTRY.items():
        combos = 1
        for v in ind.param_grid.values():
            combos *= len(v)
        print(f"  {name:30s} | {ind.category:15s} | сетка: {combos} комбинаций | {ind.source}")
    print(f"\nВсего кандидатов (с фильтрами/выходами): {count_total_candidates()}")
    print(f"\nЗафиксировано методов:")
    for category, methods in METHOD_MANIFEST.items():
        print(f"  {category}: {len(methods)} методов")