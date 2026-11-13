"""
Риск-профили: PIPFARM, FREE и CLASSIC
=======================================
ТРИ ПРОФИЛЯ (из ТЗ):
  1. PIPFARM — дневная просадка <= 2%, дневная цель 0.3-0.8%.
     При достижении цели: закрыть ВСЕ позиции, остановить торговлю до след. дня.
  2. FREE — общая просадка <= 15%. Без дневных стопов.
  3. CLASSIC — дневной убыток <= 2%, общая просадка <= 15%.
     Прибыль по правилам бота (без дневной цели — бот сам решает когда выходить).

Адаптировано из GENERATOR RUFP (risk_engine/profiles.py).
Расширено: стоп-лосс always-on, тайм-стоп, частичные выходы.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


# =============================================================================
# Конфигурации профилей
# =============================================================================

class ProfileType(str, Enum):
    PIPFARM = "pipfarm"
    FREE = "free"
    CLASSIC = "classic"


@dataclass
class RiskProfileConfig:
    """Настройки одного риск-профиля."""
    name: str
    profile_type: ProfileType

    # PIPFARM-специфичные
    daily_stop_pct: Optional[float] = None       # Дневной стоп (% от equity на начало дня)
    daily_target_pct: Optional[float] = None      # Дневная цель (% от equity на начало дня)
    halt_on_target: bool = True                   # Остановка торговли при достижении цели

    # FREE-специфичные
    max_total_dd_pct: Optional[float] = None      # Максимальная общая просадка (% от пика)

    # Общие
    per_trade_risk_pct: float = 1.5               # Риск на сделку (% equity)
    allow_overnight: bool = True
    swap_enabled: bool = True
    max_positions: int = 6
    require_sl: bool = True                        # Каждая сделка ОБЯЗАНА иметь SL
    require_tp_sl_ratio: float = 1.5               # Минимальное отношение TP:SL

    # Действия при нарушении
    on_daily_stop: str = "FLATTEN_AND_HALT"        # Закрыть всё + остановиться до конца дня
    on_daily_target: str = "FLATTEN_AND_HALT"       # Закрыть всё + остановиться
    on_total_dd: str = "FLATTEN_AND_ALERT"          # Закрыть всё + оповещение


# =============================================================================
# Предзаполненные профили (из ТЗ)
# =============================================================================

PIPFARM_PROFILES: dict[str, RiskProfileConfig] = {
    f"PIPFARM_{pct:.1f}pct": RiskProfileConfig(
        name=f"PIPFARM (target={pct:.1f}%)",
        profile_type=ProfileType.PIPFARM,
        daily_stop_pct=2.0,
        daily_target_pct=pct,
        halt_on_target=True,
        max_total_dd_pct=8.0,       # PipFarm общая DD — 8%
        per_trade_risk_pct=1.5,
        require_sl=True,
        require_tp_sl_ratio=1.5,
    )
    for pct in [0.3, 0.5, 0.8, 1.0]
}

FREE_PROFILE = RiskProfileConfig(
    name="FREE",
    profile_type=ProfileType.FREE,
    max_total_dd_pct=15.0,
    per_trade_risk_pct=2.0,          # Чуть больше риска без дневного стопа
    require_sl=True,
    require_tp_sl_ratio=1.5,
)

CLASSIC_PROFILE = RiskProfileConfig(
    name="CLASSIC",
    profile_type=ProfileType.CLASSIC,
    daily_stop_pct=2.0,              # Дневной убыток <= 2% (как PIPFARM)
    daily_target_pct=None,            # БЕЗ дневной цели — прибыль по правилам бота
    halt_on_target=False,             # Не останавливаться при прибыли
    max_total_dd_pct=15.0,           # Общая просадка <= 15% (как FREE)
    per_trade_risk_pct=1.5,
    allow_overnight=True,
    swap_enabled=True,
    max_positions=6,
    require_sl=True,
    require_tp_sl_ratio=1.5,
    on_daily_stop="FLATTEN_AND_HALT",    # Стоп-лосс дня: закрыть всё + halt
    on_total_dd="FLATTEN_AND_ALERT",     # Общая DD: закрыть всё + alert
)

ALL_PROFILES = {**PIPFARM_PROFILES, "FREE": FREE_PROFILE, "CLASSIC": CLASSIC_PROFILE}


# =============================================================================
# Риск-движок
# =============================================================================

class RiskProfileEngine:
    """
    Отслеживает состояние риск-профиля в реальном времени.

    Для PIPFARM:
      - В начале дня фиксирует day_start_equity
      - На каждом баре проверяет: daily_return ≥ target → FLATTEN_AND_HALT
      - daily_return ≤ -daily_stop → FLATTEN_AND_HALT
      - В полночь сбрасывает halted = False

    Для FREE:
      - На каждом баре проверяет: total_dd ≥ max_total_dd → FLATTEN_AND_ALERT
    """

    def __init__(self, config: RiskProfileConfig):
        self.cfg = config
        self.current_day: Optional[date] = None
        self.day_start_equity: float = 0.0
        self.halted: bool = False
        self.peak_equity: float = 0.0
        self.trigger_log: list[dict] = []

    def on_day_start(self, ts: datetime, equity: float) -> None:
        """Вызывается в начале каждого торгового дня."""
        self.current_day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
        self.day_start_equity = float(equity)
        self.halted = False
        if equity > self.peak_equity:
            self.peak_equity = float(equity)

    def allow_entry(self) -> bool:
        """Можно ли открывать новые позиции?"""
        return not self.halted

    def daily_return(self, equity: float) -> float:
        """Дневная доходность от начала дня."""
        if self.current_day is None or self.day_start_equity <= 0:
            return 0.0
        return float(equity) / self.day_start_equity - 1.0

    def total_drawdown(self, equity: float) -> float:
        """Общая просадка от пика."""
        if self.peak_equity <= 0:
            return 0.0
        return 1.0 - float(equity) / self.peak_equity

    def check(self, ts: datetime, equity: float) -> list[dict]:
        """
        Проверить состояние риск-профиля на данном баре.
        Возвращает список действий.
        """
        actions = []
        if self.halted:
            return actions

        # Обновить пик
        if equity > self.peak_equity:
            self.peak_equity = float(equity)

        daily_ret = self.daily_return(equity)

        eps = 1e-9  # Допуск плавающей точки

        if self.cfg.profile_type == ProfileType.PIPFARM:
            # Проверка дневного стопа
            if self.cfg.daily_stop_pct is not None and daily_ret <= -self.cfg.daily_stop_pct / 100.0 + eps:
                actions.append({
                    "action": self.cfg.on_daily_stop,
                    "trigger": "daily_stop",
                    "value": daily_ret * 100,
                    "threshold": -self.cfg.daily_stop_pct,
                    "ts": str(ts),
                })
                self.halted = True
                self.trigger_log.append(actions[-1])

            # Проверка дневной цели
            elif self.cfg.daily_target_pct is not None and daily_ret >= self.cfg.daily_target_pct / 100.0 - eps:
                actions.append({
                    "action": self.cfg.on_daily_target,
                    "trigger": "daily_target",
                    "value": daily_ret * 100,
                    "threshold": self.cfg.daily_target_pct,
                    "ts": str(ts),
                })
                self.halted = True
                self.trigger_log.append(actions[-1])

        elif self.cfg.profile_type == ProfileType.FREE:
            total_dd = self.total_drawdown(equity)
            if self.cfg.max_total_dd_pct is not None and total_dd >= self.cfg.max_total_dd_pct / 100.0 - eps:
                actions.append({
                    "action": self.cfg.on_total_dd,
                    "trigger": "total_drawdown",
                    "value": total_dd * 100,
                    "threshold": self.cfg.max_total_dd_pct,
                    "ts": str(ts),
                })
                self.halted = True
                self.trigger_log.append(actions[-1])

        elif self.cfg.profile_type == ProfileType.CLASSIC:
            # CLASSIC: дневной стоп 2% + общая просадка 15%, БЕЗ дневной цели
            halted_by_stop = False

            # 1. Проверка дневного стопа (как PIPFARM)
            if self.cfg.daily_stop_pct is not None and daily_ret <= -self.cfg.daily_stop_pct / 100.0 + eps:
                actions.append({
                    "action": self.cfg.on_daily_stop,
                    "trigger": "daily_stop",
                    "value": daily_ret * 100,
                    "threshold": -self.cfg.daily_stop_pct,
                    "ts": str(ts),
                })
                self.halted = True
                halted_by_stop = True
                self.trigger_log.append(actions[-1])

            # 2. Проверка общей просадки (как FREE) — даже если daily_stop уже сработал
            total_dd = self.total_drawdown(equity)
            if self.cfg.max_total_dd_pct is not None and total_dd >= self.cfg.max_total_dd_pct / 100.0 - eps:
                actions.append({
                    "action": self.cfg.on_total_dd,
                    "trigger": "total_drawdown",
                    "value": total_dd * 100,
                    "threshold": self.cfg.max_total_dd_pct,
                    "ts": str(ts),
                })
                self.halted = True
                self.trigger_log.append(actions[-1])

            # 3. Прибыль — по правилам бота (без daily_target, без halt_on_target)

        return actions

    def should_close_all(self, ts: datetime, equity: float) -> bool:
        """Нужно ли закрыть все позиции прямо сейчас?"""
        actions = self.check(ts, equity)
        return any("FLATTEN" in a["action"] for a in actions)


# =============================================================================
# Статические метрики просадок (пост-фактум)
# =============================================================================

def compute_total_drawdown(equity: pd.Series) -> float:
    """Максимальная общая просадка (% от пика)."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min()) * 100


def compute_max_daily_drawdown(equity: pd.Series) -> float:
    """Максимальная внутридневная просадка (% от начала дня)."""
    worst = 0.0
    for _, group in equity.groupby(equity.index.date):
        start = float(group.iloc[0])
        if start > 0:
            dd = float(group.min()) / start - 1.0
            worst = min(worst, dd)
    return -worst * 100


def profile_gate_pass(equity: pd.Series, config: RiskProfileConfig) -> dict:
    """
    Проверить, проходит ли эквити-кривая через ворота риск-профиля.
    Возвращает словарь: {passed: bool, reason: str, metrics: dict}.
    """
    total_dd = compute_total_drawdown(equity)
    daily_dd = compute_max_daily_drawdown(equity)

    metrics = {
        "total_dd_pct": total_dd,
        "max_daily_dd_pct": daily_dd,
    }

    if config.profile_type == ProfileType.PIPFARM:
        if config.max_total_dd_pct and total_dd > config.max_total_dd_pct:
            return {"passed": False, "reason": f"Общая DD {total_dd:.1f}% > {config.max_total_dd_pct}%", "metrics": metrics}
        if config.daily_stop_pct and daily_dd > config.daily_stop_pct:
            return {"passed": False, "reason": f"Дневная DD {daily_dd:.1f}% > {config.daily_stop_pct}%", "metrics": metrics}

    elif config.profile_type == ProfileType.FREE:
        if config.max_total_dd_pct and total_dd > config.max_total_dd_pct:
            return {"passed": False, "reason": f"Общая DD {total_dd:.1f}% > {config.max_total_dd_pct}%", "metrics": metrics}

    elif config.profile_type == ProfileType.CLASSIC:
        # CLASSIC: дневной стоп (как PIPFARM) + общая просадка (как FREE)
        if config.daily_stop_pct and daily_dd > config.daily_stop_pct:
            return {"passed": False, "reason": f"Дневная DD {daily_dd:.1f}% > {config.daily_stop_pct}%", "metrics": metrics}
        if config.max_total_dd_pct and total_dd > config.max_total_dd_pct:
            return {"passed": False, "reason": f"Общая DD {total_dd:.1f}% > {config.max_total_dd_pct}%", "metrics": metrics}

    return {"passed": True, "reason": "OK", "metrics": metrics}


# =============================================================================
# Быстрый тест
# =============================================================================

if __name__ == "__main__":
    # Тест PIPFARM profile
    engine = RiskProfileEngine(PIPFARM_PROFILES["PIPFARM_0.5pct"])

    # Симуляция: день, эквити растёт
    engine.on_day_start(datetime(2026, 11, 10, 7, 0), 50000.0)

    # +0.3% - ещё не цель
    actions = engine.check(datetime(2026, 11, 10, 12, 0), 50150.0)
    print(f"PIPFARM +0.3%: halted={engine.halted}, actions={len(actions)}")

    # +0.5% - цель достигнута
    actions = engine.check(datetime(2026, 11, 10, 14, 0), 50250.0)
    print(f"PIPFARM +0.5%: halted={engine.halted}, actions={actions}")

    # Тест CLASSIC profile
    print("\n--- CLASSIC ---")
    ce = RiskProfileEngine(CLASSIC_PROFILE)
    ce.on_day_start(datetime(2026, 11, 10, 7, 0), 50000.0)

    # +1.5% прибыль - бот решает когда выходить, CLASSIC не останавливает
    a = ce.check(datetime(2026, 11, 10, 14, 0), 50750.0)
    print(f"CLASSIC +1.5%%: halted={ce.halted} (не должен — прибыль по боту)")

    # -2.1% дневной убыток - CLASSIC останавливает
    ce2 = RiskProfileEngine(CLASSIC_PROFILE)
    ce2.on_day_start(datetime(2026, 11, 10, 7, 0), 50000.0)
    a = ce2.check(datetime(2026, 11, 10, 14, 0), 48950.0)
    print(f"CLASSIC -2.1%%: halted={ce2.halted}, actions={a}")

    # Общая просадка >15%
    ce3 = RiskProfileEngine(CLASSIC_PROFILE)
    ce3.on_day_start(datetime(2026, 11, 10, 7, 0), 50000.0)
    ce3.peak_equity = 50000.0
    a = ce3.check(datetime(2026, 11, 10, 14, 0), 42000.0)
    print(f"CLASSIC DD 16%%: halted={ce3.halted}, actions={a}")

    print(f"\nДоступные профили: {list(ALL_PROFILES.keys())}")
    for name, cfg in ALL_PROFILES.items():
        dd = cfg.max_total_dd_pct or "N/A"
        ds = cfg.daily_stop_pct or "N/A"
        dt = cfg.daily_target_pct or "N/A"
        print(f"  {name}: type={cfg.profile_type.value}, total_dd={dd}%, daily_stop={ds}%, target={dt}%")