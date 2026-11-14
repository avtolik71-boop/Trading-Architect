"""
Ядро издержек: спред + комиссия + свопы
========================================
ПРАВИЛО: Издержки вычитаются ВСЕГДА. Результат без издержек — не результат.

Адаптировано из GENERATOR RUFP (costs/model.py, risk_engine/profiles.py).
Данные: Pipfarm cTrader / Tickmill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
import os

import numpy as np
import pandas as pd


# =============================================================================
# Модели издержек
# =============================================================================

@dataclass
class CostConfig:
    """Конфигурация издержек для одного инструмента."""
    symbol: str
    # Спред
    avg_spread_pips: float = 0.0          # Средний спред в пипсах (перезаписывает данные фида)
    spread_type: str = "additive"          # "additive" (добавить к цене) | "replacement" (заменить)
    # Комиссия
    commission_per_lot: float = 0.0        # $ за лот
    commission_round_turn: bool = True     # True = списание при ОТКРЫТИИ (PipFarm); False = при открытии И закрытии
    commission_min: float = 0.0            # Минимальная комиссия
    # Своп
    swap_long_points: float = 0.0          # Своп лонг (пункты/день)
    swap_short_points: float = 0.0         # Своп шорт (пункты/день)
    swap_triple_day: int = 3               # День тройного свопа (3 = среда, ISO weekday)
    # Проскальзывание
    slippage_pips: float = 0.0             # Фиксированное проскальзывание
    slippage_pct_of_spread: float = 0.0    # Проскальзывание как % от спреда


@dataclass
class CostConfigSet:
    """Набор конфигураций издержек для всех инструментов."""
    configs: dict[str, CostConfig] = field(default_factory=dict)
    source: str = ""    # "PIPFARM_CTRADER" или "TICKMILL"

    def get(self, symbol: str) -> CostConfig:
        if symbol not in self.configs:
            raise KeyError(f"Издержки для '{symbol}' не настроены. "
                           f"Добавьте в cost_configs.yaml. Имеющиеся: {list(self.configs.keys())}")
        return self.configs[symbol]


class CostEngine:
    """
    Движок издержек. Применяет спред, комиссию, свопы и проскальзывание
    к каждой сделке. Вызывается бэктестером при открытии и закрытии позиции.
    """

    def __init__(self, cost_set: CostConfigSet, instrument_config: dict):
        """
        instrument_config: словарь с полями pip_size, contract_size, lot_size для каждого символа.
        """
        self.cost_set = cost_set
        self.instruments = instrument_config
        self._swap_cache: dict[tuple, float] = {}

    def apply_entry_costs(self, symbol: str, side: str, volume_lots: float,
                          entry_price: float, bar_ts: datetime) -> dict:
        """
        Вычислить издержки ПРИ ОТКРЫТИИ позиции.
        Возвращает {"net_entry_price": float, "spread_cost": float, "commission": float, "slippage": float}.
        """
        cfg = self.cost_set.get(symbol)
        pip_size = self.instruments.get(symbol, {}).get("pip_size", 0.0001)

        # 1. Спред — всегда
        spread_cost = volume_lots * cfg.avg_spread_pips * (pip_size / self.instruments.get(symbol, {}).get("pip_size", 0.0001))
        spread_price = cfg.avg_spread_pips * pip_size
        net_entry = entry_price + (spread_price if side.upper() == "BUY" else -spread_price)

        # 2. Комиссия — всегда
        commission = volume_lots * cfg.commission_per_lot
        if cfg.commission_round_turn:
            # PipFarm: вся комиссия при открытии
            pass
        else:
            # Стандарт: половина при открытии
            commission = commission / 2

        # 3. Проскальзывание — всегда
        slippage_pips = cfg.slippage_pips + cfg.slippage_pct_of_spread * cfg.avg_spread_pips
        slippage = slippage_pips * pip_size
        net_entry = net_entry + (slippage if side.upper() == "BUY" else -slippage)

        return {
            "net_entry_price": net_entry,
            "spread_cost_pips": cfg.avg_spread_pips,
            "commission_usd": commission,
            "slippage_pips": slippage_pips,
            "gross_price": entry_price,
        }

    def apply_exit_costs(self, symbol: str, side: str, volume_lots: float,
                         exit_price: float, bar_ts: datetime) -> dict:
        """Вычислить издержки ПРИ ЗАКРЫТИИ."""
        cfg = self.cost_set.get(symbol)
        pip_size = self.instruments.get(symbol, {}).get("pip_size", 0.0001)

        # Спред уже учтён при входе — при выходе только комиссия и проскальзывание
        commission = 0.0
        if not cfg.commission_round_turn:
            # Вторая половина комиссии при закрытии
            commission = volume_lots * cfg.commission_per_lot / 2

        slippage_pips = cfg.slippage_pips + cfg.slippage_pct_of_spread * cfg.avg_spread_pips
        slippage = slippage_pips * pip_size
        net_exit = exit_price - (slippage if side.upper() == "BUY" else -slippage)

        return {
            "net_exit_price": net_exit,
            "commission_usd": commission,
            "slippage_pips": slippage_pips,
        }

    def compute_swap(self, symbol: str, side: str, volume_lots: float,
                     nights_held: int, bar_ts: datetime) -> float:
        """
        Вычислить своп за N ночей удержания.
        Учитывает тройной своп (среда для Forex).
        """
        cfg = self.cost_set.get(symbol)
        swap_points = cfg.swap_long_points if side.upper() == "BUY" else cfg.swap_short_points

        if swap_points == 0:
            return 0.0

        # Разбиваем ночи по дням, считаем triple
        total_swap = 0.0
        ts = pd.Timestamp(bar_ts)
        for i in range(nights_held):
            day_iso = (ts + pd.Timedelta(days=i)).dayofweek + 1  # 1=Mon ... 7=Sun
            multiplier = 3 if day_iso == cfg.swap_triple_day else 1
            total_swap += swap_points * multiplier

        return total_swap * volume_lots / 100.0  # пункты → USD (приблизительно)

    def compute_total_trade_cost(self, symbol: str, side: str, volume_lots: float,
                                  entry_price: float, exit_price: float,
                                  entry_ts: datetime, exit_ts: datetime) -> dict:
        """
        ПОЛНЫЙ РАСЧЁТ ИЗДЕРЖЕК СДЕЛКИ.
        Возвращает детальную разбивку и итог.
        """
        entry_costs = self.apply_entry_costs(symbol, side, volume_lots, entry_price, entry_ts)
        exit_costs = self.apply_exit_costs(symbol, side, volume_lots, exit_price, exit_ts)

        nights = (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).days
        swap = self.compute_swap(symbol, side, volume_lots, max(0, nights), exit_ts)

        gross_pnl = volume_lots * (exit_price - entry_price)
        total_costs = (entry_costs["commission_usd"] + exit_costs["commission_usd"] + swap)

        return {
            "gross_pnl": gross_pnl,
            "spread_cost": entry_costs["spread_cost_pips"],
            "commission_total": entry_costs["commission_usd"] + exit_costs["commission_usd"],
            "swap_total": swap,
            "slippage_total": entry_costs["slippage_pips"] + exit_costs["slippage_pips"],
            "total_costs_usd": total_costs,
            "net_pnl": gross_pnl - total_costs,
            "nights_held": nights,
        }


# =============================================================================
# Предзаполненные конфигурации (данные Pipfarm / Tickmill)
# =============================================================================

PIPFARM_COSTS = CostConfigSet(
    source="PIPFARM_CTRADER",
    configs={
        "XAUUSD": CostConfig(
            symbol="XAUUSD", avg_spread_pips=20.0, spread_type="additive",
            commission_per_lot=3.0, commission_round_turn=True,
            swap_long_points=-35.0, swap_short_points=5.0,
            slippage_pips=5.0,
        ),
        "US100": CostConfig(
            symbol="US100", avg_spread_pips=150.0, spread_type="additive",
            commission_per_lot=1.0, commission_round_turn=True,
            swap_long_points=-15.0, swap_short_points=2.0,
            slippage_pips=10.0,
        ),
        "GBPUSD": CostConfig(
            symbol="GBPUSD", avg_spread_pips=1.0, spread_type="additive",
            commission_per_lot=3.0, commission_round_turn=True,
            swap_long_points=-3.5, swap_short_points=0.5,
            slippage_pips=0.5,
        ),
        "USDJPY": CostConfig(
            symbol="USDJPY", avg_spread_pips=1.0, spread_type="additive",
            commission_per_lot=3.0, commission_round_turn=True,
            swap_long_points=5.0, swap_short_points=-15.0,
            slippage_pips=0.5,
        ),
        "EURUSD": CostConfig(
            symbol="EURUSD", avg_spread_pips=0.5, spread_type="additive",
            commission_per_lot=3.0, commission_round_turn=True,
            swap_long_points=-2.5, swap_short_points=-1.0,
            slippage_pips=0.3,
        ),
        "BTCUSD": CostConfig(
            symbol="BTCUSD", avg_spread_pips=25.0, spread_type="additive",
            commission_per_lot=3.0, commission_round_turn=True,
            swap_long_points=-20.0, swap_short_points=2.0,
            slippage_pips=5.0,
        ),
    },
)

TICKMILL_COSTS = CostConfigSet(
    source="TICKMILL",
    configs={
        "XAUUSD": CostConfig(
            symbol="XAUUSD", avg_spread_pips=15.0, spread_type="additive",
            commission_per_lot=2.0, commission_round_turn=False,
            swap_long_points=-30.0, swap_short_points=3.0,
            slippage_pips=3.0,
        ),
        "GBPUSD": CostConfig(
            symbol="GBPUSD", avg_spread_pips=0.8, spread_type="additive",
            commission_per_lot=2.0, commission_round_turn=False,
            swap_long_points=-3.0, swap_short_points=0.3,
            slippage_pips=0.3,
        ),
        "USDJPY": CostConfig(
            symbol="USDJPY", avg_spread_pips=0.7, spread_type="additive",
            commission_per_lot=2.0, commission_round_turn=False,
            swap_long_points=4.0, swap_short_points=-12.0,
            slippage_pips=0.3,
        ),
        "BTCUSD": CostConfig(
            symbol="BTCUSD", avg_spread_pips=20.0, spread_type="additive",
            commission_per_lot=2.0, commission_round_turn=False,
            swap_long_points=-18.0, swap_short_points=1.5,
            slippage_pips=3.0,
        ),
    },
)

ACTIVE_COST_SET = PIPFARM_COSTS  # Изменить на TICKMILL_COSTS при смене брокера


# =============================================================================
# Проверка
# =============================================================================

if __name__ == "__main__":
    instruments = {
        "XAUUSD": {"pip_size": 0.1, "contract_size": 100, "lot_size": 100},
        "GBPUSD": {"pip_size": 0.0001, "contract_size": 100000, "lot_size": 100000},
        "USDJPY": {"pip_size": 0.01, "contract_size": 100000, "lot_size": 100000},
    }
    engine = CostEngine(PIPFARM_COSTS, instruments)

    # Пример: лонг XAUUSD 0.1 лота, вход 2650, выход 2660, держали 3 дня
    result = engine.compute_total_trade_cost(
        "XAUUSD", "BUY", 0.1, 2650.0, 2660.0,
        datetime(2026, 11, 10, 14, 0), datetime(2026, 11, 13, 10, 0),
    )
    print("=== Пример расчёта издержек XAUUSD ===")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nАктивный брокер: {ACTIVE_COST_SET.source}")
    for sym, cfg in ACTIVE_COST_SET.configs.items():
        print(f"  {sym}: spread={cfg.avg_spread_pips} pip, comm=${cfg.commission_per_lot}/лот, round_turn={cfg.commission_round_turn}")