"""
РЕЕСТР ИДЕЙ И СТРАТЕГИЙ
========================
Пополняемая база с формальной статусной моделью.

Жизненный цикл идеи:
  IDEA          - сырая гипотеза, ещё не формализована
  REGISTERED    - формализована: индикатор, параметры, фильтры, выходы
  BACKTESTED    - прогнана через in-sample бэктест
  WALK_FORWARD  - прошла walk-forward валидацию
  GO_CANDIDATE  - одобрена для боевого переноса
  LIVE          - торгуется на реальном счёте
  PAUSED        - приостановлена (рыночные условия изменились)
  RETIRED       - выведена из эксплуатации

Добавление новой идеи:
  1. IdeasRegistry.register(idea_dict)
  2. После формализации -> StrategyRegistry.register(strategy_dict)
  3. После тестов -> update_result() с метриками
  4. После GO -> update_status("GO_CANDIDATE")
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import json
import os

# =============================================================================
# Статусы
# =============================================================================

class IdeaStatus(str, Enum):
    IDEA = "IDEA"
    REGISTERED = "REGISTERED"
    BACKTESTED = "BACKTESTED"
    WALK_FORWARD = "WALK_FORWARD"
    GO_CANDIDATE = "GO_CANDIDATE"
    LIVE = "LIVE"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


# Разрешённые переходы статусов
STATUS_TRANSITIONS: dict[IdeaStatus, list[IdeaStatus]] = {
    IdeaStatus.IDEA:        [IdeaStatus.REGISTERED, IdeaStatus.REJECTED],
    IdeaStatus.REGISTERED:  [IdeaStatus.BACKTESTED, IdeaStatus.REJECTED],
    IdeaStatus.BACKTESTED:  [IdeaStatus.WALK_FORWARD, IdeaStatus.REGISTERED, IdeaStatus.REJECTED],
    IdeaStatus.WALK_FORWARD:[IdeaStatus.GO_CANDIDATE, IdeaStatus.BACKTESTED, IdeaStatus.REJECTED],
    IdeaStatus.GO_CANDIDATE:[IdeaStatus.LIVE, IdeaStatus.WALK_FORWARD, IdeaStatus.PAUSED, IdeaStatus.REJECTED],
    IdeaStatus.LIVE:        [IdeaStatus.PAUSED, IdeaStatus.RETIRED],
    IdeaStatus.PAUSED:      [IdeaStatus.LIVE, IdeaStatus.RETIRED],
    IdeaStatus.RETIRED:     [],
    IdeaStatus.REJECTED:    [IdeaStatus.IDEA],  # Можно переосмыслить
}


class SourceType(str, Enum):
    RUFP = "RUFP"                    # Из GENERATOR RUFP
    MQL5 = "MQL5"                    # MQL5-статьи
    TRADINGVIEW = "TRADINGVIEW"      # TradingView скрипты
    MANUAL = "MANUAL"                # Ручная гипотеза
    RESEARCH = "RESEARCH"            # Из исследовательской литературы
    DEEPSEEK = "DEEPSEEK"            # Сгенерировано DeepSeek
    IMPORTED = "IMPORTED"            # Импортировано из внешнего источника


class AssetClass(str, Enum):
    FX = "FX"
    METAL = "METAL"
    INDEX = "INDEX"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"
    ANY = "ANY"


# =============================================================================
# Типы данных реестра
# =============================================================================

@dataclass
class IdeaEntry:
    """Одна идея в реестре."""
    idea_id: str                           # Уникальный ID: "I-001", "I-002"
    title: str                             # Короткое название
    description: str                       # Развёрнутое описание гипотезы
    status: IdeaStatus = IdeaStatus.IDEA
    source: SourceType = SourceType.MANUAL
    source_ref: str = ""                   # Ссылка на источник (файл, URL, статья)
    asset_classes: list[AssetClass] = field(default_factory=lambda: [AssetClass.ANY])
    timeframes: list[str] = field(default_factory=list)  # ["H1", "H4", "D1"]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)  # ["mean-reversion", "trend", "volatility"]
    status_history: list[dict] = field(default_factory=list)  # [{from:, to:, at:, reason:}]
    linked_strategy_id: Optional[str] = None   # Привязка к StrategyEntry после формализации

    def transition(self, new_status: IdeaStatus, reason: str = "") -> bool:
        """Перевести идею в новый статус с проверкой допустимости."""
        if new_status not in STATUS_TRANSITIONS.get(self.status, []):
            raise ValueError(f"Недопустимый переход: {self.status.value} -> {new_status.value}. "
                             f"Разрешено: {[s.value for s in STATUS_TRANSITIONS.get(self.status, [])]}")
        old = self.status
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.status_history.append({
            "from": old.value, "to": new_status.value,
            "at": self.updated_at, "reason": reason,
        })
        return True


@dataclass
class StrategyEntry:
    """Формализованная стратегия - привязана к идее."""
    strategy_id: str                       # "S-001"
    idea_id: str                           # Привязка к IdeaEntry
    name: str                              # Человеческое название
    indicator: str                         # Имя индикатора из INDICATOR_REGISTRY
    params: dict[str, Any] = field(default_factory=dict)   # {"n": 55, "k": 1.0}
    filter_name: str = "none"
    filter_params: dict[str, Any] = field(default_factory=dict)
    exit_name: str = "fixed_sl_tp"
    exit_params: dict[str, Any] = field(default_factory=dict)
    risk_profile: str = "PIPFARM_0.5pct"   # Имя профиля из ALL_PROFILES
    instruments: list[str] = field(default_factory=list)
    horizon: str = "H1"

    # Результаты тестирования
    backtest_result: Optional[dict] = None  # {return_pct, sharpe, max_dd, n_trades, ...}
    walk_forward_result: Optional[dict] = None
    null_model_verdict: str = ""            # ALPHA / BETA_OR_TIMING / NO_EDGE

    # Метаданные
    status: IdeaStatus = IdeaStatus.REGISTERED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    notes: str = ""


# =============================================================================
# РЕЕСТР ИДЕЙ
# =============================================================================

class IdeasRegistry:
    """
    Пополняемый реестр торговых идей.

    Использование:
      registry = IdeasRegistry()
      registry.register(IdeaEntry(idea_id="I-001", title="...", ...))
      registry.transition("I-001", IdeaStatus.REGISTERED, "Формализована в S-001")
    """

    def __init__(self):
        self._ideas: dict[str, IdeaEntry] = {}

    def register(self, idea: IdeaEntry) -> IdeaEntry:
        if idea.idea_id in self._ideas:
            raise ValueError(f"Идея '{idea.idea_id}' уже существует.")
        self._ideas[idea.idea_id] = idea
        return idea

    def get(self, idea_id: str) -> IdeaEntry:
        if idea_id not in self._ideas:
            raise KeyError(f"Идея '{idea_id}' не найдена. Имеющиеся: {list(self._ideas.keys())}")
        return self._ideas[idea_id]

    def transition(self, idea_id: str, new_status: IdeaStatus, reason: str = "") -> IdeaEntry:
        idea = self.get(idea_id)
        idea.transition(new_status, reason)
        return idea

    def link_strategy(self, idea_id: str, strategy_id: str) -> None:
        idea = self.get(idea_id)
        idea.linked_strategy_id = strategy_id
        idea.updated_at = datetime.now(timezone.utc).isoformat()

    def list_by_status(self, status: IdeaStatus) -> list[IdeaEntry]:
        return [i for i in self._ideas.values() if i.status == status]

    def list_by_tag(self, tag: str) -> list[IdeaEntry]:
        return [i for i in self._ideas.values() if tag in i.tags]

    def summary(self) -> dict:
        counts = {s.value: 0 for s in IdeaStatus}
        for idea in self._ideas.values():
            counts[idea.status.value] += 1
        return {"total": len(self._ideas), "by_status": counts}

    def dump(self) -> list[dict]:
        return [asdict(i) for i in self._ideas.values()]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dump(), f, indent=2, ensure_ascii=False, default=str)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            idea = IdeaEntry(
                idea_id=item["idea_id"],
                title=item["title"],
                description=item["description"],
                status=IdeaStatus(item["status"]),
                source=SourceType(item.get("source", "MANUAL")),
                source_ref=item.get("source_ref", ""),
                asset_classes=[AssetClass(a) for a in item.get("asset_classes", ["ANY"])],
                timeframes=item.get("timeframes", []),
                created_at=item.get("created_at", ""),
                updated_at=item.get("updated_at", ""),
                tags=item.get("tags", []),
                status_history=item.get("status_history", []),
                linked_strategy_id=item.get("linked_strategy_id"),
            )
            self._ideas[idea.idea_id] = idea
            # Don't overwrite status_history, use it
            idea.status_history = item.get("status_history", [])

    def __len__(self):
        return len(self._ideas)

    def __contains__(self, idea_id: str) -> bool:
        return idea_id in self._ideas


# =============================================================================
# РЕЕСТР СТРАТЕГИЙ
# =============================================================================

class StrategyRegistry:
    """
    Пополняемый реестр формализованных стратегий.

    Каждая стратегия привязана к идее из IdeasRegistry.
    """

    def __init__(self):
        self._strategies: dict[str, StrategyEntry] = {}

    def register(self, strategy: StrategyEntry) -> StrategyEntry:
        if strategy.strategy_id in self._strategies:
            raise ValueError(f"Стратегия '{strategy.strategy_id}' уже существует.")
        self._strategies[strategy.strategy_id] = strategy
        return strategy

    def get(self, strategy_id: str) -> StrategyEntry:
        if strategy_id not in self._strategies:
            raise KeyError(f"Стратегия '{strategy_id}' не найдена. Имеющиеся: {list(self._strategies.keys())}")
        return self._strategies[strategy_id]

    def update_result(self, strategy_id: str, result: dict, result_type: str = "backtest") -> None:
        """Обновить результаты тестирования."""
        s = self.get(strategy_id)
        if result_type == "backtest":
            s.backtest_result = result
        elif result_type == "walk_forward":
            s.walk_forward_result = result
        s.updated_at = datetime.now(timezone.utc).isoformat()

    def update_null_verdict(self, strategy_id: str, verdict: str) -> None:
        self.get(strategy_id).null_model_verdict = verdict

    def list_by_indicator(self, indicator_name: str) -> list[StrategyEntry]:
        return [s for s in self._strategies.values() if s.indicator == indicator_name]

    def list_by_risk_profile(self, profile: str) -> list[StrategyEntry]:
        return [s for s in self._strategies.values() if s.risk_profile == profile]

    def list_go_candidates(self) -> list[StrategyEntry]:
        return [s for s in self._strategies.values()
                if s.null_model_verdict == "ALPHA" and s.walk_forward_result]

    def summary(self) -> dict:
        indicators = {}
        profiles = {}
        for s in self._strategies.values():
            indicators[s.indicator] = indicators.get(s.indicator, 0) + 1
            profiles[s.risk_profile] = profiles.get(s.risk_profile, 0) + 1
        return {
            "total": len(self._strategies),
            "by_indicator": indicators,
            "by_risk_profile": profiles,
            "go_candidates": len(self.list_go_candidates()),
        }

    def dump(self) -> list[dict]:
        return [asdict(s) for s in self._strategies.values()]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dump(), f, indent=2, ensure_ascii=False, default=str)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            s = StrategyEntry(
                strategy_id=item["strategy_id"],
                idea_id=item["idea_id"],
                name=item["name"],
                indicator=item["indicator"],
                params=item.get("params", {}),
                filter_name=item.get("filter_name", "none"),
                filter_params=item.get("filter_params", {}),
                exit_name=item.get("exit_name", "fixed_sl_tp"),
                exit_params=item.get("exit_params", {}),
                risk_profile=item.get("risk_profile", "PIPFARM_0.5pct"),
                instruments=item.get("instruments", []),
                horizon=item.get("horizon", "H1"),
                backtest_result=item.get("backtest_result"),
                walk_forward_result=item.get("walk_forward_result"),
                null_model_verdict=item.get("null_model_verdict", ""),
                status=IdeaStatus(item.get("status", "REGISTERED")),
                created_at=item.get("created_at", ""),
                updated_at=item.get("updated_at", ""),
                notes=item.get("notes", ""),
            )
            self._strategies[s.strategy_id] = s

    def __len__(self):
        return len(self._strategies)


# =============================================================================
# ПРЕДЗАПОЛНЕННЫЕ ИДЕИ (из STRATEGY_CATALOG.md + RUFP PROJECT)
# =============================================================================

def seed_ideas() -> IdeasRegistry:
    """Заполнить реестр идеями из существующих источников."""
    registry = IdeasRegistry()

    seeds = [
        IdeaEntry(
            idea_id="I-001", title="Пробой Дончиан-канала на золоте",
            description="XAUUSD H1: пробой N-дневного максимума/минимума с сессионным фильтром 7-20 UTC. "
                        "SL=500, TP=400. Walk-forward OOS: 4/4 варианта PF 1.26-1.44 - лучший кандидат волны-1.",
            status=IdeaStatus.GO_CANDIDATE, source=SourceType.RUFP,
            source_ref="GENERATOR RUFP / STRATEGY_CATALOG.md / WF_20261101_172930",
            asset_classes=[AssetClass.METAL], timeframes=["H1"],
            tags=["trend", "donchian", "go-candidate"],
        ),
        IdeaEntry(
            idea_id="I-002", title="Волатильный пробой USDJPY",
            description="USDJPY H1: vol_breakout k=0.5, ATR 24, сессия 7-20, SL=500, TP=400. "
                        "Единственный не-дончиановский PASS волны-1: PF=1.83 OOS, n=57.",
            status=IdeaStatus.GO_CANDIDATE, source=SourceType.RUFP,
            source_ref="GENERATOR RUFP / WF_20261101_172930 / validation.csv",
            asset_classes=[AssetClass.FX], timeframes=["H1"],
            tags=["volatility", "breakout", "go-candidate"],
        ),
        IdeaEntry(
            idea_id="I-003", title="Bollinger Fade - возврат внутрь полосы",
            description="Пробой BB(35,2) -> следующий бар закрылся внутри полос -> вход против пробоя. "
                        "Mean-reversion на H4/D1. Из каталога стратегий (приоритет-1).",
            status=IdeaStatus.REGISTERED, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / BB fade",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H4", "D1"],
            tags=["mean-reversion", "bollinger"],
        ),
        IdeaEntry(
            idea_id="I-004", title="Holy Grail - ADX + откат к EMA",
            description="ADX(14) растёт -> откат к EMA(20) -> закрытие обратно в сторону тренда. "
                        "Трейлинг-стоп. H4/D1. Линда Рашке / Тереза Ло.",
            status=IdeaStatus.REGISTERED, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / Holy Grail",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H4", "D1"],
            tags=["trend", "pullback", "adx"],
        ),
        IdeaEntry(
            idea_id="I-005", title="Тройной экран Элдера",
            description="Старший ТФ D1: MACD-гистограмма. Средний ТФ H4/H1: RSI(14) <30 и восстановление -> long. "
                        "Стоп: min/max предыдущего дня.",
            status=IdeaStatus.REGISTERED, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / Elder Triple Screen",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["D1+H4", "D1+H1"],
            tags=["multi-tf", "trend", "rsi"],
        ),
        IdeaEntry(
            idea_id="I-006", title="MACD Pullback мульти-ТФ",
            description="D1: MACD(5,13,8) > 0 = тренд. H4: кросс вниз = откат. Вход при PSAR-flip или повторном кроссе вверх. "
                        "SL: last swing low -15 пипс. TP = 2xR.",
            status=IdeaStatus.REGISTERED, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / MACD Pullback",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["D1+H4"],
            tags=["multi-tf", "trend", "macd"],
        ),
        IdeaEntry(
            idea_id="I-007", title="MA Separation - расхождение скользящих",
            description="Обе EMA в одну сторону (slope) + расстояние >= m×ATR и растёт -> вход. "
                        "Вместо классического кросса. H4/D1.",
            status=IdeaStatus.REGISTERED, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / MA Separation",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H4", "D1"],
            tags=["trend", "ema", "separation"],
        ),
        IdeaEntry(
            idea_id="I-008", title="Mean-Reversion по Z-score",
            description="Отклонение цены от MA(50) на >= 2.5σ -> fade. "
                        "Фильтр: ADX < 25 (только ranging рынок). H1/H4.",
            status=IdeaStatus.REGISTERED, source=SourceType.MANUAL,
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H1", "H4"],
            tags=["mean-reversion", "zscore", "ranging"],
        ),
        IdeaEntry(
            idea_id="I-009", title="Адаптивный Дончиан - const/ADX",
            description="Длина канала = const/ADX(14), clamp(1,40). "
                        "На волатильном рынке канал уже, на спокойном - шире. In-sample XAUUSD +28.2%.",
            status=IdeaStatus.BACKTESTED, source=SourceType.RUFP,
            source_ref="GENERATOR RUFP / signals.py / RUN_20261101_163156",
            asset_classes=[AssetClass.METAL, AssetClass.INDEX, AssetClass.FX],
            timeframes=["H1", "H4"],
            tags=["trend", "adaptive", "donchian"],
        ),
        IdeaEntry(
            idea_id="I-010", title="Свечные паттерны (поглощения/пин-бары)",
            description="Алгоритмический поиск свечных паттернов по методологии Нисона. "
                        "Поглощение + пин-бар на H4/D1 с подтверждением следующего бара.",
            status=IdeaStatus.IDEA, source=SourceType.MQL5,
            source_ref="MQL5-статьи PDF / свечные паттерны алгоритмически (ч.18)",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H4", "D1"],
            tags=["price-action", "candlestick", "patterns"],
        ),
        IdeaEntry(
            idea_id="I-011", title="Стоп по квантилю теней вместо фикс. пипсов",
            description="SL = 90-100% перцентиль длины тени H1 инструмента. "
                        "Динамический стоп вместо магических пипсов. Дрозд/Сафин.",
            status=IdeaStatus.IDEA, source=SourceType.MQL5,
            source_ref="MQL5-статьи / Дрозд/Сафин - стоп по теням",
            asset_classes=[AssetClass.FX, AssetClass.METAL], timeframes=["H1"],
            tags=["risk", "stop-loss", "shadow"],
        ),
        IdeaEntry(
            idea_id="I-012", title="Два контракта (Franz) - частичный выход",
            description="Половина позиции с быстрым выходом 3×ATR + половина по основной системе. "
                        "Снижает просадку, сохраняет участие в тренде.",
            status=IdeaStatus.IDEA, source=SourceType.RESEARCH,
            source_ref="STRATEGY_CATALOG.md / методологические находки",
            asset_classes=[AssetClass.ANY], timeframes=["H1", "H4", "D1"],
            tags=["position-mgmt", "partial-exit", "atr"],
        ),
        IdeaEntry(
            idea_id="I-013", title="Fakeout понедельника - fade первого пробоя",
            description="Первый день недели. ±20 пипс от утренней цены. "
                        "Первый пробой -> лимитка на противоположной границе. TP=50/SL=40.",
            status=IdeaStatus.IDEA, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / Fakeout понедельника",
            asset_classes=[AssetClass.FX], timeframes=["M15", "M30"],
            tags=["mean-reversion", "monday", "fakeout"],
        ),
        IdeaEntry(
            idea_id="I-014", title="Разворотный период - fade первых минут сессии",
            description="Вход против движения первых минут сессии в фиксированное время (9:50-10:10 ET). "
                        "Только FREE-профиль (intraday). US100 на M1.",
            status=IdeaStatus.IDEA, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / Wolski reversal period",
            asset_classes=[AssetClass.INDEX], timeframes=["M1"],
            tags=["mean-reversion", "intraday", "seasonality"],
        ),
        IdeaEntry(
            idea_id="I-015", title="Lead-lag коррелированных инструментов",
            description="Ведомый копирует лидера с лагом 1-5 мин. "
                        "Пример: XAUUSD тандем - лидер/ведомый. Кросс-инструментальный модуль.",
            status=IdeaStatus.IDEA, source=SourceType.RUFP,
            source_ref="STRATEGY_CATALOG.md / Lead-lag correlation",
            asset_classes=[AssetClass.METAL], timeframes=["M5", "M15"],
            tags=["correlation", "lead-lag", "cross-instrument"],
        ),
    ]

    for idea in seeds:
        registry.register(idea)

    return registry


# =============================================================================
# ПРЕДЗАПОЛНЕННЫЕ СТРАТЕГИИ (формализованные из идей)
# =============================================================================

def seed_strategies(ideas: IdeasRegistry) -> StrategyRegistry:
    """Создать формализованные стратегии из идей GO_CANDIDATE и REGISTERED."""
    registry = StrategyRegistry()

    seeds = [
        StrategyEntry(
            strategy_id="S-001", idea_id="I-001",
            name="Donchian n55 XAUUSD H1 session 7-20",
            indicator="donchian", params={"n": 55},
            filter_name="session", filter_params={"start_hour": 7, "end_hour": 20},
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 500, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["XAUUSD"], horizon="H1",
            walk_forward_result={"pf": 1.44, "t_stat": 2.93, "n_trades": 264, "max_dd": 3.08, "return_pct": 20.5},
            null_model_verdict="ALPHA",
            status=IdeaStatus.GO_CANDIDATE,
            notes="Лучший кандидат волны-1. 10/13 окон OOS в плюс.",
        ),
        StrategyEntry(
            strategy_id="S-002", idea_id="I-001",
            name="Donchian n55 XAUUSD H1 (без фильтра)",
            indicator="donchian", params={"n": 55},
            filter_name="none",
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 500, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["XAUUSD"], horizon="H1",
            walk_forward_result={"pf": 1.32, "t_stat": 2.52, "n_trades": 326, "max_dd": 2.55, "return_pct": 19.8},
            null_model_verdict="ALPHA",
            status=IdeaStatus.GO_CANDIDATE,
        ),
        StrategyEntry(
            strategy_id="S-003", idea_id="I-002",
            name="Vol Breakout k=0.5 USDJPY H1 session 7-20",
            indicator="vol_breakout", params={"k": 0.5, "atr_len": 24, "nr7": 0},
            filter_name="session", filter_params={"start_hour": 7, "end_hour": 20},
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 500, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["USDJPY"], horizon="H1",
            walk_forward_result={"pf": 1.83, "t_stat": 2.24, "n_trades": 57, "max_dd": 1.40, "return_pct": 5.2},
            null_model_verdict="ALPHA",
            status=IdeaStatus.GO_CANDIDATE,
            notes="Единственный не-дончиановский PASS. n=57 у нижней кромки.",
        ),
        StrategyEntry(
            strategy_id="S-004", idea_id="I-003",
            name="BB Fade 35/2.0 XAUUSD H4",
            indicator="bb_fade", params={"period": 35, "n_std": 2.0},
            filter_name="none",
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 200, "tp_pips": 400},
            risk_profile="FREE",
            instruments=["XAUUSD"], horizon="H4",
            status=IdeaStatus.REGISTERED,
            notes="Ждёт backtest в волне-2.",
        ),
        StrategyEntry(
            strategy_id="S-005", idea_id="I-008",
            name="Mean-Reversion Z-score MA50 entry 2.5σ H1",
            indicator="mean_reversion_zscore", params={"ma_period": 50, "entry_z": 2.5},
            filter_name="adx_min", filter_params={"adx_max": 25},
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 150, "tp_pips": 300},
            risk_profile="PIPFARM_0.3pct",
            instruments=["EURUSD", "GBPUSD", "XAUUSD"], horizon="H1",
            status=IdeaStatus.REGISTERED,
        ),
        StrategyEntry(
            strategy_id="S-006", idea_id="I-009",
            name="Adaptive Donchian const=150 XAUUSD H1",
            indicator="adaptive_donchian", params={"const": 150, "adx_len": 14},
            filter_name="none",
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 500, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["XAUUSD"], horizon="H1",
            backtest_result={"return_pct": 28.2, "sharpe": 0.8, "n_trades": 210},
            status=IdeaStatus.BACKTESTED,
            notes="In-sample +28.2% - ждёт walk-forward волны-2.",
        ),
        StrategyEntry(
            strategy_id="S-007", idea_id="I-001",
            name="Donchian n55 XAUUSD H1 trend_ema200",
            indicator="donchian", params={"n": 55},
            filter_name="trend_ema200",
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 500, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["XAUUSD"], horizon="H1",
            walk_forward_result={"pf": 1.27, "t_stat": 2.12, "n_trades": 330, "max_dd": 3.23, "return_pct": 16.9},
            null_model_verdict="ALPHA",
            status=IdeaStatus.GO_CANDIDATE,
            notes="EMA200 filter - лучший трендовый фильтр на OOS.",
        ),
        StrategyEntry(
            strategy_id="S-008", idea_id="I-001",
            name="Donchian n55 XAUUSD H1 session + sl=200",
            indicator="donchian", params={"n": 55},
            filter_name="session", filter_params={"start_hour": 7, "end_hour": 20},
            exit_name="fixed_sl_tp", exit_params={"sl_pips": 200, "tp_pips": 400},
            risk_profile="PIPFARM_0.5pct",
            instruments=["XAUUSD"], horizon="H1",
            walk_forward_result={"pf": 1.26, "t_stat": 2.32, "n_trades": 445, "max_dd": 2.46, "return_pct": 14.4},
            null_model_verdict="ALPHA",
            status=IdeaStatus.GO_CANDIDATE,
        ),
    ]

    for s in seeds:
        registry.register(s)
        # Обновить связь в реестре идей
        if s.idea_id in ideas:
            ideas.link_strategy(s.idea_id, s.strategy_id)

    return registry


# =============================================================================
# Единый реестр (идеи + стратегии)
# =============================================================================

class UnifiedRegistry:
    """
    Единый реестр: идеи + стратегии + перекрёстные ссылки.

    Использование:
      reg = UnifiedRegistry()
      reg.seed()                        # Заполнить из предзаполненных данных
      reg.ideas.register(IdeaEntry(...))  # Новая идея
      reg.strategies.register(StrategyEntry(...))  # Новая стратегия
      reg.ideas.transition("I-001", IdeaStatus.LIVE, "Запущена в бою")
      reg.save("research/catalogs/registry.json")
    """

    def __init__(self):
        self.ideas = IdeasRegistry()
        self.strategies = StrategyRegistry()

    def seed(self):
        """Заполнить реестры предзаполненными данными."""
        self.ideas = seed_ideas()
        self.strategies = seed_strategies(self.ideas)

    def summary(self) -> dict:
        return {
            "ideas": self.ideas.summary(),
            "strategies": self.strategies.summary(),
        }

    def save(self, directory: str) -> None:
        """Сохранить оба реестра в JSON."""
        os.makedirs(directory, exist_ok=True)
        self.ideas.save(os.path.join(directory, "ideas_registry.json"))
        self.strategies.save(os.path.join(directory, "strategies_registry.json"))

    def load(self, directory: str) -> None:
        """Загрузить оба реестра из JSON."""
        ideas_path = os.path.join(directory, "ideas_registry.json")
        strategies_path = os.path.join(directory, "strategies_registry.json")
        if os.path.exists(ideas_path):
            self.ideas.load(ideas_path)
        if os.path.exists(strategies_path):
            self.strategies.load(strategies_path)

    def go_candidates_report(self) -> str:
        """Отчёт по GO-кандидатам."""
        lines = ["=== GO-КАНДИДАТЫ ==="]
        for s in self.strategies.list_go_candidates():
            idea = self.ideas.get(s.idea_id) if s.idea_id in self.ideas else None
            wf = s.walk_forward_result or {}
            lines.append(f"\n{s.strategy_id}: {s.name}")
            lines.append(f"  Идея: {idea.title if idea else 'N/A'}")
            lines.append(f"  Инструменты: {s.instruments} | Горизонт: {s.horizon}")
            lines.append(f"  WF: PF={wf.get('pf','?')}, t={wf.get('t_stat','?')}, "
                         f"n={wf.get('n_trades','?')}, DD={wf.get('max_dd','?')}%, "
                         f"return={wf.get('return_pct','?')}%")
            lines.append(f"  Вердикт нуль-модели: {s.null_model_verdict}")
            lines.append(f"  Профиль: {s.risk_profile}")
        return "\n".join(lines)


# =============================================================================
# Демо
# =============================================================================

if __name__ == "__main__":
    reg = UnifiedRegistry()
    reg.seed()

    print("=" * 60)
    print("  РЕЕСТР ИДЕЙ И СТРАТЕГИЙ")
    print("=" * 60)

    summary = reg.summary()
    print(f"\nИДЕИ: {summary['ideas']['total']}")
    for status, count in summary['ideas']['by_status'].items():
        if count:
            print(f"  {status}: {count}")

    print(f"\nСТРАТЕГИИ: {summary['strategies']['total']}")
    print(f"  GO-кандидатов: {summary['strategies']['go_candidates']}")
    for ind, count in summary['strategies']['by_indicator'].items():
        print(f"  {ind}: {count}")

    print("\n" + reg.go_candidates_report())

    # Пример: новая идея
    print("\n" + "=" * 60)
    print("  ПРИМЕР: регистрация новой идеи")
    new_idea = IdeaEntry(
        idea_id="I-016",
        title="GARCH-волатильность -> динамический SL/TP",
        description="Использовать GARCH(1,1) прогноз волатильности для динамического "
                    "определения SL и TP. Чем выше прогнозная волатильность, тем шире стоп.",
        status=IdeaStatus.IDEA,
        source=SourceType.MQL5,
        source_ref="MQL5-статьи PDF / GARCH-волатильность",
        asset_classes=[AssetClass.FX, AssetClass.METAL],
        timeframes=["H1", "H4"],
        tags=["risk", "garch", "dynamic-sl", "volatility"],
    )
    reg.ideas.register(new_idea)
    print(f"  Зарегистрирована: {new_idea.idea_id} - {new_idea.title}")
    print(f"  Статус: {new_idea.status.value}")

    # Сохранить
    save_dir = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "registry_data")
    reg.save(save_dir)
    print(f"\n  Реестры сохранены в: {save_dir}")