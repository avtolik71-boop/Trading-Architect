"""
Каталоги стратегий, позиционного менеджмента, методов тестирования
===================================================================
ПОПОЛНЯЕМЫЕ БАЗЫ — добавлять новые методы сюда, не дублировать существующие.

Формат: Markdown-файлы с YAML-подобной структурой для машинной читаемости
+ человекочитаемое описание.

Связанные файлы:
  - indicators/indicator_registry.py  — реестр индикаторов
  - costs/cost_engine.py              — издержки
  - testing/null_models.py            — нуль-модели
  - risk_mgmt/risk_profiles.py        — риск-профили
  - catalogs/strategies.md            — каталог стратегий
  - catalogs/position_methods.md      — методы управления позицией
  - catalogs/risk_methods.md          — методы управления рисками
  - catalogs/testing_methods.md       — методы тестирования и оптимизации
"""

# =============================================================================
# КАТАЛОГ СТРАТЕГИЙ
# =============================================================================

STRATEGIES_CATALOG = {
    # Из GENERATOR RUFP / STRATEGY_CATALOG.md — ПРИОРИТЕТ-1 (уже реализованы)
    "volatility_breakout": {
        "name": "Волатильный пробой (Линда Рашке)",
        "indicator": "vol_breakout",
        "status": "IMPLEMENTED",
        "instruments": ["XAUUSD", "US100", "GBPUSD", "USDJPY"],
        "horizons": ["H1", "H4", "D1"],
        "description": "Стоп-вход open ± k×ATR(n). Фильтр NR7 опционально.",
        "params_grid": {"k": [0.5, 1.0, 1.5], "atr_len": [24, 96], "nr7": [0, 1]},
        "walk_forward_status": "USDJPY vol_breakout k=0.5 → PF=1.83 OOS — GO-кандидат",
    },
    "donchian": {
        "name": "Дончиан-канал (пробой N-дневного диапазона)",
        "indicator": "donchian",
        "status": "IMPLEMENTED",
        "instruments": ["XAUUSD", "BTCUSD", "US100"],
        "horizons": ["H1", "H4", "D1"],
        "description": "Пробой N-дневного high/low. Always-in разворотная.",
        "params_grid": {"n": [20, 55, 96]},
        "walk_forward_status": "XAUUSD donchian n55 → 4/4 варианта PF 1.26-1.44 OOS — GO-кандидат",
    },
    "adaptive_donchian": {
        "name": "Адаптивный Дончиан (Riccardo)",
        "indicator": "adaptive_donchian",
        "status": "IMPLEMENTED",
        "instruments": ["XAUUSD", "US100", "GBPUSD"],
        "horizons": ["H1", "H4", "D1"],
        "description": "Длина канала X = const/ADX(14), clamp(1,40). Разворотная always-in.",
        "params_grid": {"const": [100, 150, 200], "adx_len": [14]},
        "walk_forward_status": "In-sample XAUUSD +28.2% — ждёт walk-forward волны-2",
    },

    # Из STRATEGY_CATALOG.md — НОВЫЕ (требуют реализации во волне-2)
    "bb_fade": {
        "name": "Bollinger Fade (пробой → возврат → fade)",
        "indicator": "bb_fade",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD", "USDJPY"],
        "horizons": ["H4", "D1"],
        "description": "Пробой BB(35,2) → возврат внутрь → fade против пробоя. Опц. RSI-подтверждение.",
        "params_grid": {"period": [20, 35], "n_std": [2.0, 2.5]},
    },
    "holy_grail": {
        "name": "Holy Grail (Линда Рашке/Тереза Ло)",
        "indicator": "trend_pullback_adx",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD"],
        "horizons": ["H4", "D1"],
        "description": "ADX(14) растёт → откат к EMA(20) → закрытие в сторону тренда → вход с трейлингом.",
    },
    "elder_triple_screen": {
        "name": "Тройной экран Элдера (упрощённый)",
        "indicator": "rsi + macd",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD", "USDJPY"],
        "horizons": ["D1+H4", "D1+H1"],
        "description": "Старший ТФ: MACD-гистограмма. Средний ТФ: RSI(14) <30 восстанавливается → long.",
    },
    "macd_pullback_mtf": {
        "name": "MACD Pullback мульти-ТФ",
        "indicator": "macd_histogram",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD"],
        "horizons": ["D1+H4"],
        "description": "D1: MACD(5,13,8) > 0 = тренд. H4: кросс вниз = откат. Вход при развороте.",
    },
    "ema_separation": {
        "name": "MA Separation (вместо кросса)",
        "indicator": "ema_separation",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD"],
        "horizons": ["H4", "D1"],
        "description": "Обе EMA в одну сторону + расстояние ≥ m×ATR → вход. Сигнал в середине тренда.",
    },
    "pivot_daily": {
        "name": "Дневные пивоты (fade/breakout)",
        "indicator": "pivot_daily",
        "status": "REGISTERED",
        "instruments": ["XAUUSD", "GBPUSD", "US100"],
        "horizons": ["M15", "H1"],
        "description": "P=(H+L+C)/3. Fade на R1/S1 или breakout с подтверждением 2 баров.",
    },

    # Эталонные базы (ОБЯЗАТЕЛЬНЫ для каждого прогона)
    "random_entry": {
        "name": "Случайный вход — ЭТАЛОННАЯ БАЗА",
        "indicator": "random_entry",
        "status": "IMPLEMENTED",
        "description": "Случайный вход с prob=0.002. Все стратегии ОБЯЗАНЫ бить эту базу OOS.",
    },
    "buy_and_hold": {
        "name": "Buy & Hold — ЭТАЛОННАЯ БАЗА",
        "indicator": "buy_and_hold",
        "status": "IMPLEMENTED",
        "description": "Пассивное удержание. Стратегия с издержками должна давать альфу сверх дрейфа.",
    },
}


# =============================================================================
# КАТАЛОГ МЕТОДОВ УПРАВЛЕНИЯ ПОЗИЦИЕЙ
# =============================================================================

POSITION_METHODS = {
    "fixed_sl_tp": {
        "name": "Фиксированные SL и TP",
        "status": "IMPLEMENTED",
        "description": "SL на расстоянии N пипсов от входа, TP = M пипсов. Базовый метод.",
        "params": {"sl_pips": "int", "tp_pips": "int"},
    },
    "trailing_atr": {
        "name": "Трейлинг-стоп по ATR",
        "status": "IMPLEMENTED",
        "description": "Стоп двигается за ценой на расстоянии k×ATR от экстремума.",
        "params": {"atr_period": 14, "k": 2.0},
    },
    "chandelier_exit": {
        "name": "Люстра (Chandelier Exit)",
        "status": "REGISTERED",
        "description": "Long: highest high − k×ATR. Short: lowest low + k×ATR. Эталонный трейлинг.",
        "params": {"k": 3.0},
    },
    "time_exit": {
        "name": "Выход по времени",
        "status": "IMPLEMENTED",
        "description": "Закрыть позицию через N баров независимо от цены.",
        "params": {"bars": 20},
    },
    "reverse_signal": {
        "name": "Выход по противоположному сигналу",
        "status": "IMPLEMENTED",
        "description": "Закрыть long при short-сигнале и наоборот.",
    },
    "two_contracts_franz": {
        "name": "Два контракта (Franz)",
        "status": "REGISTERED",
        "description": "Половина позиции с быстрым выходом (3×ATR) + половина по системе.",
    },
    "shadow_quantile_stop": {
        "name": "Стоп по квантилю теней",
        "status": "REGISTERED",
        "description": "SL = 90-100% перцентиль длины тени H1. Вместо магических пипсов.",
        "source": "Дрозд/Сафин — MQL5-статьи",
    },
}


# =============================================================================
# КАТАЛОГ МЕТОДОВ УПРАВЛЕНИЯ РИСКАМИ
# =============================================================================

RISK_METHODS = {
    "fixed_fractional": {
        "name": "Фиксированная доля капитала",
        "status": "IMPLEMENTED",
        "description": "Объём = (equity × risk_pct) / (SL_distance × pip_value). Стандарт.",
    },
    "daily_loss_limit": {
        "name": "Дневной лимит убытков",
        "status": "IMPLEMENTED",
        "description": "При достижении дневного убытка N% → закрыть всё, остановиться до конца дня.",
        "profiles": ["PIPFARM"],
    },
    "daily_target_halt": {
        "name": "Дневная цель → остановка",
        "status": "IMPLEMENTED",
        "description": "При достижении дневной прибыли N% → закрыть всё, остановиться.",
        "profiles": ["PIPFARM"],
    },
    "total_drawdown_limit": {
        "name": "Лимит общей просадки",
        "status": "IMPLEMENTED",
        "description": "При просадке от пика > N% → закрыть всё, оповещение.",
        "profiles": ["PIPFARM", "FREE"],
    },
    "correlation_gate": {
        "name": "Корреляционные ворота",
        "status": "REGISTERED",
        "description": "Не открывать позицию, если коррелированная экспозиция > 70% equity.",
    },
    "news_blackout": {
        "name": "Новостной блэкаут",
        "status": "REGISTERED",
        "description": "Не открывать позиции за ±30 мин до/после NFP, FOMC, CPI.",
    },
    "no_martingale": {
        "name": "Запрет усреднения против позиции",
        "status": "IMPLEMENTED",
        "description": "Никогда не увеличивать позицию в убытке. Жёсткое правило PIPFARM.",
    },
}


# =============================================================================
# КАТАЛОГ МЕТОДОВ ТЕСТИРОВАНИЯ И ОПТИМИЗАЦИИ
# =============================================================================

TESTING_METHODS = {
    "walk_forward": {
        "name": "Walk-Forward анализ",
        "status": "IMPLEMENTED",
        "description": "train 18600 / test 3100 / embargo 48, 13-15 окон. Золотой стандарт OOS-валидации. "
                       "Ворота: t>=2, PF>=1.1, mean>0, n>=50, beat random_entry OOS.",
        "source": "validation/walkforward.py + run_walkforward.py",
    },
    "simple_split": {
        "name": "Простое разделение",
        "status": "IMPLEMENTED",
        "description": "2016–2020 in-sample, 2021–2026 out-of-sample. Быстрая проверка.",
    },
    "monte_carlo": {
        "name": "Монте-Карло симуляция",
        "status": "REGISTERED",
        "description": "Перемешивание сделок × 1000. Проверка устойчивости.",
    },
    "random_entry_baseline": {
        "name": "База: случайный вход",
        "status": "IMPLEMENTED",
        "description": "30 сидов случайного входа. Медиана = минимальный барьер.",
        "source": "testing/null_models.py",
    },
    "buy_and_hold_baseline": {
        "name": "База: Buy & Hold",
        "status": "IMPLEMENTED",
        "description": "Пассивное удержание. Альфа должна быть сверх дрейфа.",
    },
    "dsr_penalty": {
        "name": "Deflated Sharpe Ratio",
        "status": "IMPLEMENTED",
        "description": "Поправка на множественное тестирование. DSR > 0.95 → значим, 0.80-0.95 → пограничный, < 0.80 → оверфит. "
                       "Haircut = E[max SR] среди N попыток (Bailey & López de Prado 2014).",
        "source": "testing/dsr.py | Bailey & López de Prado (2014)",
    },
    "sensitivity_analysis": {
        "name": "Анализ чувствительности",
        "status": "REGISTERED",
        "description": "Сетка параметров ±20% от оптимальных. Проверка робастности.",
    },
}


# =============================================================================
# Экспорт каталогов
# =============================================================================

ALL_CATALOGS = {
    "strategies": STRATEGIES_CATALOG,
    "position_methods": POSITION_METHODS,
    "risk_methods": RISK_METHODS,
    "testing_methods": TESTING_METHODS,
}


def catalog_summary():
    """Печать сводки всех каталогов."""
    for name, cat in ALL_CATALOGS.items():
        implemented = sum(1 for v in cat.values() if v["status"] == "IMPLEMENTED")
        registered = sum(1 for v in cat.values() if v["status"] == "REGISTERED")
        print(f"{name}: {len(cat)} всего ({implemented} реализовано, {registered} зарегистрировано)")


if __name__ == "__main__":
    catalog_summary()
    print("\n=== Стратегии ===")
    for k, v in STRATEGIES_CATALOG.items():
        wf = v.get("walk_forward_status", "")
        print(f"  [{v['status']:12s}] {v['name'][:45]:45s} | {v.get('indicator','')[:20]:20s} | {wf[:60]}")