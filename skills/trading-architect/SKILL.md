---
name: trading-architect
description: Use when working inside the Trading Architect project — a Python mean-reversion-on-cTrader research/agent system with a DeepSeek V4-PRO advisory loop. Covers project structure, the populating indicator/ideas/strategies registries, extension points (add indicator, filter, idea, strategy), system prompts, risk gates, and how to integrate new methodology (e.g. Hurst) into it.
---

# Trading Architect — структура и точки расширения

Производственный контур: сигналы → DeepSeek V4-PRO advisory → self-review → risk gates → cTrader исполнение → пост-филл верификация. Mean-reversion на cTrader (1H).

## Структура

```
main.py                  # Entry point (CLI: health / dry-run / live / live --auto-execute)
core/agent_loop.py       # self-review agent loop (DeepSeek V4-PRO)
core/system_prompts.py   # SYSTEM_PROMPT + REVIEWER_PROMPT (JSON-формат advisory)
bridge/execution_bridge.py  # Python → DeepSeek → cTrader REST Proxy (порт 8443)
ml/pipeline.py           # LightGBM + PatchTST (regime) + признаки
risk/risk_gates.py       # 8 risk gates + rollback ledger
config/symbols.json      # точность символов
config/correlation_matrix.json  # FX-корреляции
db/init.sql              # TimescaleDB
research/                # ИССЛЕДОВАТЕЛЬСКИЙ КОНТУР (реестры ниже)
  indicators/indicator_registry.py
  catalogs/idea_strategy_registry.py, catalogs.py
  risk_mgmt/risk_profiles.py
  testing/dsr.py, null_models.py
  costs/cost_engine.py
  run_pipeline.py, smoke_test.py
tests/test_core.py
```

## Ключевые параметры (HARD)

`DAILY_LOSS_LIMIT_PCT=5%`, `MAX_POSITIONS=6`, `PER_TRADE_RISK_PCT=1.5%`, `MIN_TP_SL_RATIO=1.5`, `ENSEMBLE_THRESHOLD_HIGH=0.75`, `ENSEMBLE_THRESHOLD_LOW=0.55`.

## Реестр индикаторов — `research/indicators/indicator_registry.py`

Принцип: **ОДИН индикатор → СЕТКА параметров → СЕМЕЙСТВО кандидатов.** Добавление нового индикатора — только в `INDICATOR_REGISTRY`, другой код не трогать.

Единица добавления:
```python
IndicatorDef("name", "Display name", "category",  # category: momentum/trend/volatility/volume/mean_reversion/cycle/baseline
             signal_fn,                            # (df, **params) -> pd.Series[int] (-1,0,1)
             {"param": [values...]},               # param_grid
             [AssetClass.FX, AssetClass.METAL],    # compatible_asset_classes
             min_timeframe="M5", source="manual", description="...")
```
Затем `register_indicator(...)`; пайплайн (`IndicatorEnumerator`) сам развернёт в кандидатов. Вспомогательные ряды (`_ema`, `_rsi`, `_atr`, `_adx`, `_macd`, `_bb`, `_stoch`, `_cci`) — строительные блоки.

Манифесты зафиксированных методов: `EXISTING_SIGNAL_TEMPLATES`, `EXISTING_FILTERS` (none/trend_ema200/session/adx_min/nr7/volume_min/day_of_week), `EXISTING_EXITS`, `EXISTING_RISK_PROFILES`, `EXISTING_TESTING_METHODS`, `EXISTING_COST_COMPONENTS`, сводный `METHOD_MANIFEST`.

## Реестр идей/стратегий — `research/catalogs/idea_strategy_registry.py`

Жизненный цикл идеи: `IDEA → REGISTERED → BACKTESTED → WALK_FORWARD → GO_CANDIDATE → LIVE → PAUSED/RETIRED` (+ `REJECTED`). Переходы только по `STATUS_TRANSITIONS`.

- `IdeasRegistry.register(IdeaEntry(...))` — идея (idea_id, title, description, source, tags...).
- `StrategyRegistry.register(StrategyEntry(...))` — формализованная стратегия, привязана к идее (indicator, params, filter_name, exit_name, risk_profile, instruments, horizon).
- `UnifiedRegistry.seed()` заполняет из `seed_ideas()` / `seed_strategies()`.
- Метрики: `walk_forward_result` {pf, t_stat, n_trades, max_dd, return_pct}, вердикт `null_model_verdict` = ALPHA / BETA_OR_TIMING / NO_EDGE.

## Системные промпты — `core/system_prompts.py`

`SYSTEM_PROMPT` (архитектор: mean-reversion, confluence ≥2, regime-фильтр «не fade при ADX>35 со slope», сессии, time-stop) и `REVIEWER_PROMPT` (критик: 8 чеков, вердикт approved/rejected/revised). Вывод — строго JSON. LOW confidence → advisory-only.

## Как интегрировать методику Херста

1. **Индикатор** (сигнал по режиму): добавить `hurst_regime_signal` + `IndicatorDef("hurst_regime", ...)` в `indicator_registry.py` (реализовано).
2. **Фильтр**: добавить `"hurst_regime"` в `EXISTING_FILTERS` (реализовано) — режим H(t) как шлюз для трендовых/контртрендовых стратегий.
3. **Идея**: зарегистрировать `IdeaEntry` в `seed_ideas()` (реализовано: I-016).
4. **Промпт**: упомянуть режим Херста в `SYSTEM_PROMPT` как дополнительный regime-фильтр (H>0,58 тренд / H<0,42 ranging).
5. Риск: масштабировать `σ(T)=σ(1)·T^H` в `risk_gates.py` / позиционном калькуляторе.

Помнить про cTrader quircks (из skill `ctrader-mcp-servers`): Q-R4 MARKET со SL/TP может быть REJECTED; Q-R10 `amend_position` без SL удаляет существующий SL.
