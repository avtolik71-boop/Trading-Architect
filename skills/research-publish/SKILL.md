---
name: research-publish
description: Use when research results (scan / walk-forward CSVs in research/reports/) must be automatically added to the strategy catalog and the dashboard — i.e. after running run_pipeline.py or run_walkforward.py, or when asked to "publish", "add results to catalog", "update the dashboard", or "auto-add research findings". Covers the auto_publish.py pipeline, the strategy registry JSON, the results ledger, the self-contained HTML dashboard, and the RUFP dashboard bridge.
---

# Авто-публикация результатов исследований

После каждого прогона исследования результаты должны автоматически попадать в каталог стратегий и дашборд. Это делает `research/auto_publish.py`.

## Когда запускать

- Автоматически: `run_walkforward.py` вызывает `publish()` в конце прогона.
- Вручную: `python auto_publish.py` (читает последние `scan_*.csv` и `walkforward_*.csv` из `reports/`).

## Что делает `publish()`

| Шаг | Куда | Результат |
|-----|------|-----------|
| 1. `update_strategy_catalog()` | `catalogs/registry_data/strategies_registry.json` | Добавляет/обновляет `StrategyEntry`: `walk_forward_result`, `null_model_verdict`, `status` (`WALK_FORWARD` / `GO_CANDIDATE`) |
| 2. `append_results_ledger()` | `reports/results_ledger.csv` | Накопительный журнал всех прогонов (дедупликация по indicator+params) |
| 3. `build_dashboard_html()` | `reports/research_dashboard.html` | Самодостаточный HTML (тёмная тема, KPI, топ по t/DSR, честный вывод) |
| 4. `write_rufp_bridge()` | `C:\Users\Autolycus-FX\RUFP PROJECT\ДАШБОРД-СТРАТЕГИИ\research_results_auto.md` | Сводка для RUFP-дашборда |

## Идемпотентность

Повторный запуск НЕ дублирует:
- Стратегии сопоставляются по ключу `indicator | json.dumps(params, sort_keys=True)`.
- Журнал дедуплицируется по тому же ключу (старые строки заменяются свежими).
- RUFP-мост перезаписывается целиком (markdown).

## Расширение

- **Новый дашборд-вид**: добавить колонки в `build_dashboard_html()`.
- **Новый канал (Telegram/email)**: добавить функцию в `publish()` по аналогии с `write_rufp_bridge()`.
- **Смена брокера издержек**: `run_pipeline.py` → `PipelineConfig.cost_set` (`PIPFARM_CTRADER` / `TICKMILL`).

## Связь с остальным проектом

- Данные прогона: `run_pipeline.py` (scan) → `run_walkforward.py` (OOS) → `auto_publish.py` (публикация).
- Каталог стратегий: `catalogs/idea_strategy_registry.py` (`UnifiedRegistry`) — Python-реестр; JSON в `registry_data/` — снапшот для авто-публикации.
- Дашборд RUFP: внешний проект `RUFP PROJECT\ДАШБОРД-СТРАТЕГИИ` — `build_dashboard.py` собирает `strategies_dashboard.html` из `ideas.csv` и ledger'а robot-lab. Наш `research_results_auto.md` — мост для ручной сверки.
