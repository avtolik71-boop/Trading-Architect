# DeepSeek V4-PRO Trading Architect — Mean-Reversion на cTrader
# ========================================================================

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.6+-green.svg)](https://github.com/vllm-project/vllm)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3+-orange.svg)](https://lightgbm.readthedocs.io/)
[![cTrader](https://img.shields.io/badge/cTrader-REST%20Proxy-red.svg)](https://ctrader.com/)

**Производственный контур**: сигналы → DeepSeek V4-PRO advisory → self-review → risk gates → cTrader исполнение → пост-филл верификация.

## Архитектура

```
┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│  LightGBM    │───▶│  PatchTST     │───▶│  Ensemble      │
│  Classifier  │    │  Regime Model │    │  Score ≥ 0.55  │
└──────────────┘    └───────────────┘    └───────┬────────┘
                                                  │
                                          ┌───────▼────────┐
                                          │  DeepSeek V4   │
                                          │  Advisory Gen  │
                                          └───────┬────────┘
                                                  │
                                          ┌───────▼────────┐
                                          │  Self-Review   │
                                          │  (2nd LLM pass)│
                                          └───────┬────────┘
                                                  │
                                    ┌─────────────┼─────────────┐
                                    ▼             ▼             ▼
                              APPROVED       REVISED       REJECTED
                                    │             │             │
                              ┌─────▼─────┐ ┌───▼────┐  ┌─────▼─────┐
                              │ Risk Gate │ │Rephrase│  │  Log +    │
                              │ Engine    │ │  N≤2   │  │  HOLD     │
                              └─────┬─────┘ └───┬────┘  └───────────┘
                                    │           │
                              ┌─────▼─────┐     │
                              │  cTrader  │◄────┘
                              │  Execute  │
                              └─────┬─────┘
                                    │
                              ┌─────▼─────┐
                              │ Post-Fill │
                              │  Verify   │
                              └───────────┘
```

## Требования к железу

| Компонент | Минимум | Рекомендация |
|-----------|---------|--------------|
| GPU | RTX 3090/4080 24GB | A100 80GB |
| CPU | 16-core | 32-core |
| RAM | 64 GB | 128 GB |
| Disk | 100 GB SSD | 500 GB NVMe |

## Быстрый старт

### 1. Клонирование + окружение

```bash
git clone <repo> deepseek-trader
cd deepseek-trader
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux
pip install -r requirements.txt
```

### 2. vLLM сервер (Docker)

```bash
# Скачайте модель DeepSeek V4-PRO AWQ в ./models/
# Запуск:
docker compose up -d vllm redis postgres
# Проверка:
python main.py health
```

### 3. cTrader

Убедитесь, что cTrader запущен на Windows-хосте и REST Proxy активен на порту 8443.

```bash
# Создайте .env из шаблона:
copy .env.example .env
# Заполните CTRADER_ACCESS_TOKEN и CTRADER_ACCOUNT_ID
```

### 4. Dry run (без исполнения)

```bash
python main.py dry-run
```

### 5. Live trading

```bash
# Бумажная торговля (без авто-исполнения):
python main.py live

# Боевой режим (⚠️ РЕАЛЬНЫЕ ДЕНЬГИ):
python main.py live --auto-execute
```

## Структура проекта

```
deepseek-trader/
├── main.py                  # Entry point (CLI)
├── docker-compose.yml       # vLLM + Redis + PG + Grafana
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── core/
│   ├── agent_loop.py        # DeepSeek self-review agent loop
│   └── system_prompts.py    # System + reviewer prompts
├── bridge/
│   └── execution_bridge.py  # Python → DeepSeek → cTrader bridge
├── ml/
│   └── pipeline.py          # LightGBM + PatchTST + features
├── risk/
│   └── risk_gates.py        # 8 risk gates + rollback ledger
├── config/
│   ├── symbols.json         # Symbol precision table
│   └── correlation_matrix.json  # FX correlation matrix
├── db/
│   └── init.sql             # TimescaleDB schema
├── prometheus/
│   └── prometheus.yml       # Metrics scrape config
├── tests/
│   └── test_core.py         # Unit tests (risk gates + agent)
└── logs/                    # Runtime logs
```

## Ключевые параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `DAILY_LOSS_LIMIT_PCT` | 5.0% | Дневной лимит убытков → flatten |
| `MAX_POSITIONS` | 6 | Максимум одновременных позиций |
| `PER_TRADE_RISK_PCT` | 1.5% | Риск на сделку (% от equity) |
| `MIN_TP_SL_RATIO` | 1.5 | Минимальное отношение TP:SL |
| `ENSEMBLE_THRESHOLD_HIGH` | 0.75 | Высокая уверенность |
| `ENSEMBLE_THRESHOLD_LOW` | 0.55 | Низкая уверенность (advisory only) |

## LightGBM Hyperparameters (Mean-Reversion, 1H)

- `num_leaves`: 31
- `learning_rate`: 0.02
- `n_estimators`: 500 (early stopping на 50)
- `lambda_l1`: 0.1, `lambda_l2`: 1.0
- `feature_fraction`: 0.7, `bagging_fraction`: 0.8
- `scale_pos_weight`: динамический (class balance)

## PatchTST Hyperparameters (Regime Detection)

- `d_model`: 128, `n_heads`: 8, `n_layers`: 3
- `context_window`: 96 bars, `prediction_length`: 24 bars
- `patch_len`: 8, `stride`: 4
- `precision`: 16-mixed (экономит VRAM)

## cTrader Quirks (из ctrader-mcp-servers skill)

См. `Q-R4`: MARKET ордера с абсолютным SL/TP могут быть REJECTED.
См. `Q-R10`: `amend_position` без указания SL удаляет существующий SL.

## Лицензия

Proprietary. All rights reserved.

## Контакты

По вопросам развёртывания — свяжитесь с командой.