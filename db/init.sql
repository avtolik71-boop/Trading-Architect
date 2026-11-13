-- Database initialisation for DeepSeek Trader
-- Creates TimescaleDB hypertables for trade journal, signals, and feature store.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Trade journal: every execution event
CREATE TABLE IF NOT EXISTS trade_journal (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    advisory_id     UUID,
    symbol          VARCHAR(10) NOT NULL,
    action          VARCHAR(12) NOT NULL,       -- BUY, SELL, CLOSE_LONG, CLOSE_SHORT, FLATTEN
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    volume_cents    BIGINT,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    confidence_tier VARCHAR(6),                  -- HIGH, MEDIUM, LOW
    risk_score      DOUBLE PRECISION,
    pnl_realised    DOUBLE PRECISION,
    pnl_currency    VARCHAR(3),
    commission      DOUBLE PRECISION,
    swap            DOUBLE PRECISION,
    signal_source   VARCHAR(20),                 -- lightgbm, patchtst, ensemble, manual
    revision_count  INTEGER DEFAULT 0,
    verdict         VARCHAR(10),                 -- approved, revised, rejected
    checks_passed   TEXT[],
    checks_failed   TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at       TIMESTAMPTZ
);

SELECT create_hypertable('trade_journal', 'created_at', if_not_exists => TRUE);

CREATE INDEX idx_tj_symbol ON trade_journal (symbol, created_at DESC);
CREATE INDEX idx_tj_verdict ON trade_journal (verdict, created_at DESC);
CREATE INDEX idx_tj_closed ON trade_journal (closed_at DESC) WHERE closed_at IS NOT NULL;

-- Signal log: every signal from the ML pipeline
CREATE TABLE IF NOT EXISTS signal_log (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       UUID NOT NULL UNIQUE,
    source          VARCHAR(20) NOT NULL,        -- lightgbm, patchtst, ensemble
    symbol          VARCHAR(10) NOT NULL,
    action          VARCHAR(12) NOT NULL,
    confidence      DOUBLE PRECISION,
    horizon         VARCHAR(5),                  -- 5m, 15m, 1h, 4h, 1d
    z_score         DOUBLE PRECISION,
    features        JSONB,
    consumed_by     UUID,                        -- trade_journal.trade_id
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('signal_log', 'created_at', if_not_exists => TRUE);

CREATE INDEX idx_sl_symbol ON signal_log (symbol, created_at DESC);
CREATE INDEX idx_sl_source ON signal_log (source, created_at DESC);

-- Advisory log: every DeepSeek advisory (including rejected)
CREATE TABLE IF NOT EXISTS advisory_log (
    id              BIGSERIAL PRIMARY KEY,
    advisory_id     UUID NOT NULL UNIQUE,
    signal_id       UUID,
    symbol          VARCHAR(10) NOT NULL,
    action          VARCHAR(12) NOT NULL,
    confidence_tier VARCHAR(6),
    risk_score      DOUBLE PRECISION,
    rationale       TEXT[],
    verdict         VARCHAR(10),                 -- approved, revised, rejected
    revision_count  INTEGER DEFAULT 0,
    raw_response    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('advisory_log', 'created_at', if_not_exists => TRUE);

CREATE INDEX idx_al_verdict ON advisory_log (verdict, created_at DESC);

-- Daily P&L snapshot
CREATE TABLE IF NOT EXISTS daily_pnl (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    opening_balance DOUBLE PRECISION,
    closing_balance DOUBLE PRECISION,
    realised_pnl    DOUBLE PRECISION,
    unrealised_pnl  DOUBLE PRECISION,
    trades_count    INTEGER DEFAULT 0,
    win_rate        DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_dpnl_date ON daily_pnl (date DESC);

-- Feature store: ML features snapshotted at each decision point
CREATE TABLE IF NOT EXISTS feature_snapshot (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(10) NOT NULL,
    horizon         VARCHAR(5) NOT NULL,
    features        JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('feature_snapshot', 'created_at', if_not_exists => TRUE);

CREATE INDEX idx_fs_symbol_horizon ON feature_snapshot (symbol, horizon, created_at DESC);