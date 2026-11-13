"""
Risk Gate Layer — Deterministic Hard Constraints
================================================
Все проверки ДО того, как ордер уйдёт в cTrader.
Никакого LLM в этом слое — только детерминированные правила.

Gates (в порядке выполнения):
  1. Daily loss limit       — 5% от opening balance → flatten
  2. Max position count     — ≤ 6 concurrent
  3. Per-trade risk cap     — ≤ 1.5% equity
  4. TP:SL ratio            — ≥ 1.5:1
  5. Margin level           — ≥ 200%
  6. Correlation exposure   — ≤ 70%
  7. News blackout window   — ±30 min around major events
  8. Volume step validation — кратно lotSize / stepVolume
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger("deepseek.risk")


class GateResult(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    WARN = "warn"


@dataclass
class GateVerdict:
    gate_name: str
    result: GateResult
    detail: str
    threshold: Optional[float] = None
    actual_value: Optional[float] = None


@dataclass
class RiskGateReport:
    """Aggregate result of all risk gates."""
    all_passed: bool
    verdicts: list[GateVerdict]
    blocked_by: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.all_passed


class RiskGateEngine:
    """
    Hard risk gate engine.

    Configuration:
      daily_loss_limit_pct   — max daily loss as % of opening balance (default 5%)
      max_positions          — max concurrent open positions (default 6)
      per_trade_risk_pct     — max risk per trade as % of equity (default 1.5%)
      min_tp_sl_ratio        — minimum TP:SL ratio (default 1.5)
      min_margin_level_pct   — minimum margin level % (default 200%)
      max_correlation_exposure — max % of equity in correlated positions (default 70%)
      volume_step             — minimum volume increment in cents
      news_blackout_minutes   — ± minutes around major news events (default 30)
    """

    # Major news events (UTC times, ISO 8601 dates)
    # In production, fetch from an economic calendar API.
    MAJOR_NEWS_EVENTS: list[dict] = [
        # Format: {"date": "2026-11-14", "time": "13:30", "event": "US CPI", "timezone": "UTC"}
    ]

    def __init__(
        self,
        daily_loss_limit_pct: float = 5.0,
        max_positions: int = 6,
        per_trade_risk_pct: float = 1.5,
        min_tp_sl_ratio: float = 1.5,
        min_margin_level_pct: float = 200.0,
        max_correlation_exposure: float = 0.70,
        volume_step_cents: int = 1000,
        news_blackout_minutes: int = 30,
    ):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_positions = max_positions
        self.per_trade_risk_pct = per_trade_risk_pct
        self.min_tp_sl_ratio = min_tp_sl_ratio
        self.min_margin_level_pct = min_margin_level_pct
        self.max_correlation_exposure = max_correlation_exposure
        self.volume_step_cents = volume_step_cents
        self.news_blackout_minutes = news_blackout_minutes

    # ------------------------------------------------------------------
    # Gate 1: Daily Loss Limit
    # ------------------------------------------------------------------

    def gate_daily_loss(
        self,
        opening_balance: float,
        current_equity: float,
        daily_realised_pnl: float,
    ) -> GateVerdict:
        loss_pct = (opening_balance - current_equity) / opening_balance * 100 if opening_balance > 0 else 0
        if loss_pct >= self.daily_loss_limit_pct:
            return GateVerdict(
                gate_name="daily_loss_limit",
                result=GateResult.BLOCK,
                detail=f"Daily loss {loss_pct:.2f}% exceeds {self.daily_loss_limit_pct}% limit. FLATTEN EVERYTHING.",
                threshold=self.daily_loss_limit_pct,
                actual_value=loss_pct,
            )
        if loss_pct >= self.daily_loss_limit_pct * 0.75:
            return GateVerdict(
                gate_name="daily_loss_limit",
                result=GateResult.WARN,
                detail=f"Daily loss {loss_pct:.2f}% approaching {self.daily_loss_limit_pct}% limit. Reduce size.",
                threshold=self.daily_loss_limit_pct,
                actual_value=loss_pct,
            )
        return GateVerdict(
            gate_name="daily_loss_limit",
            result=GateResult.PASS,
            detail=f"Daily loss {loss_pct:.2f}% within limit.",
            threshold=self.daily_loss_limit_pct,
            actual_value=loss_pct,
        )

    # ------------------------------------------------------------------
    # Gate 2: Max Position Count
    # ------------------------------------------------------------------

    def gate_position_count(self, current_count: int, action_is_open: bool) -> GateVerdict:
        if not action_is_open:
            return GateVerdict(
                gate_name="position_count",
                result=GateResult.PASS,
                detail="Not an opening action.",
            )
        if current_count >= self.max_positions:
            return GateVerdict(
                gate_name="position_count",
                result=GateResult.BLOCK,
                detail=f"Already at max positions ({current_count}/{self.max_positions}).",
                threshold=float(self.max_positions),
                actual_value=float(current_count),
            )
        return GateVerdict(
            gate_name="position_count",
            result=GateResult.PASS,
            detail=f"Position count {current_count}/{self.max_positions} OK.",
            threshold=float(self.max_positions),
            actual_value=float(current_count),
        )

    # ------------------------------------------------------------------
    # Gate 3: Per-Trade Risk Cap
    # ------------------------------------------------------------------

    def gate_per_trade_risk(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        volume_cents: int,
        pip_value_per_lot_cents: float = 10.0,   # $ per pip per lot (forex majors)
        pip_size: float = 0.0001,               # 1 pip in price units (0.0001 for 5-digit forex)
        lot_size: int = 100000,
    ) -> GateVerdict:
        if equity <= 0:
            return GateVerdict(
                gate_name="per_trade_risk",
                result=GateResult.BLOCK,
                detail="Equity is zero or negative.",
            )
        sl_distance_abs = abs(entry_price - stop_loss)
        sl_pips = sl_distance_abs / pip_size
        lots = volume_cents / (lot_size * 100)
        risk_amount = sl_pips * lots * pip_value_per_lot_cents
        risk_pct = risk_amount / equity * 100

        if risk_pct > self.per_trade_risk_pct:
            return GateVerdict(
                gate_name="per_trade_risk",
                result=GateResult.BLOCK,
                detail=f"Per-trade risk {risk_pct:.2f}% exceeds {self.per_trade_risk_pct}% cap.",
                threshold=self.per_trade_risk_pct,
                actual_value=risk_pct,
            )
        return GateVerdict(
            gate_name="per_trade_risk",
            result=GateResult.PASS,
            detail=f"Per-trade risk {risk_pct:.2f}% within {self.per_trade_risk_pct}% cap.",
            threshold=self.per_trade_risk_pct,
            actual_value=risk_pct,
        )

    # ------------------------------------------------------------------
    # Gate 4: TP:SL Ratio
    # ------------------------------------------------------------------

    def gate_tp_sl_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> GateVerdict:
        sl_dist = abs(entry_price - stop_loss)
        tp_dist = abs(take_profit - entry_price)
        if sl_dist <= 0:
            return GateVerdict(
                gate_name="tp_sl_ratio",
                result=GateResult.BLOCK,
                detail="Stop loss distance is zero or negative.",
            )
        ratio = tp_dist / sl_dist
        if ratio < self.min_tp_sl_ratio:
            return GateVerdict(
                gate_name="tp_sl_ratio",
                result=GateResult.BLOCK,
                detail=f"TP:SL ratio {ratio:.2f} below minimum {self.min_tp_sl_ratio}.",
                threshold=self.min_tp_sl_ratio,
                actual_value=ratio,
            )
        return GateVerdict(
            gate_name="tp_sl_ratio",
            result=GateResult.PASS,
            detail=f"TP:SL ratio {ratio:.2f} ≥ {self.min_tp_sl_ratio}.",
            threshold=self.min_tp_sl_ratio,
            actual_value=ratio,
        )

    # ------------------------------------------------------------------
    # Gate 5: Margin Level
    # ------------------------------------------------------------------

    def gate_margin_level(self, margin_level_pct: float) -> GateVerdict:
        if margin_level_pct < self.min_margin_level_pct:
            return GateVerdict(
                gate_name="margin_level",
                result=GateResult.BLOCK,
                detail=f"Margin level {margin_level_pct:.0f}% below {self.min_margin_level_pct:.0f}% safety threshold.",
                threshold=self.min_margin_level_pct,
                actual_value=margin_level_pct,
            )
        if margin_level_pct < self.min_margin_level_pct * 1.5:
            return GateVerdict(
                gate_name="margin_level",
                result=GateResult.WARN,
                detail=f"Margin level {margin_level_pct:.0f}% approaching {self.min_margin_level_pct:.0f}% threshold.",
                threshold=self.min_margin_level_pct,
                actual_value=margin_level_pct,
            )
        return GateVerdict(
            gate_name="margin_level",
            result=GateResult.PASS,
            detail=f"Margin level {margin_level_pct:.0f}% healthy.",
            threshold=self.min_margin_level_pct,
            actual_value=margin_level_pct,
        )

    # ------------------------------------------------------------------
    # Gate 6: Correlation Exposure
    # ------------------------------------------------------------------

    def gate_correlation_exposure(
        self,
        equity: float,
        positions: list[dict],
        new_symbol: str,
        correlation_matrix: dict[str, dict[str, float]],
    ) -> GateVerdict:
        """
        Check if adding a new position would push correlated exposure > threshold.

        Correlation matrix: {symbol: {other_symbol: correlation_coefficient}}
        """
        if equity <= 0:
            return GateVerdict(
                gate_name="correlation_exposure",
                result=GateResult.BLOCK,
                detail="Equity is zero.",
            )

        correlated_notional = 0.0
        for pos in positions:
            pos_symbol = pos.get("symbolName", pos.get("symbol", ""))
            corr = correlation_matrix.get(new_symbol, {}).get(pos_symbol, 0)
            if abs(corr) > 0.7:  # strongly correlated
                pos_notional = abs(pos.get("volume", 0)) / 100.0  # approximate
                correlated_notional += pos_notional * abs(corr)

        exposure_pct = correlated_notional / equity

        if exposure_pct > self.max_correlation_exposure:
            return GateVerdict(
                gate_name="correlation_exposure",
                result=GateResult.BLOCK,
                detail=f"Correlated exposure {exposure_pct:.1%} exceeds {self.max_correlation_exposure:.0%} limit.",
                threshold=self.max_correlation_exposure,
                actual_value=exposure_pct,
            )
        return GateVerdict(
            gate_name="correlation_exposure",
            result=GateResult.PASS,
            detail=f"Correlated exposure {exposure_pct:.1%} within limit.",
            threshold=self.max_correlation_exposure,
            actual_value=exposure_pct,
        )

    # ------------------------------------------------------------------
    # Gate 7: News Blackout
    # ------------------------------------------------------------------

    def gate_news_blackout(self) -> GateVerdict:
        """
        Check if current time is within the blackout window of any major news event.
        In production, fetch from economic calendar API and compare with current UTC time.
        """
        now = datetime.now(timezone.utc)
        for event in self.MAJOR_NEWS_EVENTS:
            event_dt = datetime.fromisoformat(f"{event['date']}T{event['time']}:00+00:00")
            delta_minutes = abs((now - event_dt).total_seconds() / 60)
            if delta_minutes <= self.news_blackout_minutes:
                return GateVerdict(
                    gate_name="news_blackout",
                    result=GateResult.BLOCK,
                    detail=f"Within {self.news_blackout_minutes}min of {event['event']}. No new entries.",
                )
        return GateVerdict(
            gate_name="news_blackout",
            result=GateResult.PASS,
            detail="Outside news blackout window.",
        )

    # ------------------------------------------------------------------
    # Gate 8: Volume Step Validation
    # ------------------------------------------------------------------

    def gate_volume_step(self, volume_cents: int, step_volume_cents: Optional[int] = None) -> GateVerdict:
        step = step_volume_cents or self.volume_step_cents
        if volume_cents % step != 0:
            return GateVerdict(
                gate_name="volume_step",
                result=GateResult.BLOCK,
                detail=f"Volume {volume_cents} not a multiple of step {step}.",
                threshold=float(step),
                actual_value=float(volume_cents),
            )
        return GateVerdict(
            gate_name="volume_step",
            result=GateResult.PASS,
            detail=f"Volume {volume_cents} is valid step multiple.",
        )

    # ------------------------------------------------------------------
    # Run all gates
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        opening_balance: float,
        current_equity: float,
        daily_realised_pnl: float,
        current_position_count: int,
        action_is_open: bool,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        volume_cents: Optional[int] = None,
        margin_level_pct: float = 0,
        positions: Optional[list[dict]] = None,
        new_symbol: str = "",
        correlation_matrix: Optional[dict[str, dict[str, float]]] = None,
    ) -> RiskGateReport:
        """
        Run all applicable risk gates and return an aggregate report.

        Only gates applicable to the action are evaluated:
        - Non-opening actions skip position_count.
        - HOLD/FLATTEN skip TP:SL and per-trade-risk.
        - Volume check only applies when volume is specified.
        """
        verdicts: list[GateVerdict] = []

        # Gate 1: Daily loss
        verdicts.append(self.gate_daily_loss(opening_balance, current_equity, daily_realised_pnl))

        # Gate 2: Position count (only for opening trades)
        if action_is_open:
            verdicts.append(self.gate_position_count(current_position_count, action_is_open))

        # Gate 3: Per-trade risk (only when we have SL and entry)
        if all(v is not None for v in [entry_price, stop_loss, volume_cents]) and action_is_open:
            verdicts.append(self.gate_per_trade_risk(
                current_equity, entry_price, stop_loss, volume_cents,
            ))

        # Gate 4: TP:SL ratio
        if all(v is not None for v in [entry_price, stop_loss, take_profit]) and action_is_open:
            verdicts.append(self.gate_tp_sl_ratio(entry_price, stop_loss, take_profit))

        # Gate 5: Margin level
        if margin_level_pct > 0:
            verdicts.append(self.gate_margin_level(margin_level_pct))

        # Gate 6: Correlation exposure
        if action_is_open and positions and correlation_matrix:
            verdicts.append(self.gate_correlation_exposure(
                current_equity, positions, new_symbol, correlation_matrix,
            ))

        # Gate 7: News blackout
        verdicts.append(self.gate_news_blackout())

        # Gate 8: Volume step
        if volume_cents is not None:
            verdicts.append(self.gate_volume_step(volume_cents))

        # Aggregate
        blocks = [v for v in verdicts if v.result == GateResult.BLOCK]
        warns = [v for v in verdicts if v.result == GateResult.WARN]
        all_passed = len(blocks) == 0

        report = RiskGateReport(
            all_passed=all_passed,
            verdicts=verdicts,
            blocked_by=[v.gate_name for v in blocks],
            warnings=[v.detail for v in warns],
        )

        if blocks:
            logger.warning("Risk gates BLOCKED: %s", [v.detail for v in blocks])
        if warns:
            logger.warning("Risk gates WARNING: %s", [v.detail for v in warns])
        if all_passed:
            logger.info("All risk gates PASSED")

        return report


# ---------------------------------------------------------------------------
# Graceful Rollback Manager
# ---------------------------------------------------------------------------

@dataclass
class RollbackAction:
    """A recorded action that can be rolled back."""
    action_id: str
    action_type: str           # "open_position", "close_position", "amend_sl_tp"
    rollback_fn: str           # description of rollback
    symbol: str
    position_id: Optional[int] = None
    original_sl: Optional[float] = None
    original_tp: Optional[float] = None
    volume_cents: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RollbackLedger:
    """
    Maintains a sequential log of actions that can be replayed in reverse
    to gracefully unwind a sequence of trades if something goes wrong.

    Usage:
      ledger = RollbackLedger()
      ledger.record(RollbackAction(...))
      ...
      if error:
          await ledger.rollback_all(ctrader_client)
    """

    def __init__(self, max_history: int = 50):
        self._ledger: list[RollbackAction] = []
        self.max_history = max_history

    def record(self, action: RollbackAction):
        self._ledger.append(action)
        if len(self._ledger) > self.max_history:
            self._ledger = self._ledger[-self.max_history:]
        logger.info("RollbackLedger: recorded %s on %s", action.action_type, action.symbol)

    async def rollback_last(self, ctrader) -> bool:
        """Rollback the most recent action."""
        if not self._ledger:
            logger.info("RollbackLedger: nothing to rollback")
            return True

        action = self._ledger.pop()
        logger.warning("RollbackLedger: rolling back %s (%s)", action.action_type, action.action_id)

        try:
            if action.action_type == "open_position" and action.position_id:
                await ctrader.close_position(action.position_id)
                logger.info("Rollback: closed position %d", action.position_id)
            elif action.action_type == "amend_sl_tp" and action.position_id:
                await ctrader.amend_position(
                    action.position_id,
                    stop_loss=action.original_sl,
                    take_profit=action.original_tp,
                )
                logger.info("Rollback: restored SL/TP on position %d", action.position_id)
            return True
        except Exception as exc:
            logger.critical("Rollback FAILED for %s: %s", action.action_id, exc)
            return False

    async def rollback_all(self, ctrader) -> int:
        """Rollback all recorded actions in reverse order. Returns count of failed rollbacks."""
        failures = 0
        while self._ledger:
            ok = await self.rollback_last(ctrader)
            if not ok:
                failures += 1
        return failures


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _test_risk_gates():
    engine = RiskGateEngine()

    report = engine.evaluate(
        opening_balance=50000.0,
        current_equity=50320.0,
        daily_realised_pnl=320.0,
        current_position_count=2,
        action_is_open=True,
        entry_price=1.0512,
        stop_loss=1.0480,
        take_profit=1.0580,
        volume_cents=10_000_000,
        margin_level_pct=1198.0,
    )

    print(f"All passed: {report.all_passed}")
    for v in report.verdicts:
        print(f"  {v.gate_name}: {v.result.value} — {v.detail}")
    if report.blocked_by:
        print(f"BLOCKED BY: {report.blocked_by}")
    if report.warnings:
        print(f"WARNINGS: {report.warnings}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_risk_gates()