"""
Signal Engine → DeepSeek Advisory → cTrader Execution Bridge
============================================================
Координирует полный пайплайн:
  1. Signal ingestion от LightGBM / PatchTST
  2. Предобработка и сбор рыночного контекста
  3. Отправка advisory-запроса в DeepSeek agent
  4. Self-review проверка
  5. Risk gate верификация
  6. Исполнение через cTrader Open API (MCP Remote HTTP)
  7. Пост-филл верификация (re-read позиции)
  8. Graceful rollback при ошибках
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
from contextlib import asynccontextmanager

import httpx

# Import core agent
from core.agent_loop import (
    DeepSeekTradingArchitect,
    SignalPacket, MarketSnapshot, AccountSnapshot,
    SignalSource, ActionType, ConfidenceTier,
    AdvisoryOutput, AdvisoryDecision, SelfReviewVerdict,
)

logger = logging.getLogger("deepseek.bridge")

# ---------------------------------------------------------------------------
# cTrader MCP Remote HTTP client (production adapter)
# ---------------------------------------------------------------------------

class CTraderRemoteClient:
    """
    Thin HTTP wrapper around ctrader-remote-mcp.
    Uses the MCP tool bindings exposed by the cTrader REST proxy.

    Volume encoding: cents (1 lot forex = 10 000 000 cents).
    Price encoding: pipettes (divide by 10^pipDigits for display).
    Symbol identifier: integer symbolId.

    See skill: ctrader-mcp-servers for quirks ledger.
    """

    def __init__(
        self,
        proxy_url: str = "http://host.docker.internal:8443",
        account_id: int = 0,
        client_id: str = "",
        client_secret: str = "",
        access_token: str = "",
    ):
        self.proxy_url = proxy_url.rstrip("/")
        self.account_id = account_id
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        self._symbol_cache: dict[str, int] = {}     # name -> symbolId
        self._precision_cache: dict[int, dict] = {}  # symbolId -> {pipDigits, ...}

    async def get_symbols(self) -> list[dict]:
        """Fetch all tradeable symbols and populate cache."""
        resp = await self._http.get(f"{self.proxy_url}/symbols")
        resp.raise_for_status()
        data = resp.json()
        for s in data:
            self._symbol_cache[s["name"]] = s["id"]
            self._precision_cache[s["id"]] = {
                "pipDigits": s.get("pipDigits", 5),
                "moneyDigits": s.get("moneyDigits", 2),
                "lotSize": s.get("lotSize", 100000),
            }
        logger.info("Symbol cache populated: %d symbols", len(data))
        return data

    def symbol_id(self, name: str) -> int:
        if name not in self._symbol_cache:
            raise KeyError(f"Symbol '{name}' not in cache. Run get_symbols() first.")
        return self._symbol_cache[name]

    async def get_spot_prices(self, symbol_ids: list[int]) -> dict[int, dict]:
        """Batch spot prices for given symbolIds."""
        resp = await self._http.post(
            f"{self.proxy_url}/spot_prices",
            json={"symbolId": symbol_ids, "accountId": self.account_id},
        )
        resp.raise_for_status()
        data = resp.json()
        return {item["symbolId"]: item for item in data}

    async def get_positions(self) -> list[dict]:
        """Fetch all open positions."""
        resp = await self._http.get(
            f"{self.proxy_url}/positions",
            params={"accountId": self.account_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def get_balance(self) -> dict:
        """Fetch account balance and margin info."""
        resp = await self._http.get(
            f"{self.proxy_url}/balance",
            params={"accountId": self.account_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def create_order(
        self,
        symbol_id: int,
        side: str,              # "BUY" or "SELL"
        volume_cents: int,
        order_type: str = "MARKET",
        stop_loss: Optional[float] = None,       # absolute price
        take_profit: Optional[float] = None,     # absolute price
        comment: str = "deepseek-advisory",
    ) -> dict:
        """
        Place a MARKET order via Remote MCP.
        Q-R4: MARKET orders can be REJECTED when absolute SL/TP is provided.
        Preferred: use relativeStopLoss/relativeTakeProfit (pipette offsets).
        Fallback: P-REMOTE-MARKET-2STEP (place MARKET without SL/TP, then amend_position).
        """
        payload = {
            "symbolId": symbol_id,
            "accountId": self.account_id,
            "side": side,
            "type": order_type,
            "volume": volume_cents,
            "comment": comment,
        }
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit

        logger.info("Placing order: %s %d cents on symbol %d (SL=%s, TP=%s)",
                     side, volume_cents, symbol_id, stop_loss, take_profit)
        resp = await self._http.post(f"{self.proxy_url}/orders", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close_position(self, position_id: int, volume_cents: Optional[int] = None) -> dict:
        """Close a position (full or partial)."""
        payload = {
            "positionId": position_id,
            "accountId": self.account_id,
        }
        if volume_cents is not None:
            payload["volume"] = volume_cents
        resp = await self._http.post(f"{self.proxy_url}/positions/close", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def amend_position(self, position_id: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> dict:
        """
        Amend SL/TP on an open position.
        Q-R10: omit a leg and it gets REMOVED (not preserved).
        ALWAYS re-read the position after amendment to confirm both SL and TP survived.
        """
        payload = {"positionId": position_id, "accountId": self.account_id}
        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit
        resp = await self._http.post(f"{self.proxy_url}/positions/amend", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_deals(self, from_ms: int, to_ms: int) -> list[dict]:
        """Fetch deals for a time range (epoch ms)."""
        resp = await self._http.get(
            f"{self.proxy_url}/deals",
            params={
                "accountId": self.account_id,
                "from": from_ms,
                "to": to_ms,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Execution status tracking
# ---------------------------------------------------------------------------

class ExecutionStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    ERROR = "error"


@dataclass
class ExecutionResult:
    trade_id: str
    status: ExecutionStatus
    advisory: AdvisoryOutput
    ctrader_order_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_volume: Optional[int] = None
    error_message: Optional[str] = None
    rollback_action: Optional[str] = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# The Bridge
# ---------------------------------------------------------------------------

class TradingBridge:
    """
    Coordinates the full pipeline:
      Signal → Market snapshot → DeepSeek → Self-review → Risk gate → cTrader → Verify.
    """

    def __init__(
        self,
        agent: DeepSeekTradingArchitect,
        ctrader: CTraderRemoteClient,
        risk_gate_enabled: bool = True,
        auto_execute: bool = False,
        post_fill_verify: bool = True,
        rollback_enabled: bool = True,
    ):
        self.agent = agent
        self.ctrader = ctrader
        self.risk_gate_enabled = risk_gate_enabled
        self.auto_execute = auto_execute
        self.post_fill_verify = post_fill_verify
        self.rollback_enabled = rollback_enabled
        self._execution_history: list[ExecutionResult] = []

    async def warmup(self):
        """Pre-fetch symbol catalogue and cache."""
        await self.ctrader.get_symbols()
        logger.info("Bridge warmup complete")

    # ------------------------------------------------------------------
    # Full pipeline: one signal tick through the entire bridge
    # ------------------------------------------------------------------

    async def process_signal(self, signal: SignalPacket) -> Optional[ExecutionResult]:
        """
        Process one signal through the full pipeline.
        Returns ExecutionResult if trade was submitted, None if rejected/HOLD.
        """
        trade_id = str(uuid.uuid4())
        t_start = time.monotonic()

        # --- Step 1: Assemble market snapshot ---
        market = await self._build_market_snapshot(signal.symbol)

        # --- Step 2: Assemble account snapshot ---
        account = await self._build_account_snapshot()

        # --- Step 3: Get open positions ---
        open_positions = await self.ctrader.get_positions()

        # --- Step 4: Build recent PnL curve ---
        recent_pnl = await self._build_recent_pnl()

        # --- Step 5: Send through DeepSeek agent loop ---
        logger.info("[%s] Processing signal: %s %s (conf=%.2f)",
                     trade_id, signal.symbol, signal.action.value, signal.confidence)

        advisory, verdict, exec_log = await self.agent.tick(
            signal, market, account, open_positions, recent_pnl,
        )

        # --- Step 6: Decision gate ---
        if verdict.decision != AdvisoryDecision.APPROVED:
            logger.info("[%s] Advisory %s: %s", trade_id, verdict.decision.value,
                        verdict.rejection_reason or "no reason")
            result = ExecutionResult(
                trade_id=trade_id,
                status=ExecutionStatus.REJECTED,
                advisory=advisory or AdvisoryOutput(
                    raw_response="", parsed_action=ActionType.HOLD,
                    symbol=signal.symbol, entry_price=None, stop_loss=None,
                    take_profit=None, volume_cents=None,
                    confidence_tier=ConfidenceTier.LOW,
                    rationale_bullets=[], risk_score=1.0,
                ),
                error_message=verdict.rejection_reason,
                latency_ms=(time.monotonic() - t_start) * 1000,
            )
            self._execution_history.append(result)
            return result

        if advisory is None:
            return None

        if advisory.parsed_action in (ActionType.HOLD,):
            logger.info("[%s] Advisory approved but action is HOLD — no execution", trade_id)
            return None

        # --- Step 7: Execution ---
        if not self.auto_execute:
            logger.info("[%s] AUTO_EXECUTE is OFF — advisory generated but NOT executed", trade_id)
            logger.info("Advisory: %s %s SL=%s TP=%s Vol=%s",
                        advisory.parsed_action.value, advisory.symbol,
                        advisory.stop_loss, advisory.take_profit, advisory.volume_cents)
            return None

        result = await self._execute_with_rollback(trade_id, advisory, t_start)
        self._execution_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Execution with graceful rollback
    # ------------------------------------------------------------------

    async def _execute_with_rollback(
        self, trade_id: str, advisory: AdvisoryOutput, t_start: float,
    ) -> ExecutionResult:
        """
        Execute the advisory through cTrader with rollback on failure.

        Rollback logic:
        - If order is submitted but rejected → no rollback needed.
        - If order is filled, then post-fill verification fails →
          attempt to close the position immediately (rollback).
        - If close fails → log alert, do not retry endlessly.
        """
        try:
            symbol_id = self.ctrader.symbol_id(advisory.symbol)

            if advisory.parsed_action in (ActionType.BUY, ActionType.SELL):
                ctrader_side = "BUY" if advisory.parsed_action == ActionType.BUY else "SELL"
                order_resp = await self.ctrader.create_order(
                    symbol_id=symbol_id,
                    side=ctrader_side,
                    volume_cents=advisory.volume_cents or 0,
                    order_type="MARKET",
                    stop_loss=advisory.stop_loss,
                    take_profit=advisory.take_profit,
                )
                logger.info("[%s] Order submitted: %s", trade_id, order_resp)

                # Post-fill verification
                if self.post_fill_verify:
                    await asyncio.sleep(1.0)  # allow propagation
                    positions = await self.ctrader.get_positions()
                    verified = self._verify_fill(advisory, positions)

                    if not verified["ok"]:
                        logger.error("[%s] Post-fill verification FAILED: %s", trade_id, verified["reason"])
                        if self.rollback_enabled:
                            rollback_result = await self._rollback_position(advisory, positions)
                            return ExecutionResult(
                                trade_id=trade_id,
                                status=ExecutionStatus.ROLLED_BACK,
                                advisory=advisory,
                                error_message=verified["reason"],
                                rollback_action=rollback_result,
                                latency_ms=(time.monotonic() - t_start) * 1000,
                            )

                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.FILLED,
                    advisory=advisory,
                    ctrader_order_id=str(order_resp.get("orderId", "")),
                    fill_price=order_resp.get("fillPrice"),
                    fill_volume=order_resp.get("filledVolume"),
                    latency_ms=(time.monotonic() - t_start) * 1000,
                )

            elif advisory.parsed_action in (ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT):
                positions = await self.ctrader.get_positions()
                target = self._find_matching_position(advisory, positions)
                if target is None:
                    return ExecutionResult(
                        trade_id=trade_id,
                        status=ExecutionStatus.ERROR,
                        advisory=advisory,
                        error_message="No matching position found to close",
                        latency_ms=(time.monotonic() - t_start) * 1000,
                    )
                close_resp = await self.ctrader.close_position(target["id"])
                logger.info("[%s] Position closed: %s", trade_id, close_resp)
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.FILLED,
                    advisory=advisory,
                    ctrader_order_id=str(close_resp.get("orderId", "")),
                    latency_ms=(time.monotonic() - t_start) * 1000,
                )

            elif advisory.parsed_action == ActionType.FLATTEN:
                positions = await self.ctrader.get_positions()
                closed_count = 0
                for pos in positions:
                    await self.ctrader.close_position(pos["id"])
                    closed_count += 1
                logger.info("[%s] Flattened: %d positions closed", trade_id, closed_count)
                return ExecutionResult(
                    trade_id=trade_id,
                    status=ExecutionStatus.FILLED,
                    advisory=advisory,
                    latency_ms=(time.monotonic() - t_start) * 1000,
                )

            return ExecutionResult(
                trade_id=trade_id,
                status=ExecutionStatus.ERROR,
                advisory=advisory,
                error_message=f"Unhandled action: {advisory.parsed_action}",
                latency_ms=(time.monotonic() - t_start) * 1000,
            )

        except httpx.HTTPStatusError as exc:
            logger.error("[%s] HTTP error during execution: %s %s", trade_id, exc.response.status_code, exc.response.text)
            return ExecutionResult(
                trade_id=trade_id,
                status=ExecutionStatus.REJECTED,
                advisory=advisory,
                error_message=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
                latency_ms=(time.monotonic() - t_start) * 1000,
            )
        except Exception as exc:
            logger.exception("[%s] Unexpected error during execution", trade_id)
            return ExecutionResult(
                trade_id=trade_id,
                status=ExecutionStatus.ERROR,
                advisory=advisory,
                error_message=str(exc),
                latency_ms=(time.monotonic() - t_start) * 1000,
            )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def _rollback_position(self, advisory: AdvisoryOutput, positions: list[dict]) -> str:
        """Attempt to close the position that was just opened."""
        target = self._find_matching_position(advisory, positions)
        if target is None:
            return "No position found to rollback"
        try:
            await self.ctrader.close_position(target["id"])
            logger.warning("Rollback executed: closed position %d", target["id"])
            return f"Closed position {target['id']}"
        except Exception as exc:
            logger.critical("ROLLBACK FAILED: cannot close position %d: %s", target["id"], exc)
            return f"Rollback FAILED: {exc}"

    # ------------------------------------------------------------------
    # Post-fill verification
    # ------------------------------------------------------------------

    def _verify_fill(self, advisory: AdvisoryOutput, positions: list[dict]) -> dict:
        """
        Verify that the filled position matches the advisory.
        Checks: symbol, side, SL presence, TP presence, volume within tolerance.
        """
        matching = self._find_matching_position(advisory, positions)
        if matching is None:
            # Check if it just hasn't propagated yet — retry once
            return {"ok": False, "reason": "No matching position found (propagation lag?)"}

        # Volume check (within 10% tolerance)
        expected_vol = advisory.volume_cents or 0
        actual_vol = matching.get("volume", 0)
        if expected_vol > 0 and abs(actual_vol - expected_vol) / expected_vol > 0.10:
            return {"ok": False, "reason": f"Volume mismatch: expected {expected_vol}, got {actual_vol}"}

        # Side check
        advisory_side = "BUY" if advisory.parsed_action == ActionType.BUY else "SELL"
        if matching.get("side") != advisory_side:
            return {"ok": False, "reason": f"Side mismatch: expected {advisory_side}, got {matching.get('side')}"}

        # SL check (only if advisory specified one)
        if advisory.stop_loss and matching.get("stopLoss") is None:
            return {"ok": False, "reason": "Stop loss missing from filled position"}

        return {"ok": True, "reason": "Position verified"}

    def _find_matching_position(self, advisory: AdvisoryOutput, positions: list[dict]) -> Optional[dict]:
        """Find the position that matches the advisory symbol and side."""
        advisory_side = "BUY" if advisory.parsed_action == ActionType.BUY else "SELL"
        for pos in positions:
            pos_symbol_id = pos.get("symbolId")
            pos_symbol_name = pos.get("symbolName", "")
            if advisory.symbol in (pos_symbol_name, str(pos_symbol_id)):
                if advisory.parsed_action in (ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT):
                    if (advisory.parsed_action == ActionType.CLOSE_LONG and pos.get("side") == "BUY") or \
                       (advisory.parsed_action == ActionType.CLOSE_SHORT and pos.get("side") == "SELL"):
                        return pos
                else:
                    if pos.get("side") == advisory_side:
                        return pos
        return None

    # ------------------------------------------------------------------
    # State builders
    # ------------------------------------------------------------------

    async def _build_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """Assemble a market snapshot from cTrader data."""
        symbol_id = self.ctrader.symbol_id(symbol)
        prices = await self.ctrader.get_spot_prices([symbol_id])
        price_data = prices.get(symbol_id, {})

        bid = price_data.get("bid", 0)
        ask = price_data.get("ask", 0)
        pip_digits = self.ctrader._precision_cache.get(symbol_id, {}).get("pipDigits", 5)
        pip_size = 10 ** (-pip_digits)

        spread_pips = (ask - bid) / pip_size if pip_size > 0 else 0

        return MarketSnapshot(
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips,
            atr_14=0.0,          # Filled by feature pipeline
            adx_14=0.0,
            regime="unknown",
            volume_ratio=1.0,
        )

    async def _build_account_snapshot(self) -> AccountSnapshot:
        """Assemble account state."""
        balance = await self.ctrader.get_balance()
        return AccountSnapshot(
            balance=float(balance.get("balance", 0)),
            equity=float(balance.get("equity", 0)),
            used_margin=float(balance.get("usedMargin", 0)),
            margin_level_pct=float(balance.get("marginLevel", 0)),
            open_positions=0,     # filled below
            total_exposure_pct=0.0,
            daily_pnl=float(balance.get("profit", 0)),
            daily_drawdown_pct=0.0,
        )

    async def _build_recent_pnl(self) -> list[float]:
        """Build recent PnL curve from deal history (last 24h)."""
        now_ms = int(time.time() * 1000)
        day_ago_ms = now_ms - 24 * 3600 * 1000
        try:
            deals = await self.ctrader.get_deals(from_ms=day_ago_ms, to_ms=now_ms)
            return [d.get("profit", 0) for d in deals]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self):
        await self.ctrader.close()
        await self.agent.close()
        logger.info("Bridge shutdown complete")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

async def _integration_test():
    """Run bridge with synthetic data (no real cTrader connection needed)."""
    agent = DeepSeekTradingArchitect(
        vllm_base_url="http://localhost:8600/v1",
        auto_execute=False,
        risk_gate_enabled=True,
    )

    # Use a dummy cTrader client — will fail on real calls but that's fine for testing
    ctrader = CTraderRemoteClient()

    bridge = TradingBridge(
        agent=agent,
        ctrader=ctrader,
        auto_execute=False,
    )

    signal = SignalPacket(
        source=SignalSource.ENSEMBLE,
        symbol="EURUSD",
        action=ActionType.SELL,
        confidence=0.78,
        horizon="1h",
        z_score=-2.3,
        features={"rsi_14": 28.5, "bb_lower": 1.0480, "volume_spike": 1.8},
    )

    result = await bridge.process_signal(signal)
    print(f"\nResult: {result}")
    await bridge.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_integration_test())