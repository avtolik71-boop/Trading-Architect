"""
DeepSeek V4-PRO Self-Review Agent Loop — Trading Architect
==========================================================
Производственный контур: сигнал → DeepSeek advisory → self-review → execution gate.
Hardened for RTX 3090/4080 (24GB VRAM) with 4-bit quantized model via vLLM.

Архитектура цикла:
  1. Signal ingestion (LightGBM/PatchTST scores)
  2. State snapshot assembly (positions, margin, volatility regime)
  3. DeepSeek advisory generation (structured JSON output)
  4. Self-review layer (critical reflection + constraint validation)
  5. Revision gate (amend / approve / reject)
  6. Execution bridge (cTrader Open API — через ctrader-remote-mcp)
  7. Post-fill verification loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepseek.agent")

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class SignalSource(str, Enum):
    LIGHTGBM = "lightgbm"
    PATCHTST = "patchtst"
    ENSEMBLE = "ensemble"
    MANUAL = "manual"


class ActionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"
    FLATTEN = "FLATTEN"


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"       # ≥0.75 ensemble score
    MEDIUM = "MEDIUM"   # 0.55–0.74
    LOW = "LOW"         # <0.55 — advisory only, no auto-execution


class AdvisoryDecision(str, Enum):
    APPROVED = "approved"
    REVISED = "revised"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SignalPacket:
    """Normalised signal from ML pipeline."""
    source: SignalSource
    symbol: str
    action: ActionType
    confidence: float          # 0.0–1.0
    horizon: str               # "5m", "15m", "1h", "4h", "1d"
    z_score: float             # deviation from rolling mean (σ)
    features: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MarketSnapshot:
    """Aggregated market state at decision time."""
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    atr_14: float
    adx_14: float
    regime: str                 # "trending", "ranging", "volatile", "quiet"
    volume_ratio: float         # current vs 20-period MA
    support: Optional[float] = None
    resistance: Optional[float] = None


@dataclass
class AccountSnapshot:
    balance: float
    equity: float
    used_margin: float
    margin_level_pct: float
    open_positions: int
    total_exposure_pct: float   # total notional / equity
    daily_pnl: float
    daily_drawdown_pct: float


@dataclass
class AdvisoryOutput:
    """Structured advisory from DeepSeek."""
    raw_response: str
    parsed_action: ActionType
    symbol: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    volume_cents: Optional[int]         # Remote encoding (cents)
    confidence_tier: ConfidenceTier
    rationale_bullets: list[str]
    risk_score: float                   # 0.0–1.0, higher = riskier
    revision_count: int = 0


@dataclass
class SelfReviewVerdict:
    """Output of the self-review layer."""
    decision: AdvisoryDecision
    revised_output: Optional[AdvisoryOutput] = None
    rejection_reason: Optional[str] = None
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class DeepSeekTradingArchitect:
    """
    The self-review agent loop that turns DeepSeek V4-PRO into
    a persistent trading architect.

    Lifecycle per tick:
      collect_state → compose_prompt → deepseek_advisory → self_review → execute / revise / reject
    """

    def __init__(
        self,
        vllm_base_url: str = "http://localhost:8600/v1",
        model_name: str = "deepseek-v4-pro-awq",
        max_revisions: int = 2,
        auto_execute: bool = False,          # MUST be False in production
        risk_gate_enabled: bool = True,
        daily_loss_limit_pct: float = 5.0,
        max_position_count: int = 6,
        max_correlation_exposure: float = 0.70,
        log_dir: str = "./logs",
    ):
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.model_name = model_name
        self.max_revisions = max_revisions
        self.auto_execute = auto_execute
        self.risk_gate_enabled = risk_gate_enabled
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_position_count = max_position_count
        self.max_correlation_exposure = max_correlation_exposure

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._cache: dict[str, Any] = {}
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "DeepSeekTradingArchitect initialised | model=%s | vLLM=%s | max_revisions=%d",
            model_name, vllm_base_url, max_revisions,
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    SYSTEM_PROMPT: str = (
        """You are DeepSeek V4-PRO, a quantitative trading architect operating a mean-reversion strategy on cTrader.

YOUR MANDATE:
1. Analyse the provided signal packet, market snapshot, and account state.
2. Produce a structured trading advisory in the exact JSON format specified below.
3. Think critically — question the signal, check for regime mismatches, verify confluence.
4. Never exceed the risk budget. The account is real money.

TRADING PHILOSOPHY:
- Mean-reversion: fade extended deviations from the rolling VWAP / Bollinger bands.
- Confluence required: at least 2 of {z-score threshold, volume climax, RSI divergence, SR level confluence} must align.
- Regime filter: do NOT fade during ADX > 35 with directional slope (trending — mean-reversion fails here).
- Session-aware: avoid entries 30 min before/after major news (NFP, FOMC, CPI).
- Time-stop: if the position hasn't reached TP within 2× the signal horizon, close at market.

RISK RULES (HARD CONSTRAINTS — DO NOT VIOLATE):
- Per-trade risk: max 1.5% of equity.
- Correlated exposure across all positions: ≤ 70%.
- Maximum 6 concurrent positions.
- Daily loss limit: 5% of opening balance → flatten everything, no new trades.
- Always place SL. Never move SL away from the market.
- TP:SL ratio ≥ 1.5:1 for every trade.
- No martingale. No averaging down. No emotional override.

OUTPUT FORMAT (JSON only, no commentary outside the JSON block):
```json
{
  "action": "BUY|SELL|CLOSE_LONG|CLOSE_SHORT|HOLD|FLATTEN",
  "symbol": "EURUSD",
  "entry_price": null,
  "stop_loss": null,
  "take_profit": null,
  "volume_cents": null,
  "confidence_tier": "HIGH|MEDIUM|LOW",
  "rationale": [
    "Bullet 1: what the signal says",
    "Bullet 2: what the market structure confirms or contradicts",
    "Bullet 3: risk calculus"
  ],
  "risk_score": 0.0,
  "metadata": {
    "z_score_used": 0.0,
    "regime": "ranging",
    "adx_at_decision": 0.0
  }
}
```

If the action is HOLD or FLATTEN, omit entry/stop/target/volume fields (set to null).
If the confidence tier is LOW, the system will treat this as advisory-only and will NOT auto-execute.
"""
    )

    # ------------------------------------------------------------------
    # REVIEWER SYSTEM PROMPT (second pass)
    # ------------------------------------------------------------------

    REVIEWER_PROMPT: str = (
        """You are the CRITICAL REVIEWER. Your ONLY job is to find flaws in the trading advisory below.

REVIEW CHECKLIST — answer EVERY question:
1. REGIME CHECK: Does the advised action match the market regime? (Mean-reversion fails in strong trends.)
2. RISK CHECK: Is per-trade risk ≤ 1.5% equity? Is TP:SL ≥ 1.5:1?
3. CONFLUENCE CHECK: Does the signal have at least 2 confirming factors?
4. CORRELATION CHECK: Would this trade push correlated exposure > 70%?
5. TIMING CHECK: Are we within 30 min of a major news event?
6. POSITION COUNT: ≤ 6 concurrent?
7. DAILY LIMIT: Has the daily loss limit been reached?
8. SIZE CHECK: Is volume appropriate for current ATR and account equity?

If ALL checks pass, respond with: {"verdict": "approved", "checks_passed": [...], "checks_failed": []}
If ANY check fails, respond with: {"verdict": "rejected", "checks_passed": [...], "checks_failed": [...], "rejection_reason": "..."}
If checks fail but the trade structure can be salvaged by adjusting size/SL/TP, respond with: {"verdict": "revised", "revised_output": {...full advisory JSON...}, "checks_passed": [...], "checks_failed": [...]}

OUTPUT ONLY VALID JSON. No commentary outside the JSON block.
"""
    )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def assemble_advisory_prompt(
        self,
        signal: SignalPacket,
        market: MarketSnapshot,
        account: AccountSnapshot,
        open_positions: list[dict],
        recent_pnl: list[float],
    ) -> str:
        return json.dumps({
            "instruction": "Generate a trading advisory based on the following state.",
            "signal": asdict(signal),
            "market_snapshot": asdict(market),
            "account_state": asdict(account),
            "open_positions": open_positions,
            "recent_pnl_curve": recent_pnl[-20:] if recent_pnl else [],
        }, indent=2, default=str)

    def assemble_review_prompt(
        self,
        advisory: AdvisoryOutput,
        market: MarketSnapshot,
        account: AccountSnapshot,
        open_positions: list[dict],
    ) -> str:
        payload = {
            "advisory_under_review": {
                "action": advisory.parsed_action.value,
                "symbol": advisory.symbol,
                "entry_price": advisory.entry_price,
                "stop_loss": advisory.stop_loss,
                "take_profit": advisory.take_profit,
                "volume_cents": advisory.volume_cents,
                "confidence_tier": advisory.confidence_tier.value,
                "rationale": advisory.rationale_bullets,
                "risk_score": advisory.risk_score,
            },
            "market_snapshot": asdict(market),
            "account_state": asdict(account),
            "open_positions": open_positions,
        }
        return json.dumps(payload, indent=2, default=str)

    # ------------------------------------------------------------------
    # vLLM chat completion call
    # ------------------------------------------------------------------

    async def _chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        seed: int = 42,
    ) -> str:
        """Call vLLM OpenAI-compatible endpoint."""
        url = f"{self.vllm_base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "extra_body": {"top_p": 0.9, "repetition_penalty": 1.05},
        }
        start = time.monotonic()
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - start
        content = data["choices"][0]["message"]["content"]
        logger.info("vLLM inference completed in %.2fs | tokens=%d", elapsed,
                     data.get("usage", {}).get("total_tokens", -1))
        return content

    # ------------------------------------------------------------------
    # JSON extraction robust to markdown fences
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from a response that may be wrapped in markdown fences."""
        text = text.strip()
        # Remove ```json ... ``` fences
        if text.startswith("```"):
            lines = text.split("\n")
            # drop first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    # ------------------------------------------------------------------
    # Advisory parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_advisory(raw_json: dict) -> AdvisoryOutput:
        return AdvisoryOutput(
            raw_response=json.dumps(raw_json),
            parsed_action=ActionType(raw_json["action"]),
            symbol=raw_json.get("symbol", ""),
            entry_price=raw_json.get("entry_price"),
            stop_loss=raw_json.get("stop_loss"),
            take_profit=raw_json.get("take_profit"),
            volume_cents=raw_json.get("volume_cents"),
            confidence_tier=ConfidenceTier(raw_json.get("confidence_tier", "MEDIUM")),
            rationale_bullets=raw_json.get("rationale", []),
            risk_score=float(raw_json.get("risk_score", 0.5)),
        )

    # ------------------------------------------------------------------
    # Self-review parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_review(raw_json: dict) -> SelfReviewVerdict:
        decision = AdvisoryDecision(raw_json["verdict"])
        revised = None
        if decision == AdvisoryDecision.REVISED and "revised_output" in raw_json:
            revised = DeepSeekTradingArchitect._parse_advisory(raw_json["revised_output"])
        return SelfReviewVerdict(
            decision=decision,
            revised_output=revised,
            rejection_reason=raw_json.get("rejection_reason"),
            checks_passed=raw_json.get("checks_passed", []),
            checks_failed=raw_json.get("checks_failed", []),
        )

    # ------------------------------------------------------------------
    # MAIN LOOP — one tick
    # ------------------------------------------------------------------

    async def tick(
        self,
        signal: SignalPacket,
        market: MarketSnapshot,
        account: AccountSnapshot,
        open_positions: list[dict],
        recent_pnl: list[float],
    ) -> tuple[Optional[AdvisoryOutput], SelfReviewVerdict, list[dict]]:
        """
        Execute one complete agent tick.
        Returns (final_advisory_or_None, review_verdict, execution_log).
        """
        execution_log: list[dict] = []

        # --- Phase 1: Advisory generation ---
        logger.info("Phase 1: Generating advisory for %s (%s confidence=%.2f)",
                     signal.symbol, signal.action.value, signal.confidence)
        advisory_prompt = self.assemble_advisory_prompt(
            signal, market, account, open_positions, recent_pnl,
        )
        raw_response = await self._chat_completion(
            self.SYSTEM_PROMPT, advisory_prompt, temperature=0.3,
        )
        advisory_json = self._extract_json(raw_response)
        advisory = self._parse_advisory(advisory_json)
        logger.info("Advisory: action=%s confidence=%s risk_score=%.2f",
                     advisory.parsed_action.value,
                     advisory.confidence_tier.value,
                     advisory.risk_score)

        # --- Phase 2: Self-review ---
        logger.info("Phase 2: Self-review ...")
        review_prompt = self.assemble_review_prompt(
            advisory, market, account, open_positions,
        )
        review_raw = await self._chat_completion(
            self.REVIEWER_PROMPT, review_prompt, temperature=0.1,
        )
        review_json = self._extract_json(review_raw)
        verdict = self._parse_review(review_json)

        # --- Phase 3: Revision loop (if revised) ---
        revision_count = 0
        final_advisory = advisory
        while verdict.decision == AdvisoryDecision.REVISED and revision_count < self.max_revisions:
            revision_count += 1
            logger.info("Revision %d/%d ...", revision_count, self.max_revisions)
            revised = verdict.revised_output
            if revised is None:
                break
            revised = AdvisoryOutput(
                raw_response=revised.raw_response,
                parsed_action=revised.parsed_action,
                symbol=revised.symbol,
                entry_price=revised.entry_price,
                stop_loss=revised.stop_loss,
                take_profit=revised.take_profit,
                volume_cents=revised.volume_cents,
                confidence_tier=revised.confidence_tier,
                rationale_bullets=revised.rationale_bullets,
                risk_score=revised.risk_score,
                revision_count=revision_count,
            )
            review_prompt = self.assemble_review_prompt(
                revised, market, account, open_positions,
            )
            review_raw = await self._chat_completion(
                self.REVIEWER_PROMPT, review_prompt, temperature=0.1,
            )
            review_json = self._extract_json(review_raw)
            verdict = self._parse_review(review_json)
            if verdict.decision == AdvisoryDecision.APPROVED:
                final_advisory = revised
                break

        # --- Phase 4: Risk gate (hard constraints) ---
        if verdict.decision == AdvisoryDecision.APPROVED and self.risk_gate_enabled:
            gate_result = self._risk_gate_check(
                final_advisory, account, open_positions, recent_pnl,
            )
            execution_log.append({"phase": "risk_gate", "result": gate_result})
            if not gate_result["passed"]:
                logger.warning("Risk gate blocked: %s", gate_result["reason"])
                verdict = SelfReviewVerdict(
                    decision=AdvisoryDecision.REJECTED,
                    rejection_reason=f"Risk gate: {gate_result['reason']}",
                    checks_failed=[gate_result["reason"]],
                )
                return None, verdict, execution_log

        # --- Phase 5: Execution (if approved & auto-execute enabled) ---
        if verdict.decision == AdvisoryDecision.APPROVED and self.auto_execute:
            exec_result = await self._execute_advisory(final_advisory)
            execution_log.append({"phase": "execution", "result": exec_result})
            logger.info("Executed: %s", exec_result)

        return final_advisory, verdict, execution_log

    # ------------------------------------------------------------------
    # Risk gate (deterministic, no LLM)
    # ------------------------------------------------------------------

    def _risk_gate_check(
        self,
        advisory: AdvisoryOutput,
        account: AccountSnapshot,
        open_positions: list[dict],
        recent_pnl: list[float],
    ) -> dict:
        checks = []

        # 1. Daily loss limit
        daily_pnl_pct = account.daily_drawdown_pct
        if abs(daily_pnl_pct) >= self.daily_loss_limit_pct:
            return {"passed": False, "reason": f"Daily loss limit reached: {daily_pnl_pct:.2f}%", "checks": checks}

        # 2. Max position count
        if len(open_positions) >= self.max_position_count and advisory.parsed_action in (ActionType.BUY, ActionType.SELL):
            return {"passed": False, "reason": f"Max positions ({self.max_position_count}) reached", "checks": checks}

        # 3. Per-trade risk cap (1.5% equity)
        if advisory.stop_loss and advisory.entry_price and advisory.volume_cents and advisory.parsed_action in (ActionType.BUY, ActionType.SELL):
            sl_distance = abs(advisory.entry_price - advisory.stop_loss)
            # approximate notional from cents
            notional = advisory.volume_cents / 100.0  # crude — refine per symbol
            risk_pct = (sl_distance * notional / account.equity) * 100 if account.equity > 0 else float("inf")
            if risk_pct > 1.5:
                return {"passed": False, "reason": f"Per-trade risk {risk_pct:.2f}% exceeds 1.5% cap", "checks": checks}

        # 4. Margin level sanity
        if account.margin_level_pct < 200:
            return {"passed": False, "reason": f"Margin level {account.margin_level_pct:.0f}% below 200% safety threshold", "checks": checks}

        return {"passed": True, "reason": "All gates passed", "checks": ["daily_loss", "position_count", "per_trade_risk", "margin_level"]}

    # ------------------------------------------------------------------
    # Execution stub — delegates to ctrader-remote-mcp via HTTP
    # ------------------------------------------------------------------

    async def _execute_advisory(self, advisory: AdvisoryOutput) -> dict:
        """
        Placeholder: real execution goes through ctrader-remote-mcp.
        In production, this calls create_order / close_position via the MCP tool bindings.
        """
        logger.info("EXECUTION STUB: would execute %s on %s | SL=%s TP=%s Vol=%s",
                     advisory.parsed_action.value, advisory.symbol,
                     advisory.stop_loss, advisory.take_profit,
                     advisory.volume_cents)
        return {
            "status": "simulated",
            "action": advisory.parsed_action.value,
            "symbol": advisory.symbol,
            "volume_cents": advisory.volume_cents,
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        """Probe vLLM server readiness."""
        try:
            resp = await self._http.get(f"{self.vllm_base_url}/models")
            return {"status": "ok", "models": resp.json()}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    async def close(self):
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Standalone integration test
# ---------------------------------------------------------------------------

async def _integration_test():
    """Run a synthetic tick against a local vLLM instance."""
    agent = DeepSeekTradingArchitect(
        vllm_base_url="http://localhost:8600/v1",
        auto_execute=False,
        risk_gate_enabled=True,
    )

    health = await agent.health_check()
    print("Health:", json.dumps(health, indent=2))
    if health["status"] != "ok":
        print("vLLM not reachable — skipping integration test")
        await agent.close()
        return

    signal = SignalPacket(
        source=SignalSource.ENSEMBLE,
        symbol="EURUSD",
        action=ActionType.SELL,
        confidence=0.78,
        horizon="1h",
        z_score=-2.3,
        features={"rsi_14": 28.5, "bb_lower": 1.0480, "volume_spike": 1.8},
    )
    market = MarketSnapshot(
        symbol="EURUSD",
        bid=1.0512, ask=1.0514,
        spread_pips=0.2,
        atr_14=0.0018, adx_14=22.0,
        regime="ranging",
        volume_ratio=1.8,
        support=1.0480, resistance=1.0580,
    )
    account = AccountSnapshot(
        balance=50000.0, equity=50320.0,
        used_margin=4200.0, margin_level_pct=1198.0,
        open_positions=2, total_exposure_pct=35.0,
        daily_pnl=320.0, daily_drawdown_pct=0.0,
    )
    open_positions = [
        {"symbol": "EURUSD", "side": "BUY", "volume": 10000000, "entry": 1.0540, "sl": 1.0500, "tp": 1.0620, "pnl_usd": -280.0},
        {"symbol": "GBPUSD", "side": "SELL", "volume": 5000000, "entry": 1.2430, "sl": 1.2480, "tp": 1.2350, "pnl_usd": 150.0},
    ]

    advisory, verdict, log = await agent.tick(
        signal, market, account, open_positions,
        recent_pnl=[50, -20, 80, -10, 120, -30, 200, -15, 60, 40],
    )

    print("\n=== ADVISORY ===")
    print(advisory)
    print("\n=== VERDICT ===")
    print(f"Decision: {verdict.decision.value}")
    print(f"Passed: {verdict.checks_passed}")
    print(f"Failed: {verdict.checks_failed}")
    print("\n=== EXECUTION LOG ===")
    print(json.dumps(log, indent=2))

    await agent.close()


if __name__ == "__main__":
    asyncio.run(_integration_test())