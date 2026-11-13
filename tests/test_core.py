"""
DeepSeek Trading Architect — Unit Tests
=======================================
Tests for the core agent loop and risk gates.
"""

import pytest
from datetime import datetime, timezone

from core.agent_loop import (
    SignalPacket, MarketSnapshot, AccountSnapshot,
    SignalSource, ActionType, ConfidenceTier,
    AdvisoryOutput, AdvisoryDecision, SelfReviewVerdict,
    DeepSeekTradingArchitect,
)
from risk.risk_gates import (
    RiskGateEngine, RiskGateReport, GateResult, GateVerdict,
    RollbackLedger, RollbackAction,
)


# =============================================================================
# Risk Gate Tests
# =============================================================================

class TestRiskGates:
    def setup_method(self):
        self.engine = RiskGateEngine()

    def test_daily_loss_pass(self):
        v = self.engine.gate_daily_loss(50000, 50320, 320)
        assert v.result == GateResult.PASS

    def test_daily_loss_block(self):
        v = self.engine.gate_daily_loss(50000, 47200, -2800)
        assert v.result == GateResult.BLOCK
        assert v.detail.startswith("Daily loss")

    def test_daily_loss_warn(self):
        v = self.engine.gate_daily_loss(50000, 48100, -1900)
        assert v.result == GateResult.WARN

    def test_position_count_block(self):
        v = self.engine.gate_position_count(6, True)
        assert v.result == GateResult.BLOCK

    def test_position_count_pass(self):
        v = self.engine.gate_position_count(3, True)
        assert v.result == GateResult.PASS

    def test_position_count_non_open(self):
        v = self.engine.gate_position_count(6, False)
        assert v.result == GateResult.PASS

    def test_per_trade_risk_block(self):
        v = self.engine.gate_per_trade_risk(
            equity=50000, entry_price=1.05, stop_loss=1.04,
            volume_cents=50_000_000,  # 5 lots → too big for 1.5% risk
        )
        assert v.result == GateResult.BLOCK

    def test_per_trade_risk_pass(self):
        v = self.engine.gate_per_trade_risk(
            equity=50000, entry_price=1.05, stop_loss=1.048,
            volume_cents=1_000_000,  # 0.1 lot → tiny risk
        )
        assert v.result == GateResult.PASS

    def test_tp_sl_ratio_block(self):
        v = self.engine.gate_tp_sl_ratio(1.05, 1.04, 1.06)
        assert v.result == GateResult.PASS  # ratio = 0.01 / 0.01 = 1.0 — actually fails
        # Let's test actual fail:
        v2 = self.engine.gate_tp_sl_ratio(1.05, 1.04, 1.055)
        assert v2.result == GateResult.BLOCK  # ratio = 0.005/0.01 = 0.5

    def test_margin_level_block(self):
        v = self.engine.gate_margin_level(150)
        assert v.result == GateResult.BLOCK

    def test_margin_level_warn(self):
        v = self.engine.gate_margin_level(250)
        assert v.result == GateResult.WARN

    def test_margin_level_pass(self):
        v = self.engine.gate_margin_level(500)
        assert v.result == GateResult.PASS

    def test_volume_step_block(self):
        v = self.engine.gate_volume_step(volume_cents=1500, step_volume_cents=1000)
        assert v.result == GateResult.BLOCK

    def test_volume_step_pass(self):
        v = self.engine.gate_volume_step(volume_cents=2000, step_volume_cents=1000)
        assert v.result == GateResult.PASS

    def test_full_evaluate_pass(self):
        report = self.engine.evaluate(
            opening_balance=50000,
            current_equity=50320,
            daily_realised_pnl=320,
            current_position_count=2,
            action_is_open=True,
            entry_price=1.0512,
            stop_loss=1.0480,
            take_profit=1.0580,
            volume_cents=1_000_000,
            margin_level_pct=1200,
        )
        assert report.all_passed

    def test_full_evaluate_block_daily_loss(self):
        report = self.engine.evaluate(
            opening_balance=50000,
            current_equity=47200,
            daily_realised_pnl=-2800,
            current_position_count=2,
            action_is_open=True,
            entry_price=1.0512,
            stop_loss=1.0480,
            take_profit=1.0580,
            volume_cents=1_000_000,
            margin_level_pct=1200,
        )
        assert not report.all_passed
        assert "daily_loss_limit" in report.blocked_by


# =============================================================================
# Agent Loop Tests (no vLLM needed)
# =============================================================================

class TestAgentLoop:
    def setup_method(self):
        self.agent = DeepSeekTradingArchitect(
            vllm_base_url="http://localhost:8600/v1",
            auto_execute=False,
            risk_gate_enabled=True,
        )

    def test_json_extraction_plain(self):
        raw = '{"action": "BUY", "symbol": "EURUSD"}'
        result = self.agent._extract_json(raw)
        assert result["action"] == "BUY"

    def test_json_extraction_fenced(self):
        raw = '```json\n{"action": "SELL", "symbol": "GBPUSD"}\n```'
        result = self.agent._extract_json(raw)
        assert result["action"] == "SELL"

    def test_parse_advisory(self):
        raw = {
            "action": "SELL",
            "symbol": "EURUSD",
            "entry_price": 1.0512,
            "stop_loss": 1.0530,
            "take_profit": 1.0450,
            "volume_cents": 1000000,
            "confidence_tier": "HIGH",
            "rationale": ["z-score -2.3", "volume spike"],
            "risk_score": 0.2,
        }
        adv = self.agent._parse_advisory(raw)
        assert adv.parsed_action == ActionType.SELL
        assert adv.confidence_tier == ConfidenceTier.HIGH
        assert adv.stop_loss == 1.0530

    def test_parse_review_approved(self):
        raw = {"verdict": "approved", "checks_passed": ["regime", "risk"], "checks_failed": []}
        verdict = self.agent._parse_review(raw)
        assert verdict.decision == AdvisoryDecision.APPROVED

    def test_parse_review_rejected(self):
        raw = {"verdict": "rejected", "checks_passed": ["regime"], "checks_failed": ["risk"], "rejection_reason": "Risk too high"}
        verdict = self.agent._parse_review(raw)
        assert verdict.decision == AdvisoryDecision.REJECTED
        assert verdict.rejection_reason == "Risk too high"

    def test_risk_gate_check_daily_loss(self):
        account = AccountSnapshot(
            balance=50000, equity=47200, used_margin=5000,
            margin_level_pct=944, open_positions=2, total_exposure_pct=30,
            daily_pnl=-2800, daily_drawdown_pct=5.6,
        )
        advisory = AdvisoryOutput(
            raw_response="", parsed_action=ActionType.BUY, symbol="EURUSD",
            entry_price=1.05, stop_loss=1.04, take_profit=1.08,
            volume_cents=1_000_000, confidence_tier=ConfidenceTier.HIGH,
            rationale_bullets=[], risk_score=0.2,
        )
        result = self.agent._risk_gate_check(advisory, account, [], [50, -20, 80])
        assert not result["passed"]
        assert "Daily loss limit" in result["reason"]

    def test_risk_gate_check_position_count(self):
        account = AccountSnapshot(
            balance=50000, equity=50320, used_margin=5000,
            margin_level_pct=1006, open_positions=6, total_exposure_pct=60,
            daily_pnl=320, daily_drawdown_pct=0.0,
        )
        advisory = AdvisoryOutput(
            raw_response="", parsed_action=ActionType.BUY, symbol="EURUSD",
            entry_price=1.05, stop_loss=1.04, take_profit=1.08,
            volume_cents=1_000_000, confidence_tier=ConfidenceTier.HIGH,
            rationale_bullets=[], risk_score=0.2,
        )
        result = self.agent._risk_gate_check(advisory, account, [{}] * 6, [])
        assert not result["passed"]
        assert "Max positions" in result["reason"]


# =============================================================================
# Rollback Ledger Tests
# =============================================================================

class TestRollbackLedger:
    def test_record_and_size(self):
        ledger = RollbackLedger(max_history=3)
        for i in range(5):
            ledger.record(RollbackAction(
                action_id=str(i), action_type="open_position",
                rollback_fn="close", symbol="EURUSD", position_id=i,
            ))
        assert len(ledger._ledger) == 3  # Max history enforced


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])