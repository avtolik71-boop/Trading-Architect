"""Inline tests — no pytest dependency."""
import sys
sys.path.insert(0, '.')

from risk.risk_gates import RiskGateEngine, RollbackLedger, RollbackAction, GateResult
from core.agent_loop import DeepSeekTradingArchitect, ActionType, ConfidenceTier, AdvisoryDecision

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f'  OK: {name}')
    else:
        failed += 1; print(f'  FAIL: {name}')

# === Risk Gates ===
engine = RiskGateEngine()

def gate_check(name, expected_pass, **kwargs):
    v = getattr(engine, name)(**kwargs)
    check(name, (v.result == GateResult.PASS) == expected_pass)

gate_check('gate_daily_loss', True, opening_balance=50000, current_equity=50320, daily_realised_pnl=320)
gate_check('gate_daily_loss', False, opening_balance=50000, current_equity=47200, daily_realised_pnl=-2800)
v = engine.gate_daily_loss(50000, 48100, -1900)
check('daily_loss_warn', v.result == GateResult.WARN)

gate_check('gate_position_count', False, current_count=6, action_is_open=True)
gate_check('gate_position_count', True, current_count=3, action_is_open=True)
gate_check('gate_position_count', True, current_count=6, action_is_open=False)

# 5 lots (50M cents), SL=100 pips → risk = 100*5*10 = $5000 = 10% of 50k → BLOCK
gate_check('gate_per_trade_risk', False, equity=50000, entry_price=1.0500, stop_loss=1.0400, volume_cents=50_000_000, pip_size=0.0001)
# 0.1 lot (1M cents), SL=20 pips → risk = 20*0.1*10 = $20 = 0.04% of 50k → PASS
gate_check('gate_per_trade_risk', True, equity=50000, entry_price=1.0500, stop_loss=1.0480, volume_cents=1_000_000, pip_size=0.0001)

gate_check('gate_tp_sl_ratio', False, entry_price=1.05, stop_loss=1.04, take_profit=1.055)
gate_check('gate_tp_sl_ratio', True, entry_price=1.05, stop_loss=1.04, take_profit=1.07)

gate_check('gate_margin_level', False, margin_level_pct=150)
gate_check('gate_margin_level', True, margin_level_pct=350)  # PASS: 350 > 300 (1.5× thresh)
gate_check('gate_margin_level', True, margin_level_pct=500)
v = engine.gate_margin_level(250)
check('margin_level_warn', v.result == GateResult.WARN)  # 250 < 300 (1.5× threshold) = WARN

gate_check('gate_volume_step', False, volume_cents=1500, step_volume_cents=1000)
gate_check('gate_volume_step', True, volume_cents=2000, step_volume_cents=1000)

report = engine.evaluate(
    opening_balance=50000, current_equity=50320, daily_realised_pnl=320,
    current_position_count=2, action_is_open=True,
    entry_price=1.0512, stop_loss=1.0480, take_profit=1.0580,
    volume_cents=1_000_000, margin_level_pct=1200,
)
check('full_evaluate_pass', report.all_passed)

report2 = engine.evaluate(
    opening_balance=50000, current_equity=47200, daily_realised_pnl=-2800,
    current_position_count=2, action_is_open=True,
    entry_price=1.0512, stop_loss=1.0480, take_profit=1.0580,
    volume_cents=1_000_000, margin_level_pct=1200,
)
check('full_evaluate_block', not report2.all_passed)
check('blocked_by_daily_loss', 'daily_loss_limit' in report2.blocked_by)

# === Agent Loop ===
agent = DeepSeekTradingArchitect(vllm_base_url='http://localhost:8600/v1', auto_execute=False)

r1 = agent._extract_json('{"action": "BUY", "symbol": "EURUSD"}')
check('json_plain', r1['action'] == 'BUY')

r2 = agent._extract_json('```json\n{"action": "SELL", "symbol": "GBPUSD"}\n```')
check('json_fenced', r2['action'] == 'SELL')

adv = agent._parse_advisory({
    'action': 'SELL', 'symbol': 'EURUSD', 'entry_price': 1.0512,
    'stop_loss': 1.0530, 'take_profit': 1.0450, 'volume_cents': 1000000,
    'confidence_tier': 'HIGH', 'rationale': ['z-score -2.3'], 'risk_score': 0.2,
})
check('parse_action', adv.parsed_action == ActionType.SELL)
check('parse_tier', adv.confidence_tier == ConfidenceTier.HIGH)

v1 = agent._parse_review({'verdict': 'approved', 'checks_passed': ['a'], 'checks_failed': []})
check('review_approved', v1.decision == AdvisoryDecision.APPROVED)

v2 = agent._parse_review({'verdict': 'rejected', 'checks_passed': ['a'], 'checks_failed': ['b'], 'rejection_reason': 'risky'})
check('review_rejected', v2.decision == AdvisoryDecision.REJECTED)
check('review_reason', v2.rejection_reason == 'risky')

# === Rollback Ledger ===
ledger = RollbackLedger(max_history=3)
for i in range(5):
    ledger.record(RollbackAction(action_id=str(i), action_type='open_position', rollback_fn='close', symbol='EURUSD', position_id=i))
check('rollback_max', len(ledger._ledger) == 3)

# === Summary ===
print(f'\n{"="*40}')
print(f'RESULTS: {passed}/{passed+failed} passed')
if failed == 0:
    print('ALL TESTS PASSED')
else:
    print(f'{failed} TESTS FAILED')
print(f'{"="*40}')