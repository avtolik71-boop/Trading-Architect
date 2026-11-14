"""Smoke-тест пайплайна: один символ, ограниченный перебор."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import time

from run_pipeline import ResearchPipeline, PipelineConfig

config = PipelineConfig(
    symbols=["GBPUSD"],
    max_candidates_per_indicator=3,
    min_trades=10,
)
p = ResearchPipeline(config)
t0 = time.time()
res = p.run_all("2018-01-01", "2020-12-31")
print(f"\nВсего: {time.time()-t0:.1f}s")
for sym, r in res.items():
    print(f"{sym}: n_trials={r['n_trials']}, valid={len(r['results'])}")
    for row in sorted(r["results"], key=lambda x: -x["return_pct"])[:5]:
        print(f"  {row['indicator']:22s} ret={row['return_pct']:7.2f}% "
              f"n={row['n_trades']:4d} PF={row['profit_factor']:5.2f} "
              f"DD={row['max_dd_pct']:5.2f}% DSR={row['dsr']:.3f}({row['dsr_verdict']})")
