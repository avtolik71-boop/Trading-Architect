"""
Авто-публикация результатов исследований
========================================
После прогона исследования (scan → walk-forward) результаты автоматически:
  1. Добавляются в каталог стратегий (реестр UnifiedRegistry → JSON)
  2. Дописываются в накопительный журнал результатов (results_ledger.csv)
  3. Генерируют самодостаточный HTML-дашборд (research_dashboard.html)
  4. Пишут сводку-мост в RUFP дашборд (ДАШБОРД-СТРАТЕГИИ)

Запуск:
  python auto_publish.py                # вручную
  # или вызывается автоматически из run_walkforward.py / run_pipeline.py

Принцип: ИДЕМПОТЕНТНО. Повторный запуск не дублирует записи —
стратегии сопоставляются по (indicator + params), журнал — по
символу/индикатору/дате прогона.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
CATALOG_DIR = HERE / "catalogs" / "registry_data"

# RUFP-дашборд (внешний проект — источник витрины стратегий)
RUFP_DASH_DIR = Path(r"C:\Users\Autolycus-FX\RUFP PROJECT\ДАШБОРД-СТРАТЕГИИ")


# =============================================================================
# 1. Загрузка последних результатов
# =============================================================================

def load_latest_results() -> dict:
    """Найти последние scan и walkforward отчёты."""
    scan_csvs = sorted(REPORTS.glob("scan_*.csv"))
    wf_csvs = sorted(REPORTS.glob("walkforward_*.csv"))

    if not scan_csvs:
        return {"scan_df": pd.DataFrame(), "wf_df": pd.DataFrame(),
                "scan_path": None, "wf_path": None}

    scan_df = pd.read_csv(scan_csvs[-1]) if scan_csvs else pd.DataFrame()
    wf_df = pd.read_csv(wf_csvs[-1]) if wf_csvs else pd.DataFrame()

    return {
        "scan_df": scan_df,
        "wf_df": wf_df,
        "scan_path": scan_csvs[-1] if scan_csvs else None,
        "wf_path": wf_csvs[-1] if wf_csvs else None,
    }


# =============================================================================
# 2. Обновление каталога стратегий (реестр)
# =============================================================================

def _params_key(indicator: str, params) -> str:
    """Стабильный ключ для сопоставления (indicator + params)."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            params = {"raw": params}
    return f"{indicator}|{json.dumps(params, sort_keys=True, ensure_ascii=False)}"


def update_strategy_catalog(wf_df: pd.DataFrame) -> dict:
    """
    Добавить walk-forward результаты в реестр стратегий (JSON).

    Идемпотентно: сопоставляет по (indicator + params), обновляет
    walk_forward_result / null_model_verdict / status, сохраняет JSON.
    """
    if wf_df.empty:
        return {"updated": 0, "go_candidates": 0}

    # Загрузить существующий реестр (или создать пустой)
    strategies_path = CATALOG_DIR / "strategies_registry.json"
    if strategies_path.exists():
        with open(strategies_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = []

    existing_keys = {_params_key(s["indicator"], s.get("params", {})): s for s in registry}

    updated = 0
    go = 0
    for _, row in wf_df.iterrows():
        key = _params_key(row["indicator"], row["params"])
        wf_result = {
            "pf": row.get("pf_oos"),
            "t_stat": row.get("t_stat_oos"),
            "n_trades": int(row.get("n_trades_oos", 0)),
            "max_dd": row.get("max_dd_oos"),
            "return_pct": row.get("total_return_oos"),
            "mean_return": row.get("mean_return_oos"),
            "windows_profitable": row.get("windows_profitable"),
            "n_windows": row.get("n_windows"),
            "run_ts": datetime.now().isoformat(),
        }
        passed = bool(row.get("passed_gates", False))

        if key in existing_keys:
            s = existing_keys[key]
            s["walk_forward_result"] = wf_result
            s["null_model_verdict"] = "ALPHA" if passed else (
                "BETA_OR_TIMING" if row.get("beat_random") else "NO_EDGE")
            if passed:
                s["status"] = "GO_CANDIDATE"
                go += 1
            else:
                s["status"] = "WALK_FORWARD"
            updated += 1
        else:
            # Новая стратегия из прогона
            params = row.get("params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            registry.append({
                "strategy_id": f"S-AUTO-{len(registry)+1:03d}",
                "idea_id": f"I-AUTO-{row['indicator']}",
                "name": f"{row['indicator']} {row['symbol']} (auto)",
                "indicator": row["indicator"],
                "params": params,
                "filter_name": "none",
                "exit_name": "fixed_sl_tp",
                "risk_profile": "FREE",
                "instruments": [row["symbol"]],
                "horizon": "H1",
                "walk_forward_result": wf_result,
                "null_model_verdict": "ALPHA" if passed else "NO_EDGE",
                "status": "GO_CANDIDATE" if passed else "WALK_FORWARD",
                "created_at": datetime.now().isoformat(),
            })
            updated += 1
            if passed:
                go += 1

    os.makedirs(CATALOG_DIR, exist_ok=True)
    with open(strategies_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, default=str)

    return {"updated": updated, "go_candidates": go, "total": len(registry)}


# =============================================================================
# 3. Накопительный журнал результатов
# =============================================================================

def append_results_ledger(wf_df: pd.DataFrame) -> int:
    """
    Дописать walk-forward результаты в накопительный журнал.
    Возвращает число новых строк.
    """
    if wf_df.empty:
        return 0

    ledger_path = REPORTS / "results_ledger.csv"
    new_rows = wf_df.copy()
    new_rows["run_ts"] = datetime.now().isoformat()

    if ledger_path.exists():
        old = pd.read_csv(ledger_path)
        # Дедупликация по (symbol, indicator, params)
        if not old.empty and "params" in old.columns:
            old_keys = set(old.apply(
                lambda r: _params_key(r["indicator"], r.get("params", "{}")), axis=1))
            new_keys = set(new_rows.apply(
                lambda r: _params_key(r["indicator"], r.get("params", "{}")), axis=1))
            # убираем дубли, оставляем свежие
            mask = [k not in old_keys for k in new_keys]
            new_rows = new_rows[mask]
            combined = pd.concat([old, new_rows], ignore_index=True)
        else:
            combined = new_rows
    else:
        combined = new_rows

    combined.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    return len(new_rows)


# =============================================================================
# 4. HTML-дашборд
# =============================================================================

def _html_escape(s) -> str:
    import html
    return html.escape(str(s))


def build_dashboard_html(scan_df: pd.DataFrame, wf_df: pd.DataFrame) -> str:
    """Сгенерировать самодостаточный HTML-дашборд результатов."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Сводка
    n_candidates = len(scan_df) if not scan_df.empty else 0
    n_wf = len(wf_df) if not wf_df.empty else 0
    n_go = int(wf_df["passed_gates"].sum()) if (not wf_df.empty and "passed_gates" in wf_df.columns) else 0

    # Таблица walk-forward (топ по t)
    wf_rows = ""
    if not wf_df.empty:
        wf_sorted = wf_df.sort_values("t_stat_oos", ascending=False).head(30)
        for _, r in wf_sorted.iterrows():
            badge = "✅ GO" if r.get("passed_gates") else "—"
            wf_rows += (
                f"<tr><td>{_html_escape(r['symbol'])}</td>"
                f"<td>{_html_escape(r['indicator'])}</td>"
                f"<td class='num'>{r.get('t_stat_oos', 0):.2f}</td>"
                f"<td class='num'>{r.get('pf_oos', 0):.2f}</td>"
                f"<td class='num'>{int(r.get('n_trades_oos', 0))}</td>"
                f"<td class='num'>{r.get('mean_return_oos', 0):.4f}%</td>"
                f"<td class='num'>{r.get('max_dd_oos', 0):.1f}%</td>"
                f"<td>{badge}</td></tr>"
            )

    # Таблица scan (топ по DSR)
    scan_rows = ""
    if not scan_df.empty and "dsr" in scan_df.columns:
        scan_sorted = scan_df.sort_values("dsr", ascending=False).head(20)
        for _, r in scan_sorted.iterrows():
            scan_rows += (
                f"<tr><td>{_html_escape(r['symbol'])}</td>"
                f"<td>{_html_escape(r['indicator'])}</td>"
                f"<td class='num'>{r.get('return_pct', 0):.2f}%</td>"
                f"<td class='num'>{r.get('profit_factor', 0):.2f}</td>"
                f"<td class='num'>{r.get('max_dd_pct', 0):.1f}%</td>"
                f"<td class='num'>{r.get('dsr', 0):.3f}</td>"
                f"<td class='verdict {r.get('dsr_verdict', '').lower()}'>{r.get('dsr_verdict', '')}</td></tr>"
            )

    # Честный вывод
    conclusion = "Ни одна стратегия не даёт устойчивого эджа после in-sample → walk-forward → DSR." if n_go <= 1 else \
                 f"Найдено {n_go} GO-кандидатов — требуется ручная проверка DSR на множественное тестирование."

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research Dashboard</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#c9d1d9;--dim:#8b949e;--green:#3fb950;--red:#f85149;--amber:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:20px}}
header{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:16px}}
h1{{font-size:20px;margin:0}}h2{{font-size:15px;margin:20px 0 8px;color:var(--dim)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:8px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}}
.kpi b{{display:block;font-size:22px}}.kpi span{{color:var(--dim);font-size:12px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--dim);font-weight:600;font-size:12px}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.verdict.overfit{{color:var(--red)}}.verdict.significant{{color:var(--green)}}.verdict.marginal{{color:var(--amber)}}
.note{{background:var(--panel);border-left:3px solid var(--amber);padding:10px 14px;border-radius:6px;margin-top:16px;color:var(--txt)}}
</style></head><body>
<header><h1>📊 Research Dashboard</h1><span style="color:var(--dim)">собрано {now}</span></header>
<div class="kpis">
<div class="kpi"><b>{n_candidates}</b><span>кандидатов (scan)</span></div>
<div class="kpi"><b>{n_wf}</b><span>на walk-forward</span></div>
<div class="kpi"><b style="color:var(--green)">{n_go}</b><span>GO-кандидатов</span></div>
</div>
<div class="note"><b>Честный вывод:</b> {conclusion}</div>
<h2>Walk-Forward (OOS) — топ по t-статистике</h2>
<table><thead><tr><th>Символ</th><th>Индикатор</th><th>t</th><th>PF</th><th>n</th><th>mean</th><th>DD</th><th>Вердикт</th></tr></thead>
<tbody>{wf_rows or '<tr><td colspan="8" style="color:var(--dim)">нет данных</td></tr>'}</tbody></table>
<h2>In-sample — топ по DSR</h2>
<table><thead><tr><th>Символ</th><th>Индикатор</th><th>Доход</th><th>PF</th><th>DD</th><th>DSR</th><th>Вердикт</th></tr></thead>
<tbody>{scan_rows or '<tr><td colspan="7" style="color:var(--dim)">нет данных</td></tr>'}</tbody></table>
</body></html>"""

    out = REPORTS / "research_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


# =============================================================================
# 5. Мост в RUFP дашборд
# =============================================================================

def write_rufp_bridge(wf_df: pd.DataFrame) -> str:
    """
    Сводка результатов для RUFP дашборда (markdown).
    Пишется рядом с RUFP ДАШБОРД-СТРАТЕГИИ — видна при следующей сборке.
    """
    if wf_df.empty:
        return ""

    go = wf_df[wf_df["passed_gates"] == True] if "passed_gates" in wf_df.columns else wf_df.iloc[0:0]
    top = wf_df.sort_values("t_stat_oos", ascending=False).head(10)

    lines = [
        "# Результаты Trading Architect (авто-публикация)",
        "",
        f"Собрано: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"## Сводка: {len(wf_df)} на walk-forward, {len(go)} GO-кандидатов",
        "",
        "## GO-кандидаты",
    ]
    if len(go):
        for _, r in go.iterrows():
            lines.append(f"- {r['symbol']} {r['indicator']}: t={r['t_stat_oos']:.2f} PF={r['pf_oos']:.2f} n={int(r['n_trades_oos'])}")
    else:
        lines.append("- нет (все OVERFIT после DSR-поправки)")

    lines += ["", "## Топ-10 по t-статистике OOS", ""]
    for _, r in top.iterrows():
        lines.append(f"- {r['symbol']} {r['indicator']}: t={r['t_stat_oos']:.2f} PF={r['pf_oos']:.2f}")

    out = RUFP_DASH_DIR / "research_results_auto.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


# =============================================================================
# 6. Оркестратор
# =============================================================================

def publish() -> dict:
    """Полный цикл авто-публикации результатов."""
    res = load_latest_results()
    scan_df, wf_df = res["scan_df"], res["wf_df"]

    if wf_df.empty and scan_df.empty:
        print("Нет результатов для публикации (reports/ пуст).")
        return {"status": "no_results"}

    catalog = update_strategy_catalog(wf_df)
    ledger_n = append_results_ledger(wf_df)
    dashboard = build_dashboard_html(scan_df, wf_df)
    rufp = write_rufp_bridge(wf_df)

    report = {
        "status": "published",
        "catalog": catalog,
        "ledger_new_rows": ledger_n,
        "dashboard": dashboard,
        "rufp_bridge": rufp,
    }

    print("=" * 60)
    print("  АВТО-ПУБЛИКАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    print(f"  Каталог стратегий: обновлено {catalog['updated']}, GO={catalog['go_candidates']}")
    print(f"  Журнал результатов: +{ledger_n} строк")
    print(f"  Дашборд: {dashboard}")
    print(f"  RUFP-мост: {rufp or '(нет данных)'}")
    return report


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    publish()
