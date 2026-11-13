"""
DeepSeek Trading Architect — Production Entry Point
===================================================
Запуск полного контура:
  1. vLLM health check
  2. cTrader connection
  3. ML model loading
  4. Signal loop (polling cTrader bars → ML → DeepSeek → execute)
  5. Monitoring + graceful shutdown

Usage:
  python main.py --mode live       # Production trading
  python main.py --mode dry-run    # Signals without execution
  python main.py --mode backtest   # Historical replay
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal as unix_signal
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Load environment
load_dotenv()

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)
logger = logging.getLogger("deepseek.main")

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="deepseek-trader",
    help="DeepSeek V4-PRO Trading Architect — Mean-Reversion Engine",
    add_completion=False,
)


@app.command()
def live(
    vllm_url: str = typer.Option(
        "http://localhost:8600/v1",
        "--vllm-url",
        help="vLLM OpenAI-compatible endpoint",
    ),
    ctrader_host: str = typer.Option(
        "host.docker.internal",
        "--ctrader-host",
        help="cTrader REST proxy host",
    ),
    ctrader_port: int = typer.Option(8443, "--ctrader-port"),
    account_id: int = typer.Option(0, "--account-id"),
    symbols: str = typer.Option(
        "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,NZDUSD",
        "--symbols",
        help="Comma-separated symbols to trade",
    ),
    auto_execute: bool = typer.Option(
        False,
        "--auto-execute/--no-auto-execute",
        help="Enable live execution (DANGER — use with caution)",
    ),
    tick_interval_seconds: int = typer.Option(
        60,
        "--tick-interval",
        help="Seconds between signal generation ticks",
    ),
):
    """
    Run the live trading loop.

    AUTO_EXECUTE IS OFF BY DEFAULT — you must explicitly enable it.
    """
    from core.agent_loop import DeepSeekTradingArchitect
    from bridge.execution_bridge import TradingBridge, CTraderRemoteClient

    symbol_list = [s.strip() for s in symbols.split(",")]

    console.print(Panel.fit(
        f"[bold cyan]DeepSeek V4-PRO Trading Architect[/bold cyan]\n\n"
        f"Mode: [bold red]LIVE[/bold red]\n"
        f"Auto-execute: [bold {'red' if auto_execute else 'green'}]{auto_execute}[/]\n"
        f"Symbols: {', '.join(symbol_list)}\n"
        f"vLLM: {vllm_url}\n"
        f"cTrader: {ctrader_host}:{ctrader_port}\n"
        f"Tick interval: {tick_interval_seconds}s",
        title="🚀 Startup",
    ))

    # --- Validation ---
    if auto_execute:
        console.print("[bold red]⚠️  AUTO-EXECUTE IS ENABLED — REAL MONEY AT RISK ⚠️[/bold red]")
        confirm = typer.confirm("Are you sure you want to auto-execute trades?")
        if not confirm:
            console.print("[green]Aborted by user.[/green]")
            raise typer.Abort()

    async def run_loop():
        agent = DeepSeekTradingArchitect(
            vllm_base_url=vllm_url,
            auto_execute=auto_execute,
            risk_gate_enabled=True,
        )

        ctrader = CTraderRemoteClient(
            proxy_url=f"http://{ctrader_host}:{ctrader_port}",
            account_id=account_id,
            access_token=os.getenv("CTRADER_ACCESS_TOKEN", ""),
        )

        bridge = TradingBridge(
            agent=agent,
            ctrader=ctrader,
            auto_execute=auto_execute,
            post_fill_verify=True,
            rollback_enabled=True,
        )

        # --- Health checks ---
        health = await agent.health_check()
        if health["status"] != "ok":
            console.print(f"[red]vLLM not reachable: {health.get('detail')}[/red]")
            return
        console.print("[green]✓ vLLM healthy[/green]")

        await bridge.warmup()
        console.print("[green]✓ cTrader connection established[/green]")

        # --- Main loop ---
        console.print(f"\n[bold]Starting signal loop every {tick_interval_seconds}s...[/bold]\n")

        tick = 0
        shutdown_flag = False

        # Graceful shutdown handler
        def handle_shutdown(signum, frame):
            nonlocal shutdown_flag
            console.print("\n[yellow]Shutdown signal received — completing current tick...[/yellow]")
            shutdown_flag = True

        unix_signal.signal(unix_signal.SIGINT, handle_shutdown)
        unix_signal.signal(unix_signal.SIGTERM, handle_shutdown)

        try:
            while not shutdown_flag:
                tick += 1
                # TODO: Replace with real signal generation from ML pipeline
                console.print(f"[dim]Tick {tick} — waiting for ML signals...[/dim]")
                await asyncio.sleep(tick_interval_seconds)
        finally:
            await bridge.close()
            console.print("[green]Shutdown complete.[/green]")

    asyncio.run(run_loop())


@app.command()
def dry_run(
    vllm_url: str = typer.Option("http://localhost:8600/v1", "--vllm-url"),
    symbols: str = typer.Option("EURUSD,GBPUSD", "--symbols"),
):
    """
    Run the advisory pipeline WITHOUT connecting to cTrader.
    Generates and prints advisory decisions — safe for testing.
    """
    console.print("[bold yellow]DRY RUN MODE — No trades will be executed[/bold yellow]\n")

    from core.agent_loop import DeepSeekTradingArchitect

    async def run():
        agent = DeepSeekTradingArchitect(vllm_base_url=vllm_url, auto_execute=False)
        health = await agent.health_check()
        console.print(f"vLLM: {health['status']}")

        if health["status"] == "ok":
            # Run the integration test from agent_loop
            from core.agent_loop import _integration_test
            await _integration_test()
        await agent.close()

    asyncio.run(run())


@app.command()
def health():
    """Check vLLM server health."""
    import httpx

    vllm_url = os.getenv("VLLM_URL", "http://localhost:8600/v1")

    async def check():
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            try:
                resp = await client.get(f"{vllm_url}/models")
                data = resp.json()
                console.print("[green]✓ vLLM is healthy[/green]")
                console.print(data)
            except Exception as exc:
                console.print(f"[red]✗ vLLM unreachable: {exc}[/red]")

    asyncio.run(check())


@app.command()
def init():
    """Initialize the project: create directories, check dependencies, print config."""
    console.print(Panel.fit(
        "[bold green]DeepSeek Trader — Initialisation[/bold green]\n\n"
        "1. Ensure vLLM is running: docker compose up -d vllm\n"
        "2. Ensure cTrader REST proxy is running on Windows host\n"
        "3. Set environment variables in .env\n"
        "4. Run: python main.py dry-run (test LLM pipeline)\n"
        "5. Run: python main.py live (when ready for paper trading)\n"
        "6. Run: python main.py live --auto-execute (REAL MONEY)\n\n"
        "[bold red]⚠️  NEVER enable auto-execute without a paper-trading validation period[/bold red]",
        title="Setup Checklist",
    ))

    # Print config summary
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("vLLM URL", os.getenv("VLLM_URL", "http://localhost:8600/v1"))
    table.add_row("cTrader Host", os.getenv("CTRADER_HOST", "host.docker.internal"))
    table.add_row("cTrader Port", os.getenv("CTRADER_PORT", "8443"))
    table.add_row("Daily Loss Limit", f"{os.getenv('DAILY_LOSS_LIMIT_PCT', '5.0')}%")
    table.add_row("Max Positions", os.getenv("MAX_POSITIONS", "6"))
    table.add_row("Per-Trade Risk", f"{os.getenv('PER_TRADE_RISK_PCT', '1.5')}%")
    table.add_row("Min TP:SL", os.getenv("MIN_TP_SL_RATIO", "1.5"))
    table.add_row("Strategy", os.getenv("STRATEGY_STYLE", "mean_reversion"))

    console.print(table)


@app.command()
def backtest(
    symbol: str = typer.Option("EURUSD", "--symbol"),
    horizon: str = typer.Option("1h", "--horizon"),
    start: str = typer.Option("2024-01-01", "--start"),
    end: str = typer.Option("2026-01-01", "--end"),
):
    """
    Run backtest on historical data.
    Requires historical bar data in data/ directory (Parquet format).
    """
    console.print(f"[bold]Backtest: {symbol} {horizon} from {start} to {end}[/bold]")
    # TODO: Implement full backtest loop
    console.print("[yellow]Backtest engine — work in progress[/yellow]")
    console.print("Expected workflow: load bars → generate features → ML predict → simulate fills")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()