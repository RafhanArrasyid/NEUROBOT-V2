import argparse
import asyncio
import os
import sys
from typing import Optional

import pandas as pd

from brain import NeuroBrain
from config import Config
from loader import ExchangeLoader
from smc import SMCAnalyzer


def _parse_args():
    parser = argparse.ArgumentParser(description="Visual backtest for SMC + AI on Binance data.")
    parser.add_argument("--symbol", default="DOGE/USDT", help="Symbol, e.g. DOGE/USDT")
    parser.add_argument("--timeframe", default=Config.TF_ENTRY, help="Timeframe, e.g. 15m")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles")
    parser.add_argument("--output", default="", help="Output PNG path")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI validator")
    parser.add_argument("--show-setups", action="store_true", help="Plot SMC setups even if skipped by filters")
    parser.add_argument("--preview", action="store_true", help="Allow pending setups without zone touch (visual only)")
    parser.add_argument("--relax-filters", action="store_true", help="Ignore volume/session filters for preview")
    parser.add_argument("--max-setups", type=int, default=5, help="Max setups to annotate on chart")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _resolve_output(symbol: str, timeframe: str, output: str) -> str:
    if output:
        return output
    name = f"{_safe_symbol(symbol)}_{timeframe}.png"
    return os.path.join("plots", name)


def _ensure_output_dir(output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _check_exit(trade: dict, candle: pd.Series) -> tuple[Optional[str], Optional[float]]:
    if trade["side"] == "LONG":
        hit_sl = candle["low"] <= trade["sl"]
        hit_tp = candle["high"] >= trade["tp"]
        if hit_sl and hit_tp:
            return "SL", trade["sl"]
        if hit_sl:
            return "SL", trade["sl"]
        if hit_tp:
            return "TP", trade["tp"]
    else:
        hit_sl = candle["high"] >= trade["sl"]
        hit_tp = candle["low"] <= trade["tp"]
        if hit_sl and hit_tp:
            return "SL", trade["sl"]
        if hit_sl:
            return "SL", trade["sl"]
        if hit_tp:
            return "TP", trade["tp"]
    return None, None


def _format_level(value: Optional[float]) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return "n/a"


def _pick_arrow_offset(y: float, y_min: float, y_max: float, base_offset: float, prefer_up: bool) -> float:
    if prefer_up:
        if y + base_offset <= y_max:
            return base_offset
        if y - base_offset >= y_min:
            return -base_offset
        return base_offset
    if y - base_offset >= y_min:
        return -base_offset
    if y + base_offset <= y_max:
        return base_offset
    return -base_offset


def _annotate_arrow(
    ax,
    x: float,
    y: float,
    label: str,
    color: str,
    y_min: float,
    y_max: float,
    base_offset: float,
    prefer_up: bool,
) -> None:
    offset = _pick_arrow_offset(y, y_min, y_max, base_offset, prefer_up)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x, y + offset),
        textcoords="data",
        ha="center",
        va="center",
        fontsize=7,
        color="black",
        arrowprops=dict(arrowstyle="->", color=color, linewidth=0.9),
    )


def _add_setup(
    setups: list[dict],
    setup: dict,
    side: str,
    entry_time: pd.Timestamp,
    max_setups: int,
    status: str,
) -> None:
    record = {
        "side": side,
        "entry_time": entry_time,
        "entry": float(setup["entry"]),
        "sl": float(setup["sl"]),
        "tp": float(setup["tp"]),
        "entry_src": setup.get("entry_src"),
        "status": status,
    }
    setups.append(record)
    if len(setups) > max_setups:
        setups.pop(0)


def _plot_chart(df: pd.DataFrame, trades: list[dict], output_path: str, setups: Optional[list[dict]] = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        print(f"[PLOT] matplotlib not available: {exc}")
        return

    df_plot = df.copy()
    df_plot["date_num"] = mdates.date2num(df_plot["timestamp"].dt.to_pydatetime())
    dates = df_plot["date_num"].values
    if len(dates) < 2:
        print("[PLOT] Not enough data to plot.")
        return

    width = (dates[1] - dates[0]) * 0.6
    min_body = (df_plot["high"] - df_plot["low"]).mean() * 0.001
    if not min_body or min_body <= 0:
        min_body = 1e-8
    y_min = float(df_plot["low"].min())
    y_max = float(df_plot["high"].max())
    price_range = max(y_max - y_min, min_body)
    arrow_offset = price_range * 0.03
    entry_color = "#f5c542"
    sl_color = "red"
    tp_color = "green"
    close_color = "blue"

    fig, ax = plt.subplots(figsize=(16, 9))
    for _, row in df_plot.iterrows():
        t = row["date_num"]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        color = "green" if c >= o else "red"
        ax.vlines(t, l, h, color=color, linewidth=0.7, alpha=0.9)
        body_low = min(o, c)
        body_high = max(o, c)
        body_h = max(body_high - body_low, min_body)
        ax.add_patch(
            Rectangle(
                (t - width / 2, body_low),
                width,
                body_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
                alpha=0.8,
            )
        )

    last_date = df_plot["date_num"].iloc[-1]
    setups = setups or []
    for idx, st in enumerate(setups, start=1):
        entry_time = mdates.date2num(st["entry_time"].to_pydatetime())
        _annotate_arrow(
            ax,
            entry_time,
            st["entry"],
            f"S{idx}",
            entry_color,
            y_min,
            y_max,
            arrow_offset,
            prefer_up=True,
        )
        ax.plot(
            [entry_time, last_date],
            [st["entry"], st["entry"]],
            color=entry_color,
            linewidth=0.7,
            alpha=0.7,
            linestyle="--",
        )
        ax.plot(
            [entry_time, last_date],
            [st["sl"], st["sl"]],
            color=sl_color,
            linewidth=0.6,
            alpha=0.7,
            linestyle="--",
        )
        ax.plot(
            [entry_time, last_date],
            [st["tp"], st["tp"]],
            color=tp_color,
            linewidth=0.6,
            alpha=0.7,
            linestyle="--",
        )

    for idx, tr in enumerate(trades, start=1):
        entry_time = mdates.date2num(tr["entry_time"].to_pydatetime())
        exit_time = last_date
        if tr.get("exit_time") is not None:
            exit_time = mdates.date2num(tr["exit_time"].to_pydatetime())
        _annotate_arrow(
            ax,
            entry_time,
            tr["entry"],
            f"T{idx}E",
            entry_color,
            y_min,
            y_max,
            arrow_offset,
            prefer_up=True,
        )
        exit_price = tr.get("exit_price")
        if tr.get("exit_time") is not None and exit_price is not None:
            exit_dt = mdates.date2num(tr["exit_time"].to_pydatetime())
            _annotate_arrow(
                ax,
                exit_dt,
                float(exit_price),
                f"T{idx}C",
                close_color,
                y_min,
                y_max,
                arrow_offset,
                prefer_up=False,
            )
        ax.plot(
            [entry_time, exit_time],
            [tr["entry"], tr["entry"]],
            color=entry_color,
            linewidth=0.9,
            alpha=0.9,
        )
        ax.plot([entry_time, exit_time], [tr["sl"], tr["sl"]], color=sl_color, linewidth=0.8, alpha=0.9)
        ax.plot([entry_time, exit_time], [tr["tp"], tr["tp"]], color=tp_color, linewidth=0.8, alpha=0.9)

    if setups:
        lines = ["Setups (latest):"]
        for idx, st in enumerate(setups, start=1):
            parts = [
                f"S{idx}",
                st["side"],
                f"e={_format_level(st.get('entry'))}",
                f"sl={_format_level(st.get('sl'))}",
                f"tp={_format_level(st.get('tp'))}",
            ]
            entry_src = st.get("entry_src")
            if entry_src:
                parts.append(entry_src)
            status = st.get("status")
            if status:
                parts.append(status)
            lines.append(" ".join(parts))
        ax.text(
            0.01,
            0.99,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7, boxstyle="round,pad=0.3"),
        )

    ax.set_title("SMC + AI visual backtest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved: {output_path}")


async def _run():
    args = _parse_args()
    show_setups = bool(args.show_setups or args.preview)
    max_setups = max(1, int(args.max_setups))
    use_ai = bool(getattr(Config, "AI_ENABLED", True)) and not args.no_ai
    if args.seed is not None:
        try:
            import numpy as np
            np.random.seed(int(args.seed))
        except Exception:
            pass

    symbol = args.symbol
    timeframe = args.timeframe
    limit = int(args.limit)
    output_path = _resolve_output(symbol, timeframe, args.output)
    _ensure_output_dir(output_path)

    loader = ExchangeLoader()
    smc = SMCAnalyzer()
    brain = NeuroBrain()

    df = await loader.fetch_candles(symbol, timeframe, limit=limit)
    if df is None or df.empty:
        print("[DATA] No candles returned.")
        return
    df = df.reset_index(drop=True)

    macro_df = None
    if getattr(Config, "SMC_USE_HTF_FILTER", False):
        macro_df = await loader.fetch_candles(symbol, Config.TF_MACRO, limit=300)

    scan_limit = min(350, int(Config.TRAINING_LOOKBACK_CANDLES))
    start_idx = max(60, scan_limit, 260)
    if len(df) < start_idx + 5:
        print("[DATA] Not enough candles for AI/SMC warmup.")
        return

    trades: list[dict] = []
    setups: list[dict] = []
    open_trade = None

    for i in range(start_idx, len(df)):
        candle = df.iloc[i]

        if open_trade and i > open_trade["open_index"]:
            reason, exit_price = _check_exit(open_trade, candle)
            if reason:
                open_trade["exit_time"] = candle["timestamp"]
                open_trade["exit_price"] = float(exit_price)
                open_trade["exit_reason"] = reason
                trades.append(open_trade)
                open_trade = None

        if open_trade is not None:
            continue

        window = df.iloc[: i + 1]
        df_scan = window.tail(scan_limit)
        if args.preview and show_setups:
            preview_signal, preview_setup, _ = smc.analyze(
                df_scan,
                require_touch=False,
                enforce_filters=not args.relax_filters,
            )
            if preview_signal and preview_setup:
                preview_side = "LONG" if preview_signal == "LONG" else "SHORT"
                _add_setup(
                    setups,
                    preview_setup,
                    preview_side,
                    df_scan["timestamp"].iloc[-1],
                    max_setups,
                    "PREVIEW",
                )

        signal, setup, _ = smc.analyze(df_scan)
        if not signal or not setup:
            continue

        side = "LONG" if signal == "LONG" else "SHORT"

        htf_ok = True
        if getattr(Config, "SMC_USE_HTF_FILTER", False) and macro_df is not None:
            macro_slice = macro_df[macro_df["timestamp"] <= df_scan["timestamp"].iloc[-1]]
            htf_ok = smc.passes_htf_filter(signal, macro_slice)

        prob = None
        ai_ok = True
        ai_reason = None
        if use_ai and htf_ok:
            df_ai = window.tail(int(Config.TRAINING_LOOKBACK_CANDLES))
            if not brain.mcpt_validation(symbol, df_ai):
                ai_ok = False
                ai_reason = "AI_VALIDATION"
            else:
                prob = brain.predict(symbol, df_ai, signal)
                if prob < float(Config.AI_CONFIDENCE_THRESHOLD):
                    ai_ok = False
                    ai_reason = "AI_CONF"

        if show_setups and not args.preview:
            status = "TRADE"
            if not htf_ok:
                status = "SKIP_HTF"
            elif not ai_ok:
                status = f"SKIP_{ai_reason}" if ai_reason else "SKIP_AI"
            _add_setup(
                setups,
                setup,
                side,
                df_scan["timestamp"].iloc[-1],
                max_setups,
                status,
            )

        if not htf_ok or not ai_ok:
            continue

        st = brain.states.get(symbol)
        p_value = st.last_p_value if st else None
        real_score = st.last_real_score if st else None

        open_trade = {
            "symbol": symbol,
            "side": side,
            "entry_time": df_scan["timestamp"].iloc[-1],
            "entry": float(setup["entry"]),
            "sl": float(setup["sl"]),
            "tp": float(setup["tp"]),
            "open_index": i,
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "ai_prob": float(prob) if prob is not None else None,
            "ai_p_value": float(p_value) if p_value is not None else None,
            "ai_real_score": float(real_score) if real_score is not None else None,
        }

    if open_trade is not None:
        trades.append(open_trade)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        out_csv = os.path.splitext(output_path)[0] + "_trades.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"[DATA] Trades saved: {out_csv}")
    else:
        print("[DATA] No trades triggered.")

    if show_setups:
        if setups:
            print(f"[DATA] Setups shown: {len(setups)} (max {max_setups})")
        else:
            print("[DATA] No setups found for preview/plot.")

    _plot_chart(df, trades, output_path, setups if show_setups else None)
    await loader.close_connection()


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(0)
