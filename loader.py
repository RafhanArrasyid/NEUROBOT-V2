import asyncio
import json
import logging
import os
import time
from typing import Any, Optional, Callable

import ccxt
import pandas as pd

from config import Config


class BinanceSyncWrapper:
    """Membungkus ccxt sync agar bisa dipakai async via to_thread.

    Catatan penting:
    - ccxt sync client tidak didesain aman untuk akses paralel lintas thread.
    - Maka semua network call diproteksi oleh lock (serial), supaya stabil 24/7.
    """

    def __init__(self, config: dict[str, Any], on_error: Optional[Callable[[Exception], None]] = None):
        self.client = ccxt.binance(config)
        self._lock = asyncio.Lock()
        self._error_streak = 0
        self._backoff_until = 0.0
        self._on_error = on_error

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def set_on_error(self, handler: Optional[Callable[[Exception], None]]):
        self._on_error = handler

    async def _call(self, fn, *args, **kwargs):
        async with self._lock:
            try:
                if getattr(Config, 'API_BACKOFF_ENABLED', True):
                    delay = max(0.0, float(self._backoff_until) - time.time())
                    if delay > 0:
                        await asyncio.sleep(delay)
            except Exception:
                pass
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:
                if self._on_error:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass
                try:
                    if getattr(Config, 'API_BACKOFF_ENABLED', True):
                        self._error_streak = min(int(self._error_streak) + 1, 10)
                        base = float(getattr(Config, 'API_BACKOFF_BASE_SEC', 1.0))
                        max_b = float(getattr(Config, 'API_BACKOFF_MAX_SEC', 30.0))
                        if base < 0:
                            base = 0.0
                        if max_b < 0:
                            max_b = 0.0
                        backoff = min(max_b, base * (2 ** max(self._error_streak - 1, 0)))
                        self._backoff_until = time.time() + float(backoff)
                except Exception:
                    pass
                raise
            else:
                self._error_streak = 0
                return result

    async def load_markets(self):
        return await self._call(self.client.load_markets)

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        return await self._call(self.client.fetch_ohlcv, symbol, timeframe, limit=limit)

    async def fetch_ticker(self, symbol):
        return await self._call(self.client.fetch_ticker, symbol)

    async def fetch_balance(self):
        return await self._call(self.client.fetch_balance)

    async def fetch_positions(self, symbols=None):
        return await self._call(self.client.fetch_positions, symbols)

    async def fetch_open_orders(self, symbol=None):
        return await self._call(self.client.fetch_open_orders, symbol)

    async def fetch_order(self, id, symbol=None):
        return await self._call(self.client.fetch_order, id, symbol)

    async def create_order(self, symbol, type, side, amount, price=None, params=None):
        if params is None:
            params = {}
        return await self._call(self.client.create_order, symbol, type, side, amount, price, params)

    async def edit_order(self, id, symbol, type=None, side=None, amount=None, price=None, params=None):
        if params is None:
            params = {}
        if not hasattr(self.client, 'edit_order'):
            raise AttributeError("edit_order not supported")
        return await self._call(self.client.edit_order, id, symbol, type, side, amount, price, params)

    async def cancel_order(self, id, symbol):
        return await self._call(self.client.cancel_order, id, symbol)

    async def cancel_all_orders(self, symbol):
        if hasattr(self.client, 'cancel_all_orders'):
            return await self._call(self.client.cancel_all_orders, symbol)
        # fallback manual: fetch_open_orders + cancel
        orders = await self.fetch_open_orders(symbol)
        for o in orders:
            try:
                await self.cancel_order(o['id'], symbol)
            except Exception:
                pass
        return True

    async def set_leverage(self, leverage: int, symbol: str):
        # Tidak semua market support; lindungi dengan try.
        if hasattr(self.client, 'set_leverage'):
            return await self._call(self.client.set_leverage, leverage, symbol)
        return None

    async def close(self):
        # ccxt sync tidak butuh close
        return None


class ExchangeLoader:
    def __init__(self, alert_handler: Optional[Callable[[str, str], None]] = None):
        self.exchange: Optional[BinanceSyncWrapper] = None
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self.paper_state_file = os.path.join(self._base_dir, "paper_wallet.json")
        self._markets_loaded = False
        self._symbol_cache: dict[str, str] = {}
        self._alert_handler = alert_handler
        self._logger = logging.getLogger("neurobot.loader")
        self._init_exchange()
        self._init_paper_wallet()

    def set_alert_handler(self, handler: Optional[Callable[[str, str], None]]):
        self._alert_handler = handler
        if self.exchange:
            self.exchange.set_on_error(self._handle_exchange_error)

    def _emit_alert(self, message: str, level: str = "WARN"):
        if self._alert_handler:
            try:
                self._alert_handler(message, level)
                return
            except Exception:
                pass
        try:
            if level == "ERROR":
                self._logger.error(message)
            elif level == "WARN":
                self._logger.warning(message)
            else:
                self._logger.info(message)
        except Exception:
            pass

    def _extract_http_status(self, exc: Exception) -> Optional[int]:
        for attr in ("status", "http_status", "status_code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
            try:
                if val is not None:
                    return int(val)
            except Exception:
                continue
        msg = str(exc)
        if "418" in msg:
            return 418
        if "429" in msg:
            return 429
        return None

    def _handle_exchange_error(self, exc: Exception):
        status = self._extract_http_status(exc)
        is_rate = isinstance(exc, (ccxt.RateLimitExceeded, ccxt.DDoSProtection))
        if status not in (418, 429) and not is_rate:
            return
        retry_after = None
        try:
            headers = getattr(self.exchange, "client", None)
            headers = getattr(headers, "last_response_headers", None) if headers else None
            if isinstance(headers, dict):
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:
            retry_after = None
        extra = f" retry_after={retry_after}" if retry_after else ""
        code = status if status is not None else "n/a"
        self._emit_alert(f"Binance API rate limit/ban detected HTTP {code}: {exc}{extra}", "WARN")

    def _init_exchange(self):
        config = {
            'apiKey': Config.API_KEY,
            'secret': Config.SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
        }
        self.exchange = BinanceSyncWrapper(config, on_error=self._handle_exchange_error)

    def _init_paper_wallet(self):
        if Config.TRADING_MODE != "PAPER":
            return
        if not os.path.exists(self.paper_state_file):
            initial_data = {
                "balance": float(Config.PAPER_INITIAL_BALANCE),
                "positions": {},
                "history": [],
            }
            with open(self.paper_state_file, 'w') as f:
                json.dump(initial_data, f, indent=4)
            print(f"[LOADER] Paper Wallet created. Balance: ${Config.PAPER_INITIAL_BALANCE}")

    async def ensure_markets(self):
        if not self.exchange:
            return
        if self._markets_loaded and getattr(self.exchange, 'markets', None):
            return
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True
        except Exception:
            # Kalau load_markets gagal, kita lanjut; beberapa fungsi masih bisa jalan.
            self._markets_loaded = False

    async def get_balance(self) -> float:
        if Config.TRADING_MODE == "LIVE":
            try:
                bal = await self.exchange.fetch_balance()
                # Binance futures kadang meletakkan USDT di total / free / used
                if isinstance(bal, dict):
                    total = bal.get('total', {}) or {}
                    free = bal.get('free', {}) or {}
                    used = bal.get('used', {}) or {}
                    usdt = float(total.get('USDT', 0) or 0)
                    if usdt == 0:
                        usdt = float(free.get('USDT', 0) or 0) + float(used.get('USDT', 0) or 0)
                    return float(usdt)
                return 0.0
            except Exception:
                return 0.0

        # PAPER
        try:
            with open(self.paper_state_file, 'r') as f:
                data = json.load(f)
            return float(data.get("balance", 0.0))
        except Exception:
            return 0.0

    def _timeframe_to_seconds(self, timeframe: str) -> Optional[int]:
        try:
            tf = (timeframe or "").strip()
            if not tf:
                return None
            unit = tf[-1]
            value = int(tf[:-1])
        except Exception:
            return None
        if unit == 'm':
            return value * 60
        if unit == 'h':
            return value * 60 * 60
        if unit == 'd':
            return value * 24 * 60 * 60
        if unit == 'w':
            return value * 7 * 24 * 60 * 60
        if unit == 'M':
            return value * 30 * 24 * 60 * 60
        return None

    def _alt_symbol(self, symbol: str) -> Optional[str]:
        sym = (symbol or "").strip()
        if not sym or ":" in sym:
            return None
        if "/" not in sym:
            return None
        base, quote = sym.split("/", 1)
        if not base or not quote:
            return None
        return f"{base}/{quote}:{quote}"

    def resolve_symbol(self, symbol: str) -> str:
        sym = (symbol or "").strip()
        if not sym:
            return symbol
        cached = self._symbol_cache.get(sym)
        if cached:
            return cached
        markets = getattr(self.exchange, "markets", None) if self.exchange else None
        if not markets:
            return sym
        if sym in markets:
            self._symbol_cache[sym] = sym
            return sym
        alt = self._alt_symbol(sym)
        if alt and alt in markets:
            self._symbol_cache[sym] = alt
            return alt
        if ":" in sym:
            base = sym.split(":", 1)[0]
            if base in markets:
                self._symbol_cache[sym] = base
                return base
        return sym

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        try:
            await self.ensure_markets()
            sym = self.resolve_symbol(symbol)
            ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.dropna(inplace=True)
            try:
                tf_sec = self._timeframe_to_seconds(timeframe)
                if tf_sec and not df.empty:
                    last_ts = df['timestamp'].iloc[-1]
                    now = pd.Timestamp.utcnow()
                    if last_ts + pd.Timedelta(seconds=tf_sec) > now:
                        df = df.iloc[:-1]
            except Exception:
                pass
            return df
        except Exception:
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            await self.ensure_markets()
            sym = self.resolve_symbol(symbol)
            ticker = await self.exchange.fetch_ticker(sym)
            last = ticker.get('last') if isinstance(ticker, dict) else None
            if last is None:
                last = (ticker.get('close') if isinstance(ticker, dict) else None)
            return float(last) if last is not None else None
        except Exception:
            return None

    async def fetch_positions(self):
        try:
            await self.ensure_markets()
            if hasattr(self.exchange, 'fetch_positions'):
                return await self.exchange.fetch_positions()
        except Exception:
            return None
        return []

    async def fetch_open_orders(self, symbol: Optional[str] = None):
        try:
            await self.ensure_markets()
            sym = self.resolve_symbol(symbol) if symbol else None
            return await self.exchange.fetch_open_orders(sym)
        except Exception:
            return None

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            sym = self.resolve_symbol(symbol)
            return float(self.exchange.amount_to_precision(sym, amount))
        except Exception:
            return float(amount)

    def price_to_precision(self, symbol: str, price: float) -> float:
        try:
            sym = self.resolve_symbol(symbol)
            return float(self.exchange.price_to_precision(sym, price))
        except Exception:
            return float(price)

    def get_market_limits(self, symbol: str) -> dict[str, Any]:
        try:
            sym = self.resolve_symbol(symbol)
            m = (self.exchange.markets or {}).get(sym, {})
            return {
                'amount_min': (m.get('limits', {}).get('amount', {}) or {}).get('min'),
                'cost_min': (m.get('limits', {}).get('cost', {}) or {}).get('min'),
            }
        except Exception:
            return {'amount_min': None, 'cost_min': None}

    async def close_connection(self):
        if self.exchange:
            await self.exchange.close()
