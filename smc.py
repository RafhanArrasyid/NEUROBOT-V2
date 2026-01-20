import pandas as pd
import numpy as np
from typing import Optional

from config import Config

class SMCAnalyzer:
    """
    Modul Analisa Smart Money Concepts (SMC) & ICT.
    Fokus: Mendeteksi Market Structure Shift (MSS), Fair Value Gaps (FVG), 
    dan Order Blocks (OB).
    """

    def __init__(self):
        pass

    def _cfg_int(self, name: str, default: int) -> int:
        try:
            return int(getattr(Config, name, default))
        except Exception:
            return int(default)

    def _cfg_float(self, name: str, default: float) -> float:
        try:
            return float(getattr(Config, name, default))
        except Exception:
            return float(default)

    def _cfg_bool(self, name: str, default: bool) -> bool:
        try:
            return bool(getattr(Config, name, default))
        except Exception:
            return bool(default)

    def _parse_hour(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            if ":" in raw:
                parts = raw.split(":")
                try:
                    h = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 0
                    return float(h) + (float(m) / 60.0)
                except Exception:
                    return None
            try:
                return float(raw)
            except Exception:
                return None
        return None

    def _hour_in_windows(self, hour_val: float, windows: list) -> bool:
        if hour_val is None:
            return False
        for win in windows:
            start = None
            end = None
            if isinstance(win, dict):
                start = win.get("start")
                end = win.get("end")
            elif isinstance(win, (list, tuple)) and len(win) >= 2:
                start, end = win[0], win[1]
            s_val = self._parse_hour(start)
            e_val = self._parse_hour(end)
            if s_val is None or e_val is None:
                continue
            if s_val <= e_val:
                if s_val <= hour_val <= e_val:
                    return True
            else:
                # window melewati tengah malam
                if hour_val >= s_val or hour_val <= e_val:
                    return True
        return False

    def identify_swings(self, df: pd.DataFrame, pivot_len: int = 3):
        """
        Mendeteksi Swing High dan Swing Low (Fractals).
        pivot_len = jumlah candle kiri/kanan yang harus lebih rendah/tinggi.
        """
        df = df.copy()
        
        # Logika Swing High: High candle ini > High pivot_len candle kiri & kanan
        df['swing_high'] = False
        df['swing_low'] = False

        # Menggunakan window rolling untuk cek local max/min
        # (Implementasi manual loop dioptimalkan dengan numpy untuk kecepatan)
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        # Array untu menyimpan harga swing
        swing_high_prices = np.full(n, np.nan)
        swing_low_prices = np.full(n, np.nan)

        for i in range(pivot_len, n - pivot_len):
            # Cek Swing High
            if all(highs[i] > highs[i-k] for k in range(1, pivot_len+1)) and \
               all(highs[i] > highs[i+k] for k in range(1, pivot_len+1)):
                df.at[df.index[i], 'swing_high'] = True
                swing_high_prices[i] = highs[i]

            # Cek Swing Low
            if all(lows[i] < lows[i-k] for k in range(1, pivot_len+1)) and \
               all(lows[i] < lows[i+k] for k in range(1, pivot_len+1)):
                df.at[df.index[i], 'swing_low'] = True
                swing_low_prices[i] = lows[i]

        df['sh_price'] = pd.Series(swing_high_prices, index=df.index).ffill()
        df['sl_price'] = pd.Series(swing_low_prices, index=df.index).ffill()
        
        return df

    def detect_fvg(self, df: pd.DataFrame):
        """
        Mendeteksi Fair Value Gap (Imbalance).
        Bullish FVG: Low candle[0] > High candle[2]
        Bearish FVG: High candle[0] < Low candle[2]
        """
        df['fvg_bull'] = False
        df['fvg_bear'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        # Shift data untuk perbandingan 3 candle (ICT concept)
        # Candle saat ini (0), Kemarin (-1), Dua hari lalu (-2)
        # FVG terbentuk di candle -1 (tengah)
        
        high = df['high']
        low = df['low']
        
        # Bullish FVG: Low candle sekarang > High 2 candle lalu
        # Kita cek pada candle yang sudah close (i-1 sebagai konfirmasi)
        bull_cond = (low > high.shift(2))
        
        # Bearish FVG: High candle sekarang < Low 2 candle lalu
        bear_cond = (high < low.shift(2))

        df.loc[bull_cond, 'fvg_bull'] = True
        df.loc[bull_cond, 'fvg_top'] = low            # Area atas gap
        df.loc[bull_cond, 'fvg_bottom'] = high.shift(2) # Area bawah gap

        df.loc[bear_cond, 'fvg_bear'] = True
        df.loc[bear_cond, 'fvg_top'] = low.shift(2)     # Area atas gap
        df.loc[bear_cond, 'fvg_bottom'] = high          # Area bawah gap

        return df

    def _annotate_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['body'] = (df['close'] - df['open']).abs()
        window = max(2, self._cfg_int('SMC_DISPLACEMENT_WINDOW', 20))
        mult = max(0.1, self._cfg_float('SMC_DISPLACEMENT_MULT', 1.5))
        body_avg = df['body'].rolling(window=window, min_periods=2).mean()
        body_avg = body_avg.fillna(df['body'])
        df['displacement'] = df['body'] >= (body_avg * mult)
        return df

    def _find_latest_sweep(self, df: pd.DataFrame) -> Optional[dict]:
        lookback = max(3, self._cfg_int('SMC_SWEEP_LOOKBACK', 12))
        require_close_back = self._cfg_bool('SMC_SWEEP_REQUIRE_CLOSE_BACK', True)
        n = len(df)
        start_idx = max(2, n - lookback)
        for i in range(n - 1, start_idx - 1, -1):
            prev_sh = df['sh_price'].iloc[i - 1]
            prev_sl = df['sl_price'].iloc[i - 1]
            close_i = df['close'].iloc[i]
            high_i = df['high'].iloc[i]
            low_i = df['low'].iloc[i]

            if pd.notna(prev_sl) and low_i < prev_sl:
                if (not require_close_back) or (close_i > prev_sl):
                    return {
                        "side": "SELL",
                        "idx": i,
                        "level": float(prev_sl),
                        "sweep_price": float(low_i),
                    }
            if pd.notna(prev_sh) and high_i > prev_sh:
                if (not require_close_back) or (close_i < prev_sh):
                    return {
                        "side": "BUY",
                        "idx": i,
                        "level": float(prev_sh),
                        "sweep_price": float(high_i),
                    }
        return None

    def _find_mss_after_sweep(self, df: pd.DataFrame, sweep: dict) -> Optional[dict]:
        lookahead = max(2, self._cfg_int('SMC_MSS_LOOKAHEAD', 12))
        require_disp = self._cfg_bool('SMC_REQUIRE_DISPLACEMENT', True)
        start = sweep['idx'] + 1
        end = min(len(df) - 1, sweep['idx'] + lookahead)
        if start > end:
            return None
        for i in range(end, start - 1, -1):
            if sweep['side'] == "SELL":
                level = df['sh_price'].iloc[i - 1] if i - 1 >= 0 else np.nan
                if pd.notna(level) and df['close'].iloc[i] > level:
                    if (not require_disp) or bool(df['displacement'].iloc[i]):
                        return {"side": "BULLISH", "idx": i, "level": float(level)}
            else:
                level = df['sl_price'].iloc[i - 1] if i - 1 >= 0 else np.nan
                if pd.notna(level) and df['close'].iloc[i] < level:
                    if (not require_disp) or bool(df['displacement'].iloc[i]):
                        return {"side": "BEARISH", "idx": i, "level": float(level)}
        return None

    def _fvg_invalidated(self, df: pd.DataFrame, idx: int, top: float, bottom: float, side: str) -> bool:
        if idx >= len(df) - 1:
            return False
        tail = df.iloc[idx + 1:]
        if tail.empty:
            return False
        if side == "BULL":
            return bool((tail['close'] < bottom).any())
        return bool((tail['close'] > top).any())

    def _find_recent_fvg(self, df: pd.DataFrame, sweep_idx: int, mss_idx: int, direction: str) -> Optional[dict]:
        lookback = max(3, self._cfg_int('SMC_FVG_LOOKBACK', 12))
        require_after_mss = self._cfg_bool('SMC_FVG_AFTER_MSS', True)
        require_unmitigated = self._cfg_bool('SMC_REQUIRE_UNMITIGATED_FVG', True)
        allow_ifvg = self._cfg_bool('SMC_ALLOW_IFVG', True)
        min_idx = max(0, len(df) - 1 - lookback)
        if require_after_mss:
            min_idx = max(min_idx, mss_idx - 2)
        for i in range(len(df) - 1, min_idx - 1, -1):
            if direction == "LONG" and bool(df['fvg_bull'].iloc[i]):
                top = df['fvg_top'].iloc[i]
                bottom = df['fvg_bottom'].iloc[i]
                if pd.isna(top) or pd.isna(bottom):
                    continue
                if require_unmitigated and self._fvg_invalidated(df, i, float(top), float(bottom), "BULL"):
                    continue
                return {"idx": i, "top": float(top), "bottom": float(bottom), "type": "FVG"}
            if direction == "SHORT" and bool(df['fvg_bear'].iloc[i]):
                top = df['fvg_top'].iloc[i]
                bottom = df['fvg_bottom'].iloc[i]
                if pd.isna(top) or pd.isna(bottom):
                    continue
                if require_unmitigated and self._fvg_invalidated(df, i, float(top), float(bottom), "BEAR"):
                    continue
                return {"idx": i, "top": float(top), "bottom": float(bottom), "type": "FVG"}

        if not allow_ifvg:
            return None

        # IFVG: FVG yang gagal lalu jadi area kebalikan
        for i in range(len(df) - 1, min_idx - 1, -1):
            if direction == "LONG" and bool(df['fvg_bear'].iloc[i]):
                top = df['fvg_top'].iloc[i]
                bottom = df['fvg_bottom'].iloc[i]
                if pd.isna(top) or pd.isna(bottom):
                    continue
                if not self._fvg_invalidated(df, i, float(top), float(bottom), "BEAR"):
                    continue
                return {"idx": i, "top": float(top), "bottom": float(bottom), "type": "IFVG"}
            if direction == "SHORT" and bool(df['fvg_bull'].iloc[i]):
                top = df['fvg_top'].iloc[i]
                bottom = df['fvg_bottom'].iloc[i]
                if pd.isna(top) or pd.isna(bottom):
                    continue
                if not self._fvg_invalidated(df, i, float(top), float(bottom), "BULL"):
                    continue
                return {"idx": i, "top": float(top), "bottom": float(bottom), "type": "IFVG"}
        return None

    def _find_order_block(self, df: pd.DataFrame, sweep_idx: int, mss_idx: int, direction: str) -> Optional[dict]:
        lookback = max(2, self._cfg_int('SMC_OB_LOOKBACK', 8))
        use_body = self._cfg_bool('SMC_OB_USE_BODY', True)
        start = max(sweep_idx, mss_idx - lookback)
        for i in range(mss_idx - 1, start - 1, -1):
            o = float(df['open'].iloc[i])
            c = float(df['close'].iloc[i])
            h = float(df['high'].iloc[i])
            l = float(df['low'].iloc[i])
            if direction == "LONG" and c < o:
                top = max(o, c) if use_body else h
                bottom = min(o, c) if use_body else l
                return {"idx": i, "top": float(top), "bottom": float(bottom)}
            if direction == "SHORT" and c > o:
                top = max(o, c) if use_body else h
                bottom = min(o, c) if use_body else l
                return {"idx": i, "top": float(top), "bottom": float(bottom)}
        return None

    def _select_entry_zone(self, fvg: dict, ob: dict) -> Optional[tuple[float, float, str]]:
        use_overlap = self._cfg_bool('SMC_USE_OB_FVG_OVERLAP', True)
        fvg_low = min(float(fvg['top']), float(fvg['bottom']))
        fvg_high = max(float(fvg['top']), float(fvg['bottom']))
        ob_low = min(float(ob['top']), float(ob['bottom']))
        ob_high = max(float(ob['top']), float(ob['bottom']))

        if use_overlap:
            zone_low = max(fvg_low, ob_low)
            zone_high = min(fvg_high, ob_high)
            if zone_low < zone_high:
                return zone_low, zone_high, "OVERLAP"

        if fvg_low < fvg_high:
            return fvg_low, fvg_high, "FVG"
        if ob_low < ob_high:
            return ob_low, ob_high, "OB"
        return None

    def _price_touched_zone(self, candle: pd.Series, zone_low: float, zone_high: float) -> bool:
        if candle is None:
            return False
        require_close = self._cfg_bool('SMC_ENTRY_REQUIRE_CLOSE_IN_ZONE', False)
        try:
            low = float(candle['low'])
            high = float(candle['high'])
            close = float(candle['close'])
        except Exception:
            return False
        if require_close:
            return zone_low <= close <= zone_high
        return low <= zone_high and high >= zone_low

    def _build_sl_tp(self, df: pd.DataFrame, direction: str, entry: float, sweep: dict, ob: dict) -> Optional[tuple[float, float, float]]:
        buffer_pct = max(0.0, self._cfg_float('SMC_SL_BUFFER_PCT', 0.001))
        min_sl_pct = max(0.0, self._cfg_float('SMC_MIN_SL_PCT', 0.002))
        max_sl_pct = max(min_sl_pct, self._cfg_float('SMC_MAX_SL_PCT', 0.05))
        rr_min = max(0.1, self._cfg_float('MIN_RR_RATIO', getattr(Config, 'MIN_RR_RATIO', 2.0)))
        # Compute volatility (ATR) based stop distance.  This uses a simple ATR proxy based on the average of
        # (high - low) over the last N candles.  It enforces a minimum stop distance equal to ATR * multiplier.
        atr_window = max(1, int(self._cfg_int('SMC_ATR_WINDOW', 14)))
        try:
            if len(df) >= atr_window:
                # average true range proxy: average of high‑low over the window
                atr = (df['high'].iloc[-atr_window:] - df['low'].iloc[-atr_window:]).mean()
            else:
                atr = (df['high'] - df['low']).mean()
        except Exception:
            atr = 0.0
        atr_val = float(atr) if atr is not None else 0.0
        mult = float(self._cfg_float('SMC_ATR_MULT', 1.0))
        vol_dist = atr_val * mult

        entry_f = float(entry)
        if entry_f <= 0:
            return None

        if direction == "LONG":
            # Determine stop using both structure and volatility.  Anchor stop is below the sweep/OB levels.
            sl_anchor_price = min(float(sweep['sweep_price']), float(sweep['level']), float(ob['bottom']))
            anchor_sl = sl_anchor_price * (1.0 - buffer_pct)
            # Volatility‑based stop is at least ATR away from entry.
            vol_sl = entry_f - vol_dist
            # Choose the lower price (further from entry) to honour both structure and volatility.
            sl = min(anchor_sl, vol_sl)
            # ensure stop positive and not NaN
            if sl <= 0 or np.isnan(sl):
                return None
            dist_pct = (entry_f - sl) / entry_f
            # Enforce min/max stop distance relative to entry
            if dist_pct < min_sl_pct:
                # If volatility‑based stop is too tight, expand using min_sl_pct
                sl = entry_f * (1.0 - min_sl_pct)
                dist_pct = (entry_f - sl) / entry_f
            if dist_pct > max_sl_pct:
                return None
            r = entry_f - sl
            # Determine take‑profit: use last swing high if available and meets RR; otherwise compute based on rr_min
            last_sh = df['sh_price'].iloc[-2] if len(df) >= 2 else np.nan
            tp_candidate = float(last_sh) if pd.notna(last_sh) else None
            if tp_candidate is None or tp_candidate <= entry_f:
                tp_candidate = None
            if tp_candidate is None or ((tp_candidate - entry_f) / r) < rr_min:
                tp = entry_f + (rr_min * r)
            else:
                tp = tp_candidate
            return float(sl), float(tp), float(r)

        # SHORT side: structure‑based and volatility‑based stops above the entry
        sl_anchor_price = max(float(sweep['sweep_price']), float(sweep['level']), float(ob['top']))
        anchor_sl = sl_anchor_price * (1.0 + buffer_pct)
        vol_sl = entry_f + vol_dist
        sl = max(anchor_sl, vol_sl)
        dist_pct = (sl - entry_f) / entry_f
        if dist_pct < min_sl_pct:
            # Expand to minimum stop distance
            sl = entry_f * (1.0 + min_sl_pct)
            dist_pct = (sl - entry_f) / entry_f
        if dist_pct > max_sl_pct:
            return None
        r = sl - entry_f
        last_sl = df['sl_price'].iloc[-2] if len(df) >= 2 else np.nan
        tp_candidate = float(last_sl) if pd.notna(last_sl) else None
        if tp_candidate is None or tp_candidate >= entry_f:
            tp_candidate = None
        if tp_candidate is None or ((entry_f - tp_candidate) / r) < rr_min:
            tp = entry_f - (rr_min * r)
        else:
            tp = tp_candidate
        if tp <= 0:
            return None
        return float(sl), float(tp), float(r)

    def check_market_structure(self, df: pd.DataFrame):
        """
        Menentukan arah tren berdasarkan Break of Structure (BOS) / MSS.
        """
        # Ambil swing terakhir
        last_sh = df['sh_price'].iloc[-2] # Swing High terakhir (bukan candle curr)
        last_sl = df['sl_price'].iloc[-2] # Swing Low terakhir
        
        curr_close = df['close'].iloc[-1]
        
        trend = "NEUTRAL"
        
        # Logic MSS Simple:
        # Jika close menembus Swing High terakhir -> Bullish MSS
        if pd.notna(last_sh) and curr_close > last_sh:
            trend = "BULLISH"
            
        # Jika close menembus Swing Low terakhir -> Bearish MSS
        elif pd.notna(last_sl) and curr_close < last_sl:
            trend = "BEARISH"
            
        return trend, last_sh, last_sl

    def _passes_volume_filter(self, df: pd.DataFrame) -> bool:
        if not getattr(Config, 'SMC_USE_VOLUME_FILTER', False):
            return True
        if df is None or df.empty or 'volume' not in df.columns:
            return True
        window = int(getattr(Config, 'SMC_VOLUME_WINDOW', 20))
        if len(df) < window + 1:
            return True
        vol_ma = df['volume'].rolling(window=window).mean().iloc[-1]
        if pd.isna(vol_ma) or float(vol_ma) <= 0:
            return True
        min_mult = float(getattr(Config, 'SMC_VOLUME_MIN_MULT', 1.0))
        return float(df['volume'].iloc[-1]) >= float(vol_ma) * min_mult

    def _passes_session_filter(self, df: pd.DataFrame) -> bool:
        windows = getattr(Config, 'SMC_ALLOWED_UTC_WINDOWS', [])
        if not windows:
            windows = getattr(Config, 'SMC_KILLZONE_WINDOWS_UTC', [])
        allowed_hours = getattr(Config, 'SMC_ALLOWED_UTC_HOURS', [])
        if not windows and not allowed_hours:
            return True
        if df is None or df.empty or 'timestamp' not in df.columns:
            return True
        ts = df['timestamp'].iloc[-1]
        try:
            hour_val = float(ts.hour) + (float(ts.minute) / 60.0)
        except Exception:
            return True
        if windows:
            return self._hour_in_windows(hour_val, windows)
        return int(hour_val) in set(int(h) for h in allowed_hours)

    def passes_htf_filter(self, signal: str, macro_df: Optional[pd.DataFrame]) -> bool:
        if not getattr(Config, 'SMC_USE_HTF_FILTER', False):
            return True
        if macro_df is None or macro_df.empty or len(macro_df) < 50:
            return True

        pivot_len = int(getattr(Config, 'SMC_HTF_PIVOT_LEN', 5))
        df_macro = self.identify_swings(macro_df, pivot_len=pivot_len)
        macro_trend, _, _ = self.check_market_structure(df_macro)
        if macro_trend == "BULLISH" and signal == "SHORT":
            return False
        if macro_trend == "BEARISH" and signal == "LONG":
            return False
        return True

    def analyze(self, df: pd.DataFrame, *, require_touch: bool = True, enforce_filters: bool = True):
        """
        FUNGSI UTAMA yang dipanggil oleh bot.
        Menggabungkan Liquidity Sweep, MSS, FVG, dan OB sesuai ICT/SMC murni.
        """
        # Selalu return tuple (signal, setup, df_debug) agar caller tidak crash.
        if df is None or df.empty:
            return None, None, df

        pivot_len = max(2, self._cfg_int('SMC_PIVOT_LEN', 3))
        min_len = max(60, pivot_len * 4)
        if len(df) < min_len:
            return None, None, df

        # 1) Identifikasi Swing Points
        df = self.identify_swings(df, pivot_len=pivot_len)

        # 2) FVG + Displacement
        df = self.detect_fvg(df)
        df = self._annotate_displacement(df)

        # 3) Deteksi Liquidity Sweep terbaru
        sweep = self._find_latest_sweep(df)
        if not sweep:
            return None, None, df

        # 4) Konfirmasi MSS/BOS setelah sweep
        mss = self._find_mss_after_sweep(df, sweep)
        if not mss:
            return None, None, df

        direction = "LONG" if mss["side"] == "BULLISH" else "SHORT"

        # 5) FVG (atau IFVG) yang relevan
        fvg = self._find_recent_fvg(df, sweep["idx"], mss["idx"], direction)
        if not fvg:
            return None, None, df

        # 6) Order Block terakhir sebelum displacement
        ob = self._find_order_block(df, sweep["idx"], mss["idx"], direction)
        if not ob:
            return None, None, df

        # 7) Zona entry (overlap OB+FVG jika ada)
        zone = self._select_entry_zone(fvg, ob)
        if not zone:
            return None, None, df
        zone_low, zone_high, zone_src = zone

        last_candle = df.iloc[-1]
        if require_touch and not self._price_touched_zone(last_candle, zone_low, zone_high):
            return None, None, df

        entry_price = float(last_candle['close'])
        entry_src = "CLOSE"
        if not require_touch:
            entry_price = (float(zone_low) + float(zone_high)) / 2.0
            entry_src = "ZONE_MID"

        sl_tp = self._build_sl_tp(df, direction, entry_price, sweep, ob)
        if not sl_tp and not require_touch:
            entry_price = float(last_candle['close'])
            entry_src = "CLOSE"
            sl_tp = self._build_sl_tp(df, direction, entry_price, sweep, ob)
        if not sl_tp:
            return None, None, df
        sl_price, tp_price, r = sl_tp

        if enforce_filters:
            if not self._passes_volume_filter(df):
                return None, None, df
            if not self._passes_session_filter(df):
                return None, None, df

        last_sh = df['sh_price'].iloc[-2] if len(df) >= 2 else np.nan
        last_sl = df['sl_price'].iloc[-2] if len(df) >= 2 else np.nan

        setup_details = {
            "entry": float(entry_price),
            "entry_src": entry_src,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "r": float(r),
            "rr": float(getattr(Config, 'MIN_RR_RATIO', 2.0)),
            "sh": float(last_sh) if pd.notna(last_sh) else None,
            "sl_structure": float(last_sl) if pd.notna(last_sl) else None,
            "zone_low": float(zone_low),
            "zone_high": float(zone_high),
            "zone_src": str(zone_src),
            "sweep_side": sweep["side"],
            "sweep_level": sweep["level"],
            "mss_level": mss["level"],
            "fvg_type": fvg.get("type"),
            "reason": "Sweep + MSS + OB/FVG"
        }

        return direction, setup_details, df

# Unit Test (Bisa dijalankan langsung untuk cek logika)
if __name__ == "__main__":
    # Buat data dummy untuk test
    data = {
        'open': [100, 102, 101, 105, 108, 107, 110],
        'high': [103, 104, 103, 109, 110, 108, 112],
        'low':  [99, 101, 100, 104, 106, 105, 108],
        'close': [102, 101, 102, 108, 107, 106, 111],
        'volume': [1000]*7
    }
    df_test = pd.DataFrame(data)
    
    smc = SMCAnalyzer()
    sig, details, df_res = smc.analyze(df_test)
    
    print("DataFrame Hasil Analisa:")
    print(df_res[['close', 'swing_high', 'swing_low', 'fvg_bull', 'fvg_bear']].tail())
    print("\nSignal:", sig)
    print("Details:", details)
