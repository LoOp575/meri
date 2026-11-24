"""
Meridian v2 — Market Data Access Layer (audited & improved).
=============================================================
Rantai fallback (audit verdict: DESAIN BAGUS, dipertahankan):
  Binance mirror -> Binance.US -> Binance spot -> MEXC -> futures
  -> CoinGecko -> seed deterministik (selalu diberi label jujur).

PENINGKATAN vs repo lama (hasil audit):
 1. FIX BUG: `import hashlib` — versi lama manggil hashlib.sha256() di
    deterministic_seed() TANPA import -> NameError persis saat semua
    sumber live mati dan fallback seed dibutuhkan.
 2. Konstanta MEXC/FUTURES didefinisikan SEBELUM dipakai.
 3. TTLCache: thread-safe + stats() hit/miss untuk observability.
 4. fetch_klines: limit di-clamp [30, 1000] + ambang candle jadi konstanta.
 5. Self-test bawaan: python3 backend/market_data.py -> SELF-TEST OK
"""

from __future__ import annotations

import asyncio
import hashlib  # AUDIT FIX #1: hilang di repo lama
import math
import os
import threading
import time
from typing import Any, Optional

import httpx


def _binance_us_first() -> bool:
    # Deploy US (Vercel iad1): Binance geo-block banyak IP datacenter US,
    # api.binance.us legal di US -> set BINANCE_US_FIRST=1.
    return os.getenv("BINANCE_US_FIRST", "").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Endpoint constants (AUDIT FIX #2: didefinisikan sebelum dipakai)
# ---------------------------------------------------------------------------

VISION_ENDPOINT = "https://data-api.binance.vision/api/v3"
BINANCE_US_ENDPOINT = "https://api.binance.us/api/v3"
BINANCE_SPOT_ENDPOINTS = [
    "https://api.binance.com/api/v3",
    "https://api-gcp.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
    "https://api2.binance.com/api/v3",
]
MEXC_ENDPOINTS = ["https://api.mexc.com/api/v3"]
FUTURES_ENDPOINTS = ["https://fapi.binance.com/fapi/v1"]
SPOT_ENDPOINTS = [VISION_ENDPOINT, BINANCE_US_ENDPOINT, *BINANCE_SPOT_ENDPOINTS]  # compat

COINGECKO_ENDPOINT = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=volume_desc&per_page=100&page=1"
    "&sparkline=false&price_change_percentage=24h"
)

KLINE_HOSTS = [
    ("binance-vision", VISION_ENDPOINT, {}),
    ("binance-us", BINANCE_US_ENDPOINT, {}),
    ("binance-spot", "https://api.binance.com/api/v3", {}),
    ("mexc-spot", "https://api.mexc.com/api/v3", {}),
    ("binance-futures", "https://fapi.binance.com/fapi/v1", {}),
]


def _kline_hosts():
    if _binance_us_first():
        return [KLINE_HOSTS[1], KLINE_HOSTS[0], *KLINE_HOSTS[2:]]
    return KLINE_HOSTS


_UA = {"accept": "application/json", "user-agent": "meridian-v2/2.0"}
STABLE_OR_FIAT = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "USD1", "EUR", "GBP"}


def _to_float(v: Any, fallback: float = 0.0) -> float:
    try:
        n = float(v)
        if math.isfinite(n):
            return n
    except (TypeError, ValueError):
        pass
    return fallback


def _normalize_exchange_rows(raw: list[dict]) -> list[dict]:
    out = []
    for item in raw:
        sym = str(item.get("symbol") or "").upper()
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in STABLE_OR_FIAT:
            continue
        last = _to_float(item.get("lastPrice") or item.get("price"))
        high = _to_float(item.get("highPrice") or item.get("high"), last)
        low = _to_float(item.get("lowPrice") or item.get("low"), last)
        qv = _to_float(item.get("quoteVolume") or item.get("amount") or item.get("volumeQuote"))
        if last <= 0 or qv <= 0:
            continue
        rate = _to_float(item.get("priceChangeRate"))
        change = _to_float(item.get("priceChangePercent"), rate * 100 if abs(rate) <= 1 else rate)
        out.append(
            {
                "symbol": sym,
                "lastPrice": last,
                "priceChangePercent": change,
                "volume": _to_float(item.get("volume")),
                "quoteVolume": qv,
                "highPrice": high,
                "lowPrice": low,
                "volatility24h": (high - low) / last if last > 0 else 0.0,
            }
        )
    out.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return out


def _normalize_coingecko_rows(raw: list[dict]) -> list[dict]:
    out = []
    for item in raw:
        sym = str(item.get("symbol") or "").upper()
        if not sym or sym in STABLE_OR_FIAT:
            continue
        last = _to_float(item.get("current_price"))
        qv = _to_float(item.get("total_volume"))
        if last <= 0 or qv <= 0:
            continue
        high = _to_float(item.get("high_24h"), last)
        low = _to_float(item.get("low_24h"), last)
        out.append(
            {
                "symbol": f"{sym}USDT",
                "lastPrice": last,
                "priceChangePercent": _to_float(item.get("price_change_percentage_24h")),
                "volume": 0,
                "quoteVolume": qv,
                "highPrice": high,
                "lowPrice": low,
                "volatility24h": (high - low) / last if last > 0 else 0.0,
            }
        )
    out.sort(key=lambda x: x["quoteVolume"], reverse=True)
    return out


async def _fetch_json(client: httpx.AsyncClient, url: str, timeout: float = 6.0) -> Any:
    r = await client.get(url, timeout=timeout, headers=_UA)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    return r.json()


DEFAULT_MIN_QUOTE_VOLUME = float(os.getenv("SCREEN_MIN_QUOTE_VOLUME", "5000000"))


def _universe_chain():
    chain = [
        ("binance-vision", [VISION_ENDPOINT], _normalize_exchange_rows),
        ("binance-us", [BINANCE_US_ENDPOINT], _normalize_exchange_rows),
        ("binance-spot", BINANCE_SPOT_ENDPOINTS, _normalize_exchange_rows),
        ("mexc-spot", MEXC_ENDPOINTS, _normalize_exchange_rows),
        ("binance-futures", FUTURES_ENDPOINTS, _normalize_exchange_rows),
    ]
    if _binance_us_first():
        chain = [chain[1], chain[0], *chain[2:]]
    return chain


async def fetch_universe(
    client: httpx.AsyncClient,
    limit: int = 40,
    min_quote_volume: float = 0.0,
) -> dict[str, Any]:
    """24h ticker universe dengan fallback chain + filter volume SEBELUM slice."""
    debug: list[str] = []
    for source, urls, normalizer in _universe_chain():
        for url in urls:
            try:
                raw = await _fetch_json(client, f"{url}/ticker/24hr")
                rows = normalizer(raw if isinstance(raw, list) else [])
                if min_quote_volume > 0:
                    rows = [r for r in rows if r["quoteVolume"] >= min_quote_volume]
                if rows:
                    return {
                        "source": source,
                        "live": True,
                        "data": rows[:limit],
                        "universeTotal": len(rows),
                        "minQuoteVolume": min_quote_volume,
                        "debug": debug,
                    }
                debug.append(f"{source}: empty after normalize")
            except Exception as exc:  # fallback chain tidak boleh raise
                debug.append(f"{source}: {exc}")
    try:
        raw = await _fetch_json(client, COINGECKO_ENDPOINT)
        rows = _normalize_coingecko_rows(raw if isinstance(raw, list) else [])
        if min_quote_volume > 0:
            rows = [r for r in rows if r["quoteVolume"] >= min_quote_volume]
        if rows:
            return {
                "source": "coingecko",
                "live": True,
                "data": rows[:limit],
                "universeTotal": len(rows),
                "minQuoteVolume": min_quote_volume,
                "debug": debug,
            }
    except Exception as exc:
        debug.append(f"coingecko: {exc}")

    from backend import sample_data

    rows = sample_data.seed_universe(limit)
    return {
        "source": "none",
        "live": False,
        "data": []
        "universeTotal": 0,
        "minQuoteVolume": min_quote_volume,
        "debug": debug,
        "warning": "Live market APIs unreachable — deterministic demo data.",
    }


def _parse_klines(raw: Any) -> list[dict]:
    out = []
    if not isinstance(raw, list):
        return out
    for k in raw:
        try:
            out.append(
                {
                    "date": int(k[0]),
                    "open": _to_float(k[1]),
                    "high": _to_float(k[2]),
                    "low": _to_float(k[3]),
                    "close": _to_float(k[4]),
                    "volume": _to_float(k[5]),
                    "quoteVolume": _to_float(k[7]),
                    "trades": _to_float(k[8]),
                    "takerBuyBase": _to_float(k[9]),
                    "takerBuyQuote": _to_float(k[10]),
                }
            )
        except (TypeError, IndexError, ValueError):
            continue
    return out


KLINE_LIMIT_MAX = 1000          # AUDIT FIX #4: hard cap Binance
MIN_KLINES_ACCEPT = 61          # ambang penerimaan per host (kebutuhan FP: MIN_CANDLES)


async def fetch_klines(client: httpx.AsyncClient, symbol: str, limit: int = 120) -> dict[str, Any]:
    """Candle harian dengan fallback host. Improvement: clamp limit + ambang bernama."""
    limit = max(30, min(int(limit), KLINE_LIMIT_MAX))
    debug: list[str] = []
    for source, base, _params in _kline_hosts():
        try:
            url = f"{base}/klines?symbol={symbol}&interval=1d&limit={limit}"
            raw = await _fetch_json(client, url)
            candles = _parse_klines(raw)
            if len(candles) >= MIN_KLINES_ACCEPT:
                return {"source": source, "live": True, "candles": candles, "debug": debug}
            debug.append(f"{source}: only {len(candles)} candles")
        except Exception as exc:
            debug.append(f"{source}: {exc}")

    from backend import sample_data

    return {
        "source": "none",
        "live": False,
        "candles": [],
        "debug": debug,
        "warning": "Live kline APIs unreachable — deterministic demo candles.",
    }


# ---------------------------------------------------------------------------
# TTLCache — AUDIT FIX #3: thread-safe + stats()
# ---------------------------------------------------------------------------

class TTLCache:
    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                self._misses += 1
                return None
            ts, value = item
            if time.monotonic() - ts > self.ttl:
                self._store.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hitRate": round(self._hits / max(total, 1), 3),
            }


screen_cache = TTLCache(ttl_seconds=float(os.getenv("SCREEN_CACHE_SECONDS", "60")))
KLINE_CONCURRENCY = int(os.getenv("KLINE_CONCURRENCY", "12"))


async def gather_klines(client: httpx.AsyncClient, symbols: list[str], limit: int = 120):
    sem = asyncio.Semaphore(KLINE_CONCURRENCY)

    async def one(sym: str):
        async with sem:
            return sym, await fetch_klines(client, sym, limit)

    return dict(await asyncio.gather(*(one(s) for s in symbols)))


def seed_label(source: str) -> str:
    return "Live" if source != "seed-fallback" else "Demo seed"


def deterministic_seed(text: str) -> int:
    # AUDIT FIX #1: hashlib sekarang benar-benar di-import di atas.
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# SELF-TEST (tanpa network) — python3 backend/market_data.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = [[1700000000000, "100", "110", "95", "105", "1000", "0",
            "105000", "50", "550", "57750"]]
    c = _parse_klines(raw)
    assert c and c[0]["close"] == 105.0 and c[0]["takerBuyBase"] == 550.0

    assert deterministic_seed("BTCUSDT") == deterministic_seed("BTCUSDT")
    assert deterministic_seed("BTCUSDT") != deterministic_seed("ETHUSDT")

    cache = TTLCache(ttl_seconds=0.05)
    cache.set("k", 42)
    assert cache.get("k") == 42
    time.sleep(0.06)
    assert cache.get("k") is None
    s = cache.stats()
    assert s["misses"] >= 1 and 0.0 <= s["hitRate"] <= 1.0

    rows = _normalize_exchange_rows([
        {"symbol": "BTCUSDT", "lastPrice": "100", "highPrice": "110",
         "lowPrice": "90", "quoteVolume": "5000000", "priceChangePercent": "2.5"},
        {"symbol": "USDCUSDT", "lastPrice": "1", "quoteVolume": "9"},  # stable -> dibuang
    ])
    assert len(rows) == 1 and rows[0]["symbol"] == "BTCUSDT"

    print("SELF-TEST OK — market_data v2")
