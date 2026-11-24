"""
Meridian v2 — CryptoRank Tokenomics Layer (REWRITE TERBUKA).
============================================================
AUDIT repo lama: cryptorank.py lama hanya bootstrap zlib+base64 yang
exec() blob terenkripsi -> tak terbaca, tak bisa diaudit. DIBUANG,
diganti implementasi terbuka penuh ini.

Fungsi: data tokenomics untuk panel detail koin (klik di screener):
 - resolve symbol -> id        : GET /v2/currencies/search (1 credit)
 - detail pasar & supply       : GET /v2/currencies/{id} (1 credit)
 - unlock/locked/allocations   : GET /v2/currencies/{id}/vesting (1 credit, plan Pro)

Aturan proyek:
 - TANPA demo/fake data. Key hilang / plan kurang / network mati
   -> {"available": False, "reason": ...} yang jujur.
 - Normalisasi murni (pure) supaya bisa self-test tanpa network.
 - Cache TTL hemat credit (vesting jarang berubah).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import httpx

from backend.market_data import TTLCache, _fetch_json

CRYPTORANK_API_BASE = os.getenv("CRYPTORANK_API_BASE", "https://api.cryptorank.io/v2")

_id_cache = TTLCache(ttl_seconds=86_400.0)   # symbol -> id (stabil, cache 1 hari)
_bundle_cache = TTLCache(
    ttl_seconds=float(os.getenv("CRYPTORANK_CACHE_SECONDS", "3600"))
)


class CryptorankUnavailable(Exception):
    """Dilempar/dilaporkan jujur saat data tidak bisa diambil — TANPA fallback palsu."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _headers() -> dict[str, str]:
    key = os.getenv("CRYPTORANK_API_KEY", "").strip()
    return {"X-Api-Key": key} if key else {}


def _f(v: Any, fallback: Optional[float] = None) -> Optional[float]:
    try:
        n = float(v)
        return n if math.isfinite(n) else fallback
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Normalisasi MURNI (pure) — bisa di-self-test tanpa network
# ---------------------------------------------------------------------------

def pick_best_symbol_match(items: list[dict], symbol: str) -> Optional[dict]:
    """Pilih kandidat terbaik: simbol persis + lifeCycle 'traded' + rank terkecil."""
    sym = symbol.upper().strip()
    scored = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it_sym = str(it.get("symbol") or "").upper()
        if it_sym != sym:
            continue
        exact_traded = 0 if it.get("lifeCycle") == "traded" else 1
        rank = _f(it.get("rank"), 999_999.0) or 999_999.0
        scored.append((exact_traded, rank, it))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def normalize_detail(raw: dict) -> dict:
    """/v2/currencies/{id} -> payload panel yang bersih."""
    d = raw if isinstance(raw, dict) else {}
    return {
        "id": d.get("id"),
        "key": d.get("key"),
        "symbol": str(d.get("symbol") or "").upper(),
        "name": d.get("name"),
        "rank": _f(d.get("rank")),
        "price": _f(d.get("price")),
        "marketCap": _f(d.get("marketCap")),
        "fullyDilutedValuation": _f(d.get("fullyDilutedValuation")),
        "totalSupply": _f(d.get("totalSupply")),
        "maxSupply": _f(d.get("maxSupply")),
        "circulatingSupply": _f(d.get("circulatingSupply")),
        "hasVesting": bool(d.get("hasVesting")),
        "hasNextUnlock": bool(d.get("hasNextUnlock")),
        "hasFundingRounds": bool(d.get("hasFundingRounds")),
        "images": d.get("images") or {},
    }


def normalize_vesting(raw: dict) -> dict:
    """/v2/currencies/{id}/vesting -> unlock & alokasi yang bersih + turunan %."""
    d = raw if isinstance(raw, dict) else {}
    total_supply = _f(d.get("totalSupply")) or 0.0
    circ = _f(d.get("circulatingSupply")) or 0.0
    locked = _f(d.get("totalLocked")) or 0.0
    unlocked = _f(d.get("totalUnlocked")) or 0.0
    nu = d.get("nextUnlock") if isinstance(d.get("nextUnlock"), dict) else {}
    nu_tokens = _f(nu.get("tokens")) or 0.0

    allocations = []
    for a in d.get("allocations") or []:
        if not isinstance(a, dict):
            continue
        alloc_supply = _f(a.get("allocationOfSupply"))
        allocations.append(
            {
                "name": a.get("name"),
                "group": a.get("group"),
                "pctOfSupply": alloc_supply,
                "tokens": _f(a.get("tokens")),
                "unlockedTokens": _f(a.get("unlockedTokens")),
                "lockedTokens": _f(a.get("lockedTokens")),
                "tgeUnlockPercent": _f(a.get("tgeUnlockPercent")),
                "unlockFrequency": (
                    f"{a.get('unlockFrequencyValue', '?')} {a.get('unlockFrequencyType', '')}".strip()
                    if a.get("unlockFrequencyValue")
                    else None
                ),
                "vestingStart": a.get("vestingStartDate"),
                "vestingEnd": a.get("vestingEndDate"),
                "nextUnlockDate": (a.get("nextUnlock") or {}).get("date")
                if isinstance(a.get("nextUnlock"), dict)
                else None,
            }
        )

    return {
        "available": True,
        "vestingStart": d.get("vestingStartDate"),
        "vestingEnd": d.get("vestingEndDate"),
        "listingDate": d.get("listingDate"),
        "totalLocked": locked,
        "totalUnlocked": unlocked,
        "totalUntracked": _f(d.get("totalUntracked")) or 0.0,
        "supplyLockedPct": round(locked / total_supply * 100.0, 2) if total_supply > 0 else None,
        "nextUnlock": {
            "date": nu.get("date"),
            "tokens": nu_tokens,
            "pctOfCirculating": round(nu_tokens / circ * 100.0, 2) if circ > 0 else None,
        },
        "allocations": allocations,
    }


# ---------------------------------------------------------------------------
# Fetch layer (async, dengan cache & kejujuran error)
# ---------------------------------------------------------------------------

async def resolve_currency_id(client: httpx.AsyncClient, symbol: str) -> Optional[int]:
    sym = symbol.upper().strip()
    cached = _id_cache.get(sym)
    if cached is not None:
        return cached
    if not _headers():
        raise CryptorankUnavailable(
            "no-api-key", "CRYPTORANK_API_KEY belum diset — panel tokenomics nonaktif."
        )
    url = f"{CRYPTORANK_API_BASE}/currencies/search?symbol={sym}&include=lifeCycle"
    raw = await _fetch_json(client, url)
    best = pick_best_symbol_match(raw.get("data") or [], sym)
    if best is None:
        return None
    cid = int(best["id"])
    _id_cache.set(sym, cid)
    return cid


async def fetch_vesting(client: httpx.AsyncClient, cid: int) -> dict[str, Any]:
    """Vesting/unlock. Kalau plan tidak cukup (HTTP 403) -> jujur available=False."""
    url = f"{CRYPTORANK_API_BASE}/currencies/{cid}/vesting"
    try:
        raw = await _fetch_json(client, url)
    except RuntimeError as exc:
        if "HTTP 403" in str(exc):
            return {
                "available": False,
                "reason": "plan-required",
                "hint": "Endpoint vesting butuh plan Pro CryptoRank.",
            }
        raise
    return normalize_vesting(raw.get("data") or {})


async def fetch_tokenomics_bundle(
    client: httpx.AsyncClient, symbol: str, include_funding: bool = False
) -> dict[str, Any]:
    """Bundle panel detail: identitas + supply + unlock/locked/alokasi.

    TANPA demo fallback. Semua kegagalan dilaporkan jujur lewat
    available=False + reason, sehingga UI bisa menampilkan status apa adanya.
    """
    sym = symbol.upper().strip()
    cached = _bundle_cache.get(sym)
    if cached is not None:
        return cached

    out: dict[str, Any] = {
        "symbol": sym,
        "available": False,
        "sources": {"market": "cryptorank", "vesting": "cryptorank-vesting"},
        "fetchedAt": int(time.time() * 1000),
    }
    try:
        cid = await resolve_currency_id(client, sym)
    except CryptorankUnavailable as exc:
        out.update({"reason": exc.reason, "message": exc.message})
        return out
    except Exception as exc:  # network/HTTP lain
        out.update({"reason": "network", "message": str(exc)[:200]})
        return out

    if cid is None:
        out.update({"reason": "not-found", "message": f"Simbol {sym} tidak ditemukan di CryptoRank."})
        return out

    try:
        detail_raw = await _fetch_json(client, f"{CRYPTORANK_API_BASE}/currencies/{cid}")
        out.update(normalize_detail(detail_raw.get("data") or {}))
    except Exception as exc:
        out.update({"reason": "network", "message": f"detail: {exc}"[:200]})
        return out

    try:
        out["vesting"] = await fetch_vesting(client, cid)
    except Exception as exc:
        out["vesting"] = {"available": False, "reason": "network", "message": str(exc)[:200]}

    if include_funding:
        try:
            fr = await _fetch_json(client, f"{CRYPTORANK_API_BASE}/currencies/{cid}/funding-rounds")
            data = fr.get("data") or {}
            out["funding"] = {
                "available": True,
                "totalRaise": _f(data.get("totalRaise")),
                "lastValuation": _f(data.get("lastValuation")),
                "rounds": [
                    {
                        "stage": r.get("stage"),
                        "date": r.get("announcementDate"),
                        "raise": _f(r.get("raise")),
                        "valuation": _f(r.get("valuation")),
                        "leadFunds": [
                            f.get("name") for f in (r.get("funds") or []) if f.get("isLead")
                        ],
                    }
                    for r in (data.get("fundingRounds") or [])[:8]
                ],
            }
        except RuntimeError as exc:
            out["funding"] = (
                {"available": False, "reason": "plan-required"}
                if "HTTP 403" in str(exc)
                else {"available": False, "reason": "network", "message": str(exc)[:200]}
            )

    out["available"] = True
    _bundle_cache.set(sym, out)
    return out


# ---------------------------------------------------------------------------
# SELF-TEST (tanpa network) — python3 backend/cryptorank.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # pick_best_symbol_match
    items = [
        {"id": 9, "symbol": "IDOL", "name": "Idol lain", "lifeCycle": "inactive", "rank": 400},
        {"id": 7, "symbol": "IDOL", "name": "IDOL Token", "lifeCycle": "traded", "rank": 120},
        {"id": 5, "symbol": "IDOLX", "name": "Bukan", "lifeCycle": "traded", "rank": 10},
    ]
    best = pick_best_symbol_match(items, "idol")
    assert best and best["id"] == 7, "harus pilih IDOL traded rank terkecil"
    assert pick_best_symbol_match(items, "NOPE") is None

    # normalize_detail
    det = normalize_detail(
        {"id": 7, "symbol": "idol", "name": "IDOL Token", "rank": "120",
         "price": "0.42", "marketCap": "42000000", "totalSupply": "1000000000",
         "circulatingSupply": "300000000", "hasVesting": True}
    )
    assert det["symbol"] == "IDOL" and det["rank"] == 120.0 and det["hasVesting"] is True

    # normalize_vesting + turunan %
    ves = normalize_vesting(
        {"totalSupply": "1000000000", "circulatingSupply": "300000000",
         "totalLocked": "500000000", "totalUnlocked": "300000000",
         "nextUnlock": {"date": 1780272000000, "tokens": "25000000"},
         "allocations": [
             {"name": "Team", "allocationOfSupply": "20", "lockedTokens": "150000000",
              "unlockedTokens": "50000000", "tgeUnlockPercent": "5",
              "unlockFrequencyValue": "3", "unlockFrequencyType": "month",
              "nextUnlock": {"date": 1780272000000}},
         ]}
    )
    assert ves["supplyLockedPct"] == 50.0
    assert ves["nextUnlock"]["pctOfCirculating"] == round(25_000_000 / 300_000_000 * 100, 2)
    a = ves["allocations"][0]
    assert a["unlockFrequency"] == "3 month" and a["lockedTokens"] == 150_000_000.0

    # Kejujuran: tanpa API key -> resolve harus raise CryptorankUnavailable
    os.environ.pop("CRYPTORANK_API_KEY", None)
    import asyncio

    async def _no_key():
        async with httpx.AsyncClient() as c:
            try:
                await resolve_currency_id(c, "BTC")
                return "FAIL"
            except CryptorankUnavailable as e:
                return e.reason

    assert asyncio.run(_no_key()) == "no-api-key"

    print("SELF-TEST OK — cryptorank v2")
