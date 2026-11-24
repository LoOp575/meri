"""
Meridian — Server v2 (Flask, STATELESS, Vercel-safe)
====================================================
Desain: server TIDAK menyimpan state antar-request (Vercel ephemeral).
Client (localStorage) kirim neuronState & memoryState di body,
server balikin state terbaru untuk disimpan ulang.

Data pasar: Binance public API (tanpa API key), cache in-memory TTL 5 menit.
Jalankan lokal:  python3 server.py   →  http://localhost:3000
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests
from flask import Flask, jsonify, request

# --- Import modul lokal (nama file persis seperti di repo) -------------------
import Screner
from neuron import Neuron, build_dendrites
from neuron_memory import NeuronMemory
from neuron_logistics import propose
from wieght_lab import evaluate

app = Flask(__name__)

BINANCE = "https://api.binance.com"
CACHE_TTL = 300  # detik
_cache: dict[str, tuple[float, Any]] = {}


# ---------------------------------------------------------------------------
# Data pasar (Binance public + cache TTL)
# ---------------------------------------------------------------------------

def _cache_get(key: str) -> Optional[Any]:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    return None

def _cache_put(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)

def _fetch_klines(symbol: str, limit: int = 120) -> list[list]:
    r = requests.get(
        f"{BINANCE}/api/v3/klines",
        params={"symbol": symbol.upper(), "interval": "1d", "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()

def _to_candles(raw: list[list]) -> list[dict[str, Any]]:
    """Format Binance klines -> format candle yang dimakan Screner.py."""
    out = []
    for k in raw:
        out.append({
            "date": str(k[0]), "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
            "quoteVolume": float(k[7]), "trades": int(k[8]),
            "takerBuyBase": float(k[9]), "takerBuyQuote": float(k[10]),
        })
    return out

def _btc_regime() -> float:
    """FIX #7: regime = sat(return 30d BTC), netral 0.0 kalau gagal."""
    try:
        key = "regime:btc"
        hit = _cache_get(key)
        if hit is not None:
            return hit
        raw = _fetch_klines("BTCUSDT", limit=40)
        closes = [float(k[4]) for k in raw]
        regime = 0.0
        if len(closes) >= 31 and closes[-31] > 0:
            r = closes[-1] / closes[-31] - 1.0
            regime = max(-1.0, min(1.0, r / (1.0 + abs(r))))  # sat
        _cache_put(key, regime)
        return regime
    except Exception:
        return 0.0

def _screen_v2(symbol: str) -> dict[str, Any]:
    """Fetch candle -> compute_features_v2 (signature asli Screner.py)."""
    key = f"candles:{symbol.upper()}"
    candles = _cache_get(key)
    if candles is None:
        candles = _to_candles(_fetch_klines(symbol, limit=120))
        _cache_put(key, candles)
    return Screner.compute_features_v2(candles, regime=_btc_regime())


def _err(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


# ---------------------------------------------------------------------------
# 1) Health + spec screener
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"ok": True, "service": "meridian-v2", "stateless": True})

@app.route("/api/screen-v2/spec", methods=["GET"])
def api_screen_spec():
    return jsonify(Screner.formula_spec_v2())

# ---------------------------------------------------------------------------
# 2) Screener v2 — GET /api/screen-v2?symbol=BTCUSDT
# ---------------------------------------------------------------------------

@app.route("/api/screen-v2", methods=["GET"])
def api_screen_v2():
    symbol = (request.args.get("symbol") or "").upper().strip()
    if not symbol:
        return _err("parameter ?symbol= wajib, contoh: ?symbol=BTCUSDT")
    try:
        result = _screen_v2(symbol)
    except requests.HTTPError as e:
        return _err(f"symbol tidak valid / Binance error: {e}", 404)
    except Exception as e:
        return _err(f"screening gagal: {e}", 500)
    return jsonify({"ok": True, "symbol": symbol, "v2": result})

# ---------------------------------------------------------------------------
# 3) Neuron predict — POST {symbol?, dendrites?, neuronState?, memoryState?}
#    Prioritas: dendrites dari client > hitung screen dari symbol > netral
# ---------------------------------------------------------------------------

@app.route("/api/neuron/predict", methods=["POST"])
def api_neuron_predict():
    b = request.get_json(silent=True) or {}
    neuron = Neuron.from_dict(b.get("neuronState"))
    memory = NeuronMemory.from_dict(b.get("memoryState"))

    dendrites = b.get("dendrites")
    screen = None
    if not dendrites and b.get("symbol"):
        try:
            screen = _screen_v2(str(b["symbol"]))
        except Exception as e:
            return _err(f"gagal ambil data pasar: {e}", 502)
    dendrites = build_dendrites(screen=screen) if screen else (dendrites or {})

    pred = neuron.predict(dendrites)
    ep = memory.record_episode(str(b.get("symbol") or "?"), dendrites,
                               pred["y"], pred["verdict"])
    return jsonify({
        "ok": True,
        "prediction": pred,
        "episode": ep,                      # client simpan: id + dendrites
        "trust": memory.pattern_trust(dendrites),
        "neuronState": neuron.to_dict(),    # simpan balik ke localStorage
        "memoryState": memory.to_dict(),
    })

# ---------------------------------------------------------------------------
# 4) Tutup episode + belajar online — POST {episode, realizedReturn,
#    neuronState?, memoryState?}   (episode = objek dari route predict)
# ---------------------------------------------------------------------------

@app.route("/api/neuron/episode/close", methods=["POST"])
def api_neuron_episode_close():
    b = request.get_json(silent=True) or {}
    ep_in = b.get("episode") or {}
    if not ep_in.get("id") or b.get("realizedReturn") is None:
        return _err("body wajib: {episode: {id, dendrites,...}, realizedReturn}")
    memory = NeuronMemory.from_dict(b.get("memoryState"))
    try:
        ep = memory.close_episode(str(ep_in["id"]), float(b["realizedReturn"]))
    except KeyError as e:
        return _err(str(e), 404)

    learn, neuron_state = None, b.get("neuronState")
    if neuron_state and ep_in.get("dendrites"):
        neuron = Neuron.from_dict(neuron_state)
        learn = neuron.update(ep_in["dendrites"], float(b["realizedReturn"]))
        neuron_state = neuron.to_dict()

    return jsonify({
        "ok": True, "episode": ep, "learning": learn,
        "stats": memory.stats(),
        "neuronState": neuron_state,
        "memoryState": memory.to_dict(),
    })

# ---------------------------------------------------------------------------
# 5) Guru offline (refit) — POST {memoryState, neuronState?}
#    Hasil = USULAN ACCEPT/REJECT — TIDAK otomatis dipasang.
# ---------------------------------------------------------------------------

@app.route("/api/neuron/refit", methods=["POST"])
def api_neuron_refit():
    b = request.get_json(silent=True) or {}
    if not b.get("memoryState"):
        return _err("memoryState wajib (dari localStorage)")
    memory = NeuronMemory.from_dict(b["memoryState"])
    neuron = Neuron.from_dict(b.get("neuronState"))
    proposal = propose(neuron.weights, memory.closed_episodes())
    return jsonify({"ok": True, "proposal": proposal})

# ---------------------------------------------------------------------------
# 6) Weight lab — POST {memoryState, candidateWeights?, baselineWeights?}
# ---------------------------------------------------------------------------

@app.route("/api/neuron/validate", methods=["POST"])
def api_neuron_validate():
    b = request.get_json(silent=True) or {}
    if not b.get("memoryState"):
        return _err("memoryState wajib")
    memory = NeuronMemory.from_dict(b["memoryState"])
    candidate = b.get("candidateWeights") or Neuron.from_dict(b.get("neuronState")).weights
    result = evaluate(candidate, memory.episodes(),
                      baseline_weights=b.get("baselineWeights"))
    return jsonify({"ok": True, "evaluation": result})

# ---------------------------------------------------------------------------
# 7) Statistik — POST {memoryState?}
# ---------------------------------------------------------------------------

@app.route("/api/neuron/stats", methods=["POST"])
def api_neuron_stats():
    b = request.get_json(silent=True) or {}
    memory = NeuronMemory.from_dict(b.get("memoryState"))
    return jsonify({"ok": True, "stats": memory.stats(),
                    "trustedPatterns": memory.trusted_patterns()})


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
