"""
Meridian — Neuron Core v1 (otak online adaptif, modul ISOLASI)
==============================================================

Neuron tunggal ala biologis:
- 10 dendrite (x1..x10), fitur ternormalisasi [-1, +1]
- Aktivasi tanh -> y in [-1, +1] -> verdict LONG/WATCH/NEUTRAL/AVOID
- Belajar online via RMSProp (adaptif per-dendrite, tahan ganti rezim pasar)
- Surprise Gate: prediksi salah arah > 1.5% -> learning rate x2
  (otak beneran: kesalahan besar = pelajaran penting)
- Serialisasi JSON penuh -> aman untuk filesystem ephemeral Vercel
  (state nanti disimpan localStorage / API; Supabase menyusul)

Dendrite map (bobot PROVISIONAL — production trust hanya setelah lab/refit):
  x1  screenSignal  = 2*p_acc - 1           (dari screening_v2)
  x2  demandQuality = q_dr/50 - 1
  x3  stability     = 1 - 2*instability/100
  x4  fpStable      = 2*pStable - 1         (dari fokker_planck)
  x5  fpTail        = 1 - 2*pTailCrash
  x6  fpLiveDir     = arah intraday [-1,+1] (dari fp_live)
  x7  rvol          = sat(rvol - 1)
  x8  unlockGate    = [-1,+1] (unlock besar mendekat -> negatif)
  x9  vwapDist      = sat(close_vs_vwap20)
  x10 bias          = 1.0

Murni komputasi, tanpa I/O — deterministik, bisa unit-test.
Self-test:  python3 backend/neuron_core.py
"""

from __future__ import annotations

from math import sqrt, tanh
from typing import Any, Optional

NEURON_VERSION = "neuron-core-v1.0-provisional"

# --- Hyperparameter (dievaluasi, tetap eksplisit) --------------------------
LR_BASE = 0.05            # learning rate dasar
RMSPROP_DECAY = 0.9       # memori rata-rata kuadrat gradien
RMSPROP_EPS = 1e-8
SURPRISE_THRESHOLD = 0.015  # 1.5% — salah arah melewati ini -> surprise
SURPRISE_LR_MULT = 2.0      # surprise gate: lr x2
TARGET_SCALE = 0.05         # realized +5% -> target +1.0 ; -5% -> -1.0
W_CAP = 1.5                 # pengaman: bobot tidak boleh lepas kendali

LONG_T, WATCH_T, AVOID_T = 0.30, 0.05, -0.05

# --- Bobot awal PROVISIONAL -------------------------------------------------
INIT_WEIGHTS: dict[str, float] = {
    "screenSignal": 0.30, "demandQuality": 0.20, "stability": 0.20,
    "fpStable": 0.15, "fpTail": 0.15, "fpLiveDir": 0.10,
    "rvol": 0.10, "unlockGate": 0.10, "vwapDist": 0.15, "bias": 0.0,
}
DENDRITE_KEYS = [k for k in INIT_WEIGHTS if k != "bias"]

EPS = 1e-9

def sat(x: float) -> float:
    return x / (1.0 + abs(x))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _target(realized_return: float) -> float:
    """Realized return di-scale ke [-1, +1]: +5% -> +1, -5% -> -1."""
    return clamp(realized_return / TARGET_SCALE, -1.0, 1.0)

# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------

class Neuron:
    """Neuron tunggal: predict (forward) + update (RMSProp) + serialisasi JSON."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> None:
        self.weights = {k: float(v) for k, v in INIT_WEIGHTS.items()}
        self.cache = {k: 0.0 for k in self.weights}  # RMSProp memory
        self.n_updates = 0
        if weights:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = float(v)
        if state:
            self._load(state)

    # -- forward ------------------------------------------------------------
    def _vector(self, features: dict[str, float]) -> dict[str, float]:
        """Fitur mentah -> dendrite ternormalisasi [-1,+1]; hilang -> 0 (netral)."""
        x = {k: clamp(float(features.get(k, 0.0) or 0.0), -1.0, 1.0)
             for k in DENDRITE_KEYS}
        x["bias"] = 1.0
        return x

    def _forward(self, x: dict[str, float]) -> float:
        z = sum(self.weights[k] * x[k] for k in self.weights)
        return tanh(z)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        x = self._vector(features)
        y = self._forward(x)
        if y >= LONG_T:
            verdict = "LONG"
        elif y >= WATCH_T:
            verdict = "WATCH"
        elif y <= AVOID_T:
            verdict = "AVOID"
        else:
            verdict = "NEUTRAL"
        return {
            "ok": True,
            "model": NEURON_VERSION,
            "provisional": True,
            "y": round(y, 4),
            "verdict": verdict,
            "confidence": round(abs(y), 4),
            "dendrites": {k: round(v, 4) for k, v in x.items()},
            "contributions": {k: round(self.weights[k] * x[k], 4) for k in x},
            "nUpdates": self.n_updates,
        }

    # -- learning (RMSProp + surprise gate) ----------------------------------
    def update(self, features: dict[str, float], realized_return: float) -> dict[str, Any]:
        x = self._vector(features)
        y = self._forward(x)
        t = _target(realized_return)
        err = t - y

        # Surprise Gate: prediksi salah ARAH melewati 1.5% -> lr x2
        surprise = (y > 0.0 and realized_return < -SURPRISE_THRESHOLD) or (
            y < 0.0 and realized_return > SURPRISE_THRESHOLD
        )
        lr = LR_BASE * (SURPRISE_LR_MULT if surprise else 1.0)

        # dL/dw_i utk loss 0.5*(t-y)^2 dengan aktivasi tanh
        for k in self.weights:
            g = (y - t) * x[k] * (1.0 - y * y)
            self.cache[k] = RMSPROP_DECAY * self.cache[k] + (1.0 - RMSPROP_DECAY) * g * g
            step = lr * g / (sqrt(self.cache[k]) + RMSPROP_EPS)
            self.weights[k] = clamp(self.weights[k] - step, -W_CAP, W_CAP)

        self.n_updates += 1
        return {"yBefore": round(y, 4), "target": round(t, 4),
                "error": round(err, 4), "surprise": surprise,
                "lr": lr, "nUpdates": self.n_updates}

    # -- serialisasi (Vercel ephemeral-safe) ----------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": NEURON_VERSION, "weights": dict(self.weights),
                "cache": dict(self.cache), "nUpdates": self.n_updates}

    def _load(self, state: dict[str, Any]) -> None:
        w = state.get("weights") or {}
        c = state.get("cache") or {}
        for k in self.weights:
            if k in w:
                self.weights[k] = clamp(float(w[k]), -W_CAP, W_CAP)
            if k in c:
                self.cache[k] = max(0.0, float(c[k]))
        self.n_updates = int(state.get("nUpdates", 0))

    @classmethod
    def from_dict(cls, state: Optional[dict[str, Any]]) -> "Neuron":
        return cls(state=state)

# ---------------------------------------------------------------------------
# Helper wiring: hasil modul lain -> dendrite (dipakai route di Tahap B/C)
# ---------------------------------------------------------------------------

def build_dendrites(
    screen: Optional[dict[str, Any]] = None,
    fp: Optional[dict[str, Any]] = None,
    fp_live_dir: Optional[float] = None,
    unlock: Optional[float] = None,
) -> dict[str, float]:
    """Aman terhadap input None/parcial — dendrite tak tersedia = 0.0 (netral)."""
    d = {k: 0.0 for k in DENDRITE_KEYS}
    if screen and screen.get("ok"):
        sc = screen.get("scores") or {}
        f = screen.get("features") or {}
        d["screenSignal"] = clamp(2.0 * float(sc.get("pAcceleration") or 0.0) - 1.0, -1.0, 1.0)
        d["demandQuality"] = clamp(float(sc.get("demandQuality") or 0.0) / 50.0 - 1.0, -1.0, 1.0)
        d["stability"] = clamp(1.0 - 2.0 * float(sc.get("instability") or 0.0) / 100.0, -1.0, 1.0)
        d["rvol"] = sat(float(f.get("rvol", 1.0) or 1.0) - 1.0)
        d["vwapDist"] = sat(float(f.get("closeVsVwap20", 0.0) or 0.0))
    if fp and fp.get("ok"):
        m = fp.get("metrics") or {}
        d["fpStable"] = clamp(2.0 * float(m.get("pStable", 0.5) or 0.5) - 1.0, -1.0, 1.0)
        d["fpTail"] = clamp(1.0 - 2.0 * float(m.get("pTailCrash", 0.5) or 0.5), -1.0, 1.0)
    if fp_live_dir is not None:
        d["fpLiveDir"] = clamp(float(fp_live_dir), -1.0, 1.0)
    if unlock is not None:
        d["unlockGate"] = clamp(float(unlock), -1.0, 1.0)
    return d

# ---------------------------------------------------------------------------
# Self-test (deterministik, tanpa I/O)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # 1) Fitur netral -> y=0 -> NEUTRAL
    n = Neuron()
    r = n.predict({})
    assert r["verdict"] == "NEUTRAL" and abs(r["y"]) < 1e-9, f"1: {r}"

    # 2) Belajar rezim bullish: 100 episode +3% -> w1 naik, verdict LONG
    feats = {"screenSignal": 0.8}
    for _ in range(100):
        last = n.update(feats, 0.03)
    y2 = n.predict(feats)
    assert n.weights["screenSignal"] > 0.30, f"2a: w1={n.weights['screenSignal']}"
    assert y2["y"] > 0.45 and y2["verdict"] == "LONG", f"2b: {y2}"

    # 3) REGIME SWITCH: rezim berbalik -3% -> bobot membalik arah, verdict AVOID
    for _ in range(100):
        n.update(feats, -0.03)
    y3 = n.predict(feats)
    assert n.weights["screenSignal"] < 0.0, f"3a: w1={n.weights['screenSignal']}"
    assert y3["y"] < -0.30 and y3["verdict"] == "AVOID", f"3b: {y3}"

    # 4) Surprise Gate: prediksi positif tapi realized -3% -> lr x2
    ns = Neuron()
    ns.predict({"screenSignal": 0.5})
    res = ns.update({"screenSignal": 0.5}, -0.03)
    assert res["surprise"] is True and res["lr"] == LR_BASE * SURPRISE_LR_MULT, f"4: {res}"

    # 5) Serialisasi roundtrip: state JSON -> neuron baru -> prediksi identik
    d = n.to_dict()
    n2 = Neuron.from_dict(d)
    p1, p2 = n.predict(feats), n2.predict(feats)
    assert abs(p1["y"] - p2["y"]) < 1e-12, "5: state tidak identik"

    # 6) Peta verdict: semua dendrite +1 -> LONG ; semua -1 -> AVOID ; lemah -> WATCH
    all_pos = {k: 1.0 for k in DENDRITE_KEYS}
    all_neg = {k: -1.0 for k in DENDRITE_KEYS}
    weak = {"screenSignal": 0.2}
    assert n.predict(all_pos)["verdict"] == "LONG", "6a"
    assert n.predict(all_neg)["verdict"] == "AVOID", "6b"
    assert Neuron().predict(weak)["verdict"] == "WATCH", "6c"

    # 7) build_dendrites toleran input kosong
    d0 = build_dendrites()
    assert all(abs(v) < 1e-9 for v in d0.values()), "7: default tidak netral"
    d1 = build_dendrites(screen={"ok": True, "scores": {"pAcceleration": 0.8,
                         "demandQuality": 70, "instability": 20}})
    assert abs(d1["screenSignal"] - 0.6) < 1e-9, "7b"

    print("SELF-TEST OK — neuron_core (RMSProp + surprise gate + regime switch + serialisasi)")

if __name__ == "__main__":
    _self_test()
