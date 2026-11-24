"""
Meridian — Weight Lab v1 (gerbang validasi bobot, modul ISOLASI)
================================================================

Tugas: MENILAI bobot kandidat terhadap data episode tertutup, sebelum
bobot dipasang ke production. Budaya audit:

  1. Gerbang MUTLAK (semua wajib lolos):
     - Brier score  <= 0.25   (kalibrasi probabilitas naik)
     - AUC          >= 0.60   (daya pisah arah)
     - Precision top-25%, Wilson LB >= 0.55 (konservatif ala neuron_memory)
  2. Gerbang RELATIF: jika baseline disediakan DAN baseline lolos gerbang
     mutlak, kandidat harus TIDAK LEBIH BURUK di semua metrik DAN lebih
     baik di minimal satu. "Sama saja" = REJECT.
  3. Hasil ACCEPT tetap berlabel PROVISIONAL — wajib monitoring live.

Definisi event: realizedReturn > 0 (naik). Probabilitas model:
  p_up = (tanh(w·x) + 1) / 2  — y in [-1,+1] dipetakan ke [0,1].

Murni komputasi, tanpa I/O — deterministik (tanpa shuffle acak).
Self-test:  python3 backend/weight_lab.py
"""

from __future__ import annotations

import json
from math import tanh
from typing import Any, Optional

WEIGHT_LAB_VERSION = "weight-lab-v1.0"

# --- Gerbang (dievaluasi, tetap eksplisit) -----------------------------------
MIN_SAMPLES = 50            # di bawah ini: tidak layak dinilai
BRIER_MAX = 0.25            # kalibrasi: 0.5 = tebakan acak penuh
AUC_MIN = 0.60              # daya pisah: 0.5 = acak, 1.0 = sempurna
PRECISION_TOP_FRAC = 0.25   # precision diukur di top-25% skor tertinggi
PRECISION_WILSON_MIN = 0.55 # Wilson LB minimal (konsisten neuron_memory)
Z95 = 1.96
TARGET_EPS = 1e-12

# SINKRON dengan neuron_core / neuron_logistic (disengaja duplikat agar
# modul tetap isolasi & standalone — ubah ketiganya bila berubah).
DENDRITE_KEYS = [
    "screenSignal", "demandQuality", "stability", "fpStable", "fpTail",
    "fpLiveDir", "rvol", "unlockGate", "vwapDist",
]

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _wilson(successes: int, n: int, z: float = Z95) -> float:
    """Wilson lower bound 95% — sama dengan neuron_memory."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5)
    return max(0.0, (centre - margin) / denom)

def _vector(dendrites: dict[str, float]) -> dict[str, float]:
    x = {k: _clamp(float(dendrites.get(k, 0.0) or 0.0), -1.0, 1.0)
         for k in DENDRITE_KEYS}
    x["bias"] = 1.0
    return x

def _y(weights: dict[str, float], x: dict[str, float]) -> float:
    return tanh(sum(weights[k] * x[k] for k in weights))

def _prob(weights: dict[str, float], x: dict[str, float]) -> float:
    return (_y(weights, x) + 1.0) / 2.0

# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------

def _brier(weights: dict[str, float],
           data: list[tuple[dict[str, float], int]]) -> Optional[float]:
    if not data:
        return None
    return sum((_prob(weights, x) - float(e)) ** 2 for x, e in data) / len(data)

def _auc(scores: list[float], labels: list[int]) -> Optional[float]:
    """AUC rank-based (Mann-Whitney), tie-aware, deterministik."""
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # tak terdefinisi jika semua label satu arah
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # rank 1-based, ties dirata-rata
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos = sum(ranks[i2] for i2 in range(n) if labels[i2] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def _precision_topk(weights: dict[str, float],
                    data: list[tuple[dict[str, float], int]]
                    ) -> tuple[int, int, float]:
    """Precision di top-K skor tertinggi + Wilson LB. Tie-break by index."""
    n = len(data)
    k = max(1, int(round(n * PRECISION_TOP_FRAC)))
    top = sorted(range(n), key=lambda i: (-_y(weights, data[i][0]), i))[:k]
    succ = sum(1 for i in top if data[i][1])
    return succ, k, _wilson(succ, k)

def _metrics(weights: dict[str, float],
             data: list[tuple[dict[str, float], int]]) -> dict[str, Any]:
    scores = [_y(weights, x) for x, _ in data]
    labels = [e for _, e in data]
    auc = _auc(scores, labels)
    succ, k, lb = _precision_topk(weights, data)
    brier = _brier(weights, data)
    gates = {
        "brierPass": brier is not None and brier <= BRIER_MAX,
        "aucPass": auc is not None and auc >= AUC_MIN,
        "precisionPass": lb >= PRECISION_WILSON_MIN,
    }
    return {"brier": None if brier is None else round(brier, 4),
            "auc": None if auc is None else round(auc, 4),
            "precision": round(succ / k, 4), "precisionK": k,
            "precisionWilsonLB": round(lb, 4),
            "gates": gates, "gatesPassed": all(gates.values())}

# ---------------------------------------------------------------------------
# Evaluasi + keputusan
# ---------------------------------------------------------------------------

def _build_data(episodes: list[dict[str, Any]]
                ) -> list[tuple[dict[str, float], int]]:
    """Episode CLOSED + return terukur + dendrite informatif -> (x, event)."""
    data: list[tuple[dict[str, float], int]] = []
    for e in episodes:
        if e.get("status") != "CLOSED" or e.get("realizedReturn") is None:
            continue
        d = e.get("dendrites") or {}
        if not any(abs(float(d.get(k, 0.0) or 0.0)) > 1e-9 for k in DENDRITE_KEYS):
            continue
        data.append((_vector(d), 1 if float(e["realizedReturn"]) > 0.0 else 0))
    return data[-2000:]

def evaluate(candidate_weights: dict[str, float],
             episodes: list[dict[str, Any]],
             baseline_weights: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """Nilai bobot kandidat -> ACCEPT / REJECT + alasan audit lengkap."""
    out: dict[str, Any] = {
        "ok": False, "model": WEIGHT_LAB_VERSION, "provisional": True,
        "decision": "REJECT", "reasons": [],
        "candidate": None, "baseline": None,
    }

    data = _build_data(episodes)
    if len(data) < MIN_SAMPLES:
        out["reasons"] = [f"episode layak {len(data)} < minimal {MIN_SAMPLES} "
                          f"— kumpulkan data dulu"]
        return out
    out["ok"] = True
    out["n"] = len(data)

    cand = _metrics(candidate_weights, data)
    out["candidate"] = cand

    if cand["auc"] is None:
        out["reasons"].append("AUC tak terdefinisi — semua label satu arah, "
                              "data tidak bisa menilai daya pisah")
        return out

    # Gerbang mutlak
    if not cand["gates"]["brierPass"]:
        out["reasons"].append(f"Brier {cand['brier']} > ambang {BRIER_MAX}")
    if not cand["gates"]["aucPass"]:
        out["reasons"].append(f"AUC {cand['auc']} < ambang {AUC_MIN}")
    if not cand["gates"]["precisionPass"]:
        out["reasons"].append(f"precision Wilson LB {cand['precisionWilsonLB']} "
                              f"< ambang {PRECISION_WILSON_MIN}")

    # Gerbang relatif (hanya jika baseline lolos mutlak — baseline rusak tak layak jadi acuan)
    if baseline_weights is not None:
        b = _metrics(baseline_weights, data)
        out["baseline"] = b
        if b["auc"] is not None and b["gatesPassed"]:
            worse, equal = [], []
            if cand["brier"] > b["brier"] + TARGET_EPS:
                worse.append(f"Brier lebih buruk ({cand['brier']} vs {b['brier']})")
            if cand["auc"] < b["auc"] - TARGET_EPS:
                worse.append(f"AUC lebih buruk ({cand['auc']} vs {b['auc']})")
            if cand["precisionWilsonLB"] < b["precisionWilsonLB"] - TARGET_EPS:
                worse.append("precision LB lebih buruk "
                             f"({cand['precisionWilsonLB']} vs {b['precisionWilsonLB']})")
            if not worse:
                strictly = (cand["brier"] < b["brier"] - TARGET_EPS
                            or cand["auc"] > b["auc"] + TARGET_EPS
                            or cand["precisionWilsonLB"] > b["precisionWilsonLB"] + TARGET_EPS)
                if not strictly:
                    equal.append("tidak mengungguli baseline di metrik mana pun")
            out["reasons"].extend(worse + equal)

    if not out["reasons"]:
        out["decision"] = "ACCEPT"
        out["verdict"] = "LOLOS GERBANG — boleh dipasang, status tetap PROVISIONAL (wajib monitoring live)"
    else:
        out["verdict"] = "TIDAK LOLOS — jangan pasang ke production"
    return out

# ---------------------------------------------------------------------------
# Self-test (deterministik, seed tetap)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from random import Random

    rng = Random(42)

    def make_eps(n: int, ret_fn, noise: float = 0.0) -> list[dict[str, Any]]:
        eps = []
        for _ in range(n):
            d = {k: rng.uniform(-1, 1) for k in DENDRITE_KEYS}
            ret = ret_fn(d) + (rng.gauss(0.0, noise) if noise else 0.0)
            eps.append({"status": "CLOSED", "dendrites": d,
                        "realizedReturn": round(ret, 6)})
        return eps

    w_true = {k: 0.0 for k in DENDRITE_KEYS}
    w_true["bias"] = 0.0
    w_true["screenSignal"] = 1.0
    w_zero = {k: 0.0 for k in DENDRITE_KEYS}
    w_zero["bias"] = 0.0

    # 1) Data terlalu sedikit -> REJECT dengan alasan jelas
    r0 = evaluate(w_true, make_eps(10, lambda d: 0.05 * (1 if d["screenSignal"] > 0 else -1)))
    assert r0["ok"] is False and r0["decision"] == "REJECT", "1"
    assert "minimal" in r0["reasons"][0], "1b"

    # 2) Sinyal asli -> lolos semua gerbang mutlak + ungguli baseline nol -> ACCEPT
    eps_sig = make_eps(200, lambda d: 0.05 * tanh(d["screenSignal"]), noise=0.005)
    r1 = evaluate(w_true, eps_sig, baseline_weights=w_zero)
    assert r1["decision"] == "ACCEPT", f"2: {r1['reasons']}"
    assert r1["candidate"]["auc"] > 0.9, f"2b: {r1['candidate']}"
    assert r1["candidate"]["brier"] < 0.2, f"2c: {r1['candidate']}"
    assert r1["candidate"]["gatesPassed"] is True, "2d"

    # 3) Data murni noise -> gagal gerbang (AUC ~0.5, precision LB rendah) -> REJECT
    eps_noise = make_eps(200, lambda d: rng.uniform(-0.05, 0.05))
    r2 = evaluate(w_true, eps_noise)
    assert r2["decision"] == "REJECT", f"3: {r2['reasons']}"
    assert r2["candidate"]["auc"] < 0.65, f"3b: {r2['candidate']}"

    # 4) Gerbang RELATIF: kandidat lolos mutlak tapi tak ungguli baseline -> REJECT
    eps_sep = make_eps(200, lambda d: 0.05 if d["screenSignal"] > 0 else -0.05)
    w_weak = {k: 0.0 for k in DENDRITE_KEYS}
    w_weak["screenSignal"] = 0.3  # ranking sama (monotonik) -> AUC sama, kalibrasi lebih buruk
    r3 = evaluate(w_weak, eps_sep, baseline_weights=w_true)
    assert r3["decision"] == "REJECT", f"4: {r3['reasons']}"
    assert r3["candidate"]["gatesPassed"] is True, "4b: harus lolos mutlak"
    assert any("baseline" in s or "buruk" in s for s in r3["reasons"]), "4c"

    # 5) Semua label satu arah -> AUC tak terdefinisi -> REJECT
    eps_up = make_eps(60, lambda d: 0.03)
    r4 = evaluate(w_true, eps_up)
    assert r4["decision"] == "REJECT" and any("tak terdefinisi" in s for s in r4["reasons"]), "5"

    # 6) Determinisme: input sama -> hasil identik
    r5a = evaluate(w_true, eps_sig, baseline_weights=w_zero)
    r5b = evaluate(w_true, eps_sig, baseline_weights=w_zero)
    assert r5a == r5b, "6"

    # 7) Hasil JSON-serializable penuh (Vercel-safe)
    for r in (r0, r1, r2, r3, r4):
        json.dumps(r)

    # 8) Episode PENDING / netral-disaring tidak masuk hitungan
    dirty = list(eps_sig)
    dirty.append({"status": "PENDING", "dendrites": {}, "realizedReturn": None})
    dirty.append({"status": "CLOSED", "dendrites": {}, "realizedReturn": 0.03})
    assert evaluate(w_true, dirty, baseline_weights=w_zero)["n"] == 200, "8"

    print("SELF-TEST OK — weight_lab (gerbang Brier/AUC/Precision + gerbang relatif + deterministik)")

if __name__ == "__main__":
    _self_test()
