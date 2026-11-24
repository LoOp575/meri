"""
Meridian — Neuron Logistic v1 (guru offline: refit bobot dari episode)
======================================================================

Peran biologis: konsolidasi memori saat "tidur" — hippocampus menyerahkan
episode tertutup, guru ini mencoba refit bobot neuron, lalu MENUSULKAN
(usulan) — bukan memasang. Keputusan pemasangan tetap di tangan audit:

  1. Guru refit bobot dari episode CLOSED (target = realized return di-scale).
  2. Split temporal: episode lama 75% = train, episode baru 25% = test
     (menghormati urutan waktu — tidak ada kebocoran masa depan).
  3. Gerbang penerimaan: test MSE baru harus turun >= MIN_IMPROVEMENT
     relatif terhadap bobot lama. Kalau data noise -> REJECT.
  4. Usulan berlabel PROVISIONAL -> wajib lolos weight_lab.py sebelum live.

Murni komputasi, tanpa I/O — deterministik (tanpa shuffle acak).
Self-test:  python3 backend/neuron_logistic.py
"""

from __future__ import annotations

import json
from math import tanh
from typing import Any, Optional

LOGISTIC_VERSION = "neuron-logistic-v1.0-provisional"

# --- Hyperparameter refit (dievaluasi, tetap eksplisit) ----------------------
MIN_EPISODES = 30        # di bawah ini: tidak layak refit (terlalu sedikit)
TEST_FRACTION = 0.25     # 25% episode TERBARU jadi test set
EPOCHS = 400             # iterasi gradient descent full-batch
LR = 0.10                # learning rate refit
L2 = 0.01                # regularisasi: bobot kecil-kecil, anti overfit
W_CAP = 1.5              # pengaman sama dengan neuron_core
MIN_IMPROVEMENT = 0.05   # test MSE harus turun >= 5% relatif -> ACCEPT
TARGET_SCALE = 0.05      # sinkron neuron_core: +5% -> target +1.0

# SINKRON dengan neuron_core.DENDRITE_KEYS (disengaja duplikat agar modul
# tetap isolasi & bisa jalan standalone — ubah kedua tempat bila berubah).
DENDRITE_KEYS = [
    "screenSignal", "demandQuality", "stability", "fpStable", "fpTail",
    "fpLiveDir", "rvol", "unlockGate", "vwapDist",
]

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _target(realized_return: float) -> float:
    return _clamp(realized_return / TARGET_SCALE, -1.0, 1.0)

def _vector(dendrites: dict[str, float]) -> dict[str, float]:
    x = {k: _clamp(float(dendrites.get(k, 0.0) or 0.0), -1.0, 1.0)
         for k in DENDRITE_KEYS}
    x["bias"] = 1.0
    return x

# ---------------------------------------------------------------------------
# Inti refit
# ---------------------------------------------------------------------------

def _mse(weights: dict[str, float], data: list[tuple[dict[str, float], float]]) -> float:
    if not data:
        return 0.0
    total = 0.0
    for x, t in data:
        z = sum(weights[k] * x[k] for k in weights)
        total += (tanh(z) - t) ** 2
    return total / len(data)

def fit_weights(init_weights: dict[str, float],
                data: list[tuple[dict[str, float], float]],
                epochs: int = EPOCHS, lr: float = LR,
                l2: float = L2) -> dict[str, float]:
    """Gradient descent full-batch pada loss 0.5*(t - tanh(w·x))^2 + l2*||w||²."""
    w = {k: float(init_weights.get(k, 0.0)) for k in init_weights}
    n = len(data)
    if n == 0:
        return w
    for _ in range(epochs):
        grad = {k: 0.0 for k in w}
        for x, t in data:
            z = sum(w[k] * x[k] for k in w)
            y = tanh(z)
            coef = (y - t) * (1.0 - y * y)
            for k in w:
                grad[k] += coef * x[k]
        for k in w:
            g = grad[k] / n + l2 * w[k]
            w[k] = _clamp(w[k] - lr * g, -W_CAP, W_CAP)
    return w

def propose(init_weights: dict[str, float],
            closed_episodes: list[dict[str, Any]],
            min_episodes: int = MIN_EPISODES) -> dict[str, Any]:
    """Refit + gerbang penerimaan. TIDAK memasang bobot — hanya usulan."""
    base = {
        "ok": False, "model": LOGISTIC_VERSION, "provisional": True,
        "decision": "REJECT", "reason": "", "metrics": {},
    }

    # Saring: hanya episode tertutup dengan outcome terukur & dendrite valid
    data: list[tuple[dict[str, float], float]] = []
    for e in closed_episodes:
        if e.get("status") != "CLOSED" or e.get("realizedReturn") is None:
            continue
        d = e.get("dendrites") or {}
        if not any(abs(float(d.get(k, 0.0) or 0.0)) > 1e-9 for k in DENDRITE_KEYS):
            continue  # semua dendrite netral -> tidak informatif
        data.append((_vector(d), _target(float(e["realizedReturn"]))))
    data = data[-2000:]  # batas aman sama seperti ring buffer hippocampus

    if len(data) < min_episodes:
        base["reason"] = (f"episode layak {len(data)} < minimal {min_episodes} "
                          f"— kumpulkan data dulu")
        base["metrics"] = {"nEpisodes": len(data)}
        return base

    # Split temporal: lama = train, TERBARU = test (tanpa shuffle, anti bocor)
    n_test = max(1, int(len(data) * TEST_FRACTION))
    train, test = data[:-n_test], data[-n_test:]

    old_w = {k: float(init_weights.get(k, 0.0)) for k in init_weights}
    new_w = fit_weights(old_w, train)

    mse_test_old = _mse(old_w, test)
    mse_test_new = _mse(new_w, test)
    mse_train_old = _mse(old_w, train)
    mse_train_new = _mse(new_w, train)

    # Gerbang: perbaikan test MSE relatif >= MIN_IMPROVEMENT
    if mse_test_old <= 1e-12:
        improved = mse_test_new <= 1e-12
        rel = 0.0
    else:
        rel = (mse_test_old - mse_test_new) / mse_test_old
        improved = rel >= MIN_IMPROVEMENT

    base.update({
        "ok": True,
        "decision": "ACCEPT" if improved else "REJECT",
        "reason": (f"test MSE turun {rel * 100:.1f}% (ambang {MIN_IMPROVEMENT * 100:.0f}%)"
                   if improved else
                   f"test MSE hanya turun {rel * 100:.1f}% "
                   f"(ambang {MIN_IMPROVEMENT * 100:.0f}%) — data belum informatif"),
        "metrics": {
            "nEpisodes": len(data), "nTrain": len(train), "nTest": len(test),
            "trainMseOld": round(mse_train_old, 6),
            "trainMseNew": round(mse_train_new, 6),
            "testMseOld": round(mse_test_old, 6),
            "testMseNew": round(mse_test_new, 6),
            "improvementPct": round(rel * 100, 2),
        },
        "oldWeights": {k: round(v, 4) for k, v in old_w.items()},
        "newWeights": {k: round(v, 4) for k, v in new_w.items()},
    })
    return base

# ---------------------------------------------------------------------------
# Self-test (deterministik, tanpa I/O)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from random import Random

    rng = Random(42)  # seed tetap -> deterministik

    def make_eps(w_true: dict[str, float], n: int, noise: float,
                 informative: bool) -> list[dict[str, Any]]:
        eps = []
        for _ in range(n):
            d = {k: rng.uniform(-1, 1) for k in DENDRITE_KEYS}
            z = sum(w_true[k] * d[k] for k in DENDRITE_KEYS)
            t_signal = tanh(z)
            ret_signal = t_signal * TARGET_SCALE
            ret = (ret_signal + rng.gauss(0.0, noise)) if informative \
                else rng.uniform(-0.05, 0.05)
            eps.append({"status": "CLOSED", "dendrites": d,
                        "realizedReturn": round(ret, 6)})
        return eps

    w_true = {k: 0.0 for k in DENDRITE_KEYS}
    w_true["bias"] = 1.0
    w_true["screenSignal"] = 0.9
    w_true["rvol"] = 0.4

    init_w = {k: 0.0 for k in DENDRITE_KEYS}
    init_w["bias"] = 0.0

    # 1) Episode terlalu sedikit -> ok=False, REJECT dengan alasan jelas
    p0 = propose(init_w, make_eps(w_true, 10, 0.0, True))
    assert p0["ok"] is False and p0["decision"] == "REJECT", "1"
    assert "minimal" in p0["reason"], "1b"

    # 2) Sinyal kuat + noise kecil -> refit menemukan bobot, ACCEPT
    eps_sig = make_eps(w_true, 200, 0.005, True)
    p1 = propose(init_w, eps_sig)
    assert p1["ok"] is True and p1["decision"] == "ACCEPT", f"2: {p1['reason']}"
    assert p1["newWeights"]["screenSignal"] > 0.3, f"2b: {p1['newWeights']}"
    assert p1["metrics"]["testMseNew"] < p1["metrics"]["testMseOld"], "2c"

    # 3) Data murni noise (return acak) -> tidak ada perbaikan -> REJECT
    eps_noise = make_eps(w_true, 200, 0.0, False)
    p2 = propose(init_w, eps_noise)
    assert p2["ok"] is True and p2["decision"] == "REJECT", f"3: {p2['reason']}"

    # 4) Episode PENDING / netral / tanpa dendrite -> disaring, tidak masuk hitungan
    dirty = make_eps(w_true, 60, 0.005, True)
    dirty.append({"status": "PENDING", "dendrites": {}, "realizedReturn": None})
    dirty.append({"status": "CLOSED", "dendrites": {}, "realizedReturn": 0.03})
    p3 = propose(init_w, dirty)
    assert p3["metrics"]["nEpisodes"] == 60, f"4: {p3['metrics']}"

    # 5) Determinisme: input sama -> hasil identik (tanpa shuffle acak)
    p4a = propose(init_w, eps_sig)
    p4b = propose(init_w, eps_sig)
    assert p4a["newWeights"] == p4b["newWeights"], "5"

    # 6) Pengaman bobot: tidak ada yang melewati W_CAP
    assert all(abs(v) <= W_CAP + 1e-9 for v in p1["newWeights"].values()), "6"

    # 7) Usulan JSON-serializable penuh (Vercel-safe)
    json.dumps(p1), json.dumps(p2), json.dumps(p0)  # raises jika gagal

    # 8) Split temporal benar: nTest = 25% terbaru
    assert p1["metrics"]["nTrain"] == 150 and p1["metrics"]["nTest"] == 50, "8"

    print("SELF-TEST OK — neuron_logistic (refit temporal split + gerbang MSE + deterministik)")

if __name__ == "__main__":
    _self_test()
