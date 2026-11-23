"""
Meridian — Crypto Quality Screener v2 (modul ISOLASI, screening.py v1 tidak disentuh)
=====================================================================================

Perbaikan vs v1 (peta fix -> kode):
 #1  d_pos DIHAPUS dari sigmoid p^acc (g_org terhitung 3x = multicollinearity).
     d_lead tetap ada di geometric mean S^DR.
 #2  Fitur sigmoid DISTANDARDISASI (z-score) + intercept netral.
     Bobot PROVISIONAL — production trust HANYA setelah weight_lab LULUS.
 #3  Dead zone 61-89 candle dihapus: tanpa prior window, faktor akselerasi
     dikeluarkan dari geometric mean (eksponen dinormalisasi ulang 0.5/0.25/0.25),
     ditandai approx=True. Tidak lagi S^DR=0 selalu.
 #4  s_flow netral saat tanpa prior window (tidak menghukum penurunan pertama).
 #5  Satu konstanta ABN_RET_FLOOR = 5.5% — sinkron dengan docs.
 #6  rvol & vwap_dist (sudah dihitung v1 tapi tak dipakai) masuk sebagai fitur sigmoid.
 #7  Fitur `regime` opsional (mis. return 30d BTC, sat'd) — korektor rezim.

Yang SENGAJA TIDAK diubah (terevaluasi relevan):
 - Struktur geometric mean S^DR (0.4/0.2/0.2/0.2) + saturasi score100.
 - Veto filters: low-quality growth, billing distortion, red-flag (urutan override sama).
 - Threshold LONG/WATCH (55/0.60, 38/0.50) — agar A/B vs v1 apple-to-apple.
 - Model instability (event frequency, magnitude, clustering) + exit triggers.
 - Unlock gate diterapkan di level route (shared dengan v1) — bukan di modul ini.

Murni komputasi, tanpa I/O — deterministik, bisa unit-test.
Jalankan self-test:  python3 backend/screening_v2.py
"""

from __future__ import annotations

from math import exp, log
from statistics import mean, pstdev
from typing import Any, Optional

EPS = 1e-9
WINDOW = 30
VWAP_WINDOW = 20

# --- Konstanta struktural (sama dengan v1, dievaluasi relevan) -----------
LAMBDA_FLOW = 3.0
ALPHA_NRR, BETA_GRR, GAMMA_CHURN = 0.5, 0.8, 1.1
SAT_K_QDR, SAT_K_SDR, SAT_K_FILE = 0.45, 0.12, 0.35
G_MAX = 0.90
LONG_SDR, LONG_PACC = 55.0, 0.60
WATCH_SDR, WATCH_PACC = 38.0, 0.50
FILE_AVOID, PBAD_AVOID = 55.0, 0.65

# FIX #5: satu konstanta, sinkron dengan docs (5.5%)
ABN_RET_FLOOR = 0.055

# --- FIX #1/#2/#6/#7: bobot sigmoid v2 ------------------------------------
# PROVISIONAL — pada fitur terstandarisasi. WAJIB diganti output weight_lab.py
# (production_patch) setelah verdict LULUS di data nyata 3 tahun.
P_ACC_V2_WEIGHTS: dict[str, float] = {
    "intercept": -0.55,
    "g_pos": 0.90,      # organik growth (satu-satunya wakil g_org -> fix #1)
    "a_pos": 0.60,      # akselerasi
    "gap_pos": 0.70,    # demand lead vs harga
    "c_cash": 0.60,     # taker-buy share
    "s_flow": 0.50,     # stabilitas partisipasi
    "rvol": 0.40,       # FIX #6: relative volume (sat'd)
    "vwap_dist": 0.40,  # FIX #6: jarak harga ke VWAP-20 (sat'd)
    "regime": 0.30,     # FIX #7: rezim pasar (opsional)
}
# (mean, std) prior PROVISIONAL per fitur — diganti featureMean/featureStd dari lab.
P_ACC_V2_STATS: dict[str, tuple[float, float]] = {
    "g_pos": (0.25, 0.25), "a_pos": (0.25, 0.25), "gap_pos": (0.15, 0.20),
    "c_cash": (0.50, 0.10), "s_flow": (0.80, 0.20),
    "rvol": (0.0, 0.50), "vwap_dist": (0.0, 0.50), "regime": (0.0, 0.50),
}

MODEL_VERSION = "screening-v2.0-provisional"

# ---------------------------------------------------------------------------
# Helper numerik
# ---------------------------------------------------------------------------

def sat(x: float) -> float:
    return x / (1.0 + abs(x))

def pos_sat(x: float) -> float:
    return max(0.0, sat(x))

def score100(raw: float, k: float) -> float:
    raw = max(0.0, raw)
    return 100.0 * raw / (raw + k)

def sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + exp(-z))

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or abs(b) < EPS:
        return default
    return a / b

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _z(feat: str, value: float, missing: bool = False) -> float:
    """FIX #2: z-score; fitur yang tidak tersedia -> z=0 (kontribusi netral)."""
    if missing:
        return 0.0
    mu, sd = P_ACC_V2_STATS[feat]
    return clamp((value - mu) / (sd if sd > EPS else 1.0), -4.0, 4.0)

def _growth(now: float, before: float) -> float:
    return 0.0 if before <= EPS else (now - before) / before

def _vwap(candles: list[dict], window: int = VWAP_WINDOW) -> float:
    seg = candles[-window:]
    num = sum(((c["high"] + c["low"] + c["close"]) / 3.0) * c["volume"] for c in seg)
    den = sum(c["volume"] for c in seg)
    return safe_div(num, den, seg[-1]["close"] if seg else 0.0)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_features_v2(
    candles: list[dict[str, Any]],
    regime: Optional[float] = None,
) -> dict[str, Any]:
    """Screener v2. `regime`: opsional, mis. sat(return 30d BTC); None -> 0.0 netral."""
    n = len(candles)
    if n < 2 * WINDOW + 1:  # FIX #3: minimum efektif = 61 candle (2 window + edge)
        return {"ok": False, "model": MODEL_VERSION,
                "reason": f"Butuh minimal {2 * WINDOW + 1} candle harian, dapat {n}.",
                "verdict": "INSUFFICIENT_DATA"}

    closes = [c["close"] for c in candles]
    price = closes[-1]
    has_prior = n >= 3 * WINDOW  # FIX #3/#4: prior window tersedia?

    S = lambda seg, k: sum(c[k] for c in seg)
    cur = candles[n - WINDOW: n]
    prev = candles[n - 2 * WINDOW: n - WINDOW]
    prior = candles[n - 3 * WINDOW: n - 2 * WINDOW] if has_prior else None

    # --- Model 1: demand quality ------------------------------------------
    g_rev = closes[-1] / closes[n - WINDOW - 1] - 1.0
    g_qv = _growth(S(cur, "quoteVolume"), S(prev, "quoteVolume"))
    g_org = _growth(S(cur, "volume"), S(prev, "volume"))
    g_org_prev = _growth(S(prev, "volume"), S(prior, "volume")) if prior else None
    a_dr = (g_org - g_org_prev) if prior else None  # None = tidak tersedia, bukan 0
    d_lead = g_org - g_rev
    gap = safe_div(d_lead, max(1.0 + g_rev, 0.05), 0.0)
    c_cash = safe_div(S(cur, "takerBuyBase"), S(cur, "volume"), 0.5)

    p_growth = _growth(S(cur, "trades"), S(prev, "trades"))
    p_growth_prev = _growth(S(prev, "trades"), S(prior, "trades")) if prior else None
    # FIX #4: tanpa prior -> s_flow netral 1.0 (bukan menghukum penurunan pertama)
    s_flow = exp(-LAMBDA_FLOW * max(0.0, p_growth_prev - p_growth)) if prior else 1.0
    churn = max(0.0, -p_growth)

    nrr = clamp(1.0 + g_org, 0.0, 2.0)
    grr = sum(1 for c in cur if safe_div(c["takerBuyBase"], c["volume"], 0.5) >= 0.5) / WINDOW
    q_ret = ALPHA_NRR * (nrr - 1.0) + BETA_GRR * grr - GAMMA_CHURN * churn

    g_pos = pos_sat(g_org)
    q_dr_raw = g_pos * s_flow * max(1.0 + q_ret, 0.0) * (1.0 + c_cash)
    q_dr_100 = score100(q_dr_raw, SAT_K_QDR)

    r_hat = (1.0 + g_org) * c_cash
    gap_pos = max(0.0, sat(gap))
    d_pos = pos_sat(d_lead)

    # FIX #3: geometric mean — tanpa prior, faktor akselerasi dikeluarkan,
    # eksponen dinormalisasi ulang (0.5/0.25/0.25); faktor nol tetap nolkan skor.
    if g_pos > 0.0 and d_pos > 0.0 and gap_pos > 0.0:
        if prior:
            a_pos = pos_sat(a_dr)
            s_raw = (g_pos ** 0.4 * a_pos ** 0.2 * d_pos ** 0.2 * gap_pos ** 0.2
                     if a_pos > 0.0 else 0.0)
        else:
            a_pos = None
            s_raw = g_pos ** 0.5 * d_pos ** 0.25 * gap_pos ** 0.25
    else:
        a_pos, s_raw = (pos_sat(a_dr) if prior else None), 0.0
    s_dr_100 = score100(s_raw, SAT_K_SDR)

    # --- p^acc v2: fitur terstandarisasi (fix #1/#2/#6/#7) ------------------
    avg_qv_30 = mean(c["quoteVolume"] for c in cur)
    rvol = safe_div(candles[-1]["quoteVolume"], avg_qv_30, 1.0)
    vwap_20 = _vwap(candles)
    close_vs_vwap = safe_div(price - vwap_20, vwap_20, 0.0)
    reg = clamp(float(regime) if regime is not None else 0.0, -1.0, 1.0)

    W = P_ACC_V2_WEIGHTS
    z = (
        W["intercept"]
        + W["g_pos"] * _z("g_pos", g_pos)
        + W["a_pos"] * _z("a_pos", pos_sat(a_dr) if prior else 0.0, missing=not prior)
        + W["gap_pos"] * _z("gap_pos", gap_pos)
        + W["c_cash"] * _z("c_cash", c_cash)
        + W["s_flow"] * _z("s_flow", s_flow, missing=not prior)  # fix #4: netral
        + W["rvol"] * _z("rvol", sat(rvol - 1.0))
        + W["vwap_dist"] * _z("vwap_dist", sat(close_vs_vwap))
        + W["regime"] * _z("regime", reg)
    )
    p_acc = sigmoid(z)

    # --- Model 2: instability (fix #5: ABN_RET_FLOOR) -----------------------
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, n) if closes[i - 1] > EPS]
    ret_std = pstdev(rets) if len(rets) > 1 else 0.0
    ret_thr = max(2.5 * ret_std, ABN_RET_FLOOR)

    log_qv = [log(max(c["quoteVolume"], 1e-9)) for c in candles]
    sx, sxx = n * (n - 1) / 2.0, (n - 1) * n * (2 * n - 1) / 6.0
    sy = sum(log_qv)
    sxy = sum(i * v for i, v in enumerate(log_qv))
    denom = n * sxx - sx * sx
    if abs(denom) > EPS:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        resid = [v - (slope * i + intercept) for i, v in enumerate(log_qv)]
    else:
        resid = [0.0] * n
    resid_std = pstdev(resid) if len(resid) > 1 else 0.0
    qv_thr = 2.5 * resid_std if resid_std > 1e-6 else float("inf")

    event_days = [
        i for i in range(max(1, n - WINDOW), n)
        if (closes[i - 1] > EPS and abs(closes[i] / closes[i - 1] - 1.0) >= ret_thr)
        or abs(resid[i]) >= qv_thr
    ]
    n_events = len(event_days)
    if n_events:
        ev_mag = mean(abs(closes[i] / closes[i - 1] - 1.0)
                      for i in event_days if closes[i - 1] > EPS)
        q_corr = sat(safe_div(ev_mag, mean(abs(r) for r in rets) + EPS, 1.0))
    else:
        q_corr = 0.0
    r_topic = (sum(1 for i in event_days
                   if any(0 < abs(i - j) <= 2 for j in event_days if j != i)) / n_events
               if n_events >= 2 else 0.0)
    s_file_raw = (n_events / 10.0) * (0.35 + 0.65 * q_corr) * (1.0 + 1.2 * r_topic)
    s_file_100 = score100(s_file_raw, SAT_K_FILE)
    p_bad = sigmoid((s_file_100 - 52.0) / 12.0 + 0.5 * r_topic)

    # --- Verdict (urutan override SAMA dengan v1) ---------------------------
    billing = g_qv > 0.05 and g_org <= 0.0
    low_q = g_rev > 0.05 and p_growth < -0.10 and c_cash < 0.47
    red_flag = s_file_100 >= FILE_AVOID or p_bad >= PBAD_AVOID

    reasons: list[str] = []
    if billing:
        reasons.append(f"Volume USD {g_qv:+.1%} tapi volume organik {g_org:+.1%} — demand dari harga/mix, bukan adopsi (billing-distortion).")
    if low_q:
        reasons.append(f"Harga {g_rev:+.1%}, partisipasi {p_growth:+.1%}, taker-buy {c_cash:.2f} — low-quality growth, tolak.")
    if red_flag:
        reasons.append(f"{n_events} hari abnormal dalam 30d, instability {s_file_100:.0f}/100 — eksklusi red-flag.")
    reasons.append(f"Organik {g_org:+.1%} vs USD {g_qv:+.1%} vs harga {g_rev:+.1%}; lead-vs-price {d_lead:+.1%}.")
    if prior:
        reasons.append(f"Akselerasi A={a_dr:+.1%} ({g_org_prev:+.1%} → {g_org:+.1%}).")
    else:
        reasons.append("History < 90 candle: faktor akselerasi/stabilitas dinetralkan (approx).")

    if low_q:
        verdict = "AVOID"
    elif billing:
        verdict = "NO_SIGNAL"
    elif red_flag:
        verdict = "AVOID"
    elif (s_dr_100 >= LONG_SDR and p_acc >= LONG_PACC
          and 0.0 < g_org <= G_MAX and price > vwap_20):
        verdict = "LONG"
        reasons.append(f"Signal {s_dr_100:.0f}/100, p(acc)={p_acc:.2f}, harga > VWAP-20 ({close_vs_vwap:+.1%}) — kriteria long terpenuhi.")
    elif s_dr_100 >= WATCH_SDR and p_acc >= WATCH_PACC:
        verdict = "WATCH"
        reasons.append("Konstruk demand-quality positif tapi minimal satu kriteria long tak terpenuhi — watchlist.")
    else:
        verdict = "NEUTRAL"
        reasons.append("Tanpa edge: skor/probabilitas di bawah threshold.")

    exit_reasons = []
    if g_org < 0:
        exit_reasons.append("organic demand growth turned negative")
    if prior and s_flow < 0.4:
        exit_reasons.append("engagement trend unstable (flow-stability < 0.40)")
    if churn > 0.25:
        exit_reasons.append(f"participation exodus ({churn:+.1%})")

    return {
        "ok": True, "model": MODEL_VERSION, "approx": not prior,
        "verdict": verdict, "price": price,
        "features": {
            "gRev": g_rev, "gQuoteVolReported": g_qv, "gOrganic": g_org,
            "gOrganicPrev": g_org_prev, "acceleration": a_dr, "leadVsPrice": d_lead,
            "gap": gap, "cashConversion": c_cash, "participationGrowth": p_growth,
            "flowStability": s_flow, "churn": churn, "nrrAnalog": nrr,
            "grrAnalog": grr, "retentionQuality": q_ret, "expectedRecognition": r_hat,
            "rvol": rvol, "vwap20": vwap_20, "closeVsVwap20": close_vs_vwap,
            "regime": reg,
        },
        "scores": {
            "demandQuality": round(q_dr_100, 1), "signal": round(s_dr_100, 1),
            "instability": round(s_file_100, 1),
            "pAcceleration": round(p_acc, 4), "pAdverse": round(p_bad, 4),
        },
        "risk": {"abnormalEvents30d": n_events, "correctionMagnitude": q_corr,
                 "clusteredEvents": r_topic, "internalControl": None},
        "filters": {"billingDistortion": billing, "lowQualityGrowth": low_q,
                    "redFlagExcluded": red_flag, "belowVwap": price <= vwap_20,
                    "aboveBlowoffCap": g_org > G_MAX, "approxFeatures": not prior},
        "reasons": reasons, "exitTriggers": exit_reasons,
    }

def formula_spec_v2() -> dict[str, Any]:
    """Metadata rumus v2 — untuk route /api/screen-v2/spec (Tahap B)."""
    return {
        "ok": True, "model": MODEL_VERSION,
        "fixes": {
            "1": "d_pos dihapus dari sigmoid (multicollinearity g_org 3x)",
            "2": "fitur z-score + intercept netral; bobot PROVISIONAL",
            "3": "dead zone 61-89 dihapus; geomean renormalisasi 0.5/0.25/0.25 saat approx",
            "4": "s_flow netral tanpa prior window",
            "5": "ABN_RET_FLOOR=5.5% sinkron docs",
            "6": "rvol & vwap_dist masuk sigmoid",
            "7": "fitur regime opsional",
        },
        "weightsProvisional": P_ACC_V2_WEIGHTS,
        "featureStats": P_ACC_V2_STATS,
        "adoptionRule": "Bobot produksi HANYA setelah weight_lab verdict LULUS (Brier/AUC/precision gates).",
    }

# ---------------------------------------------------------------------------
# Self-test (deterministik, tanpa I/O)
# ---------------------------------------------------------------------------

def _mk(closes, vols, trades=None, taker=None):
    out = []
    for i, (c, v) in enumerate(zip(closes, vols)):
        o = closes[i - 1] if i else c
        tb = v * (taker[i] if taker else 0.55)
        out.append({"date": f"d{i:03d}", "open": o, "high": max(o, c) * 1.001,
                    "low": min(o, c) * 0.999, "close": c, "volume": v,
                    "quoteVolume": v * c, "trades": trades[i] if trades else int(v * 0.5) + 1,
                    "takerBuyBase": tb, "takerBuyQuote": tb * c})
    return out

def _self_test():
    # FIX struktural
    assert "d_pos" not in P_ACC_V2_WEIGHTS, "fix #1 gagal"
    assert ABN_RET_FLOOR == 0.055, "fix #5 gagal"
    assert {"rvol", "vwap_dist", "regime"} <= set(P_ACC_V2_WEIGHTS), "fix #6/#7 gagal"

    # Kasus A: organik step-up sehat -> LONG
    cl = [100.0 + i * 0.04 for i in range(120)]
    vl = [1000.0] * 90 + [1500.0] * 30
    r = compute_features_v2(_mk(cl, vl))
    assert r["verdict"] == "LONG", f"A: {r['verdict']} s={r['scores']['signal']} p={r['scores']['pAcceleration']}"
    assert not r["approx"]

    # Kasus B: USD naik, organik flat -> NO_SIGNAL
    cl = [100.0 + i * 0.25 for i in range(120)]
    r = compute_features_v2(_mk(cl, [1000.0] * 120))
    assert r["verdict"] == "NO_SIGNAL", f"B: {r['verdict']}"

    # Kasus C: harga naik, partisipasi anjlok, taker lemah -> AVOID
    cl = [100.0] * 90 + [100.0 + i * 0.4 for i in range(1, 31)]
    tr = [5000] * 90 + [5000 - i * 36 for i in range(1, 31)]
    r = compute_features_v2(_mk(cl, [1000.0] * 120, trades=tr, taker=[0.42] * 120))
    assert r["verdict"] == "AVOID", f"C: {r['verdict']}"

    # Kasus D: 4 hari abnormal berkelompok -> AVOID (red flag)
    cl = ([100.0] * 95 + [92.0, 100.0] + [100.0] * 13 + [92.0, 100.0] + [100.0] * 8)
    r = compute_features_v2(_mk(cl, [1000.0] * 120))
    assert r["verdict"] == "AVOID" and r["risk"]["abnormalEvents30d"] == 4, f"D: {r['verdict']}"

    # Kasus E: 70 candle (dead zone v1) -> tetap dapat sinyal, approx=True
    cl = [100.0] * 40 + [100.0] * 30
    vl = [1000.0] * 40 + [1500.0] * 30
    r = compute_features_v2(_mk(cl, vl))
    assert r["ok"] and r["approx"], "E: approx flag hilang"
    assert r["scores"]["signal"] > 38, f"E: fix #3 gagal, signal={r['scores']['signal']}"
    assert r["verdict"] in ("WATCH", "LONG"), f"E: {r['verdict']}"

    # Kasus F: 50 candle -> INSUFFICIENT_DATA
    r = compute_features_v2(_mk([100.0] * 50, [1000.0] * 50))
    assert not r["ok"] and r["verdict"] == "INSUFFICIENT_DATA"

    print("SELF-TEST OK — screening_v2 (7 fix terpasang, verdict A-F sesuai desain)")
    for tag, args in [("A", (cl_a := [100.0 + i * 0.04 for i in range(120)],
                             [1000.0] * 90 + [1500.0] * 30))]:
        pass  # (ringkas: detail sudah di-assert di atas)

if __name__ == "__main__":
    _self_test()
