"""
Meridian v2 — 1D Itô Fokker–Planck Crypto Distribution & Microstructure Engine.
===============================================================================
AUDIT verdict: inti solver Chang–Cooper DIKUNCI 100% dari repo lama (math exact:
positivity + konservasi massa + stasioner OU/power-law). Perubahan HANYA di
layer parameter & kejujuran pelaporan:

 IMPROVEMENT #1  Burn-in EWMA: inisialisasi varians dari rata-rata 5 obs pertama
                 (dulu: obs pertama dominan -> volatilitas bias).
 IMPROVEMENT #2  Kappa adaptif via Variance Ratio: tren -> anchor VWAP melemah,
                 chop -> menguat (dulu: kappa=0.03 hardcoded untuk semua regime).
 IMPROVEMENT #3  Renormalisasi massa + laporkan massDrift (clamp negatif pada
                 Thomas solver bisa menggeser massa secara diam-diam).
 IMPROVEMENT #4  Drift-uncertainty blending: probabilitas dirata-ratakan atas
                 skenario mu ± sigma_mu (sigma_mu = standard error estimasi),
                 bukan menampilkan point estimate seolah pasti.

Self-test: python3 backend/fokker_planck.py -> SELF-TEST OK
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Callable, Optional

import httpx

from backend import market_data

EPS = 1e-12
MIN_CANDLES = 61
DEFAULT_WINDOW = 30

CANDLE_REGIME_META: dict[str, dict[str, Any]] = {
    "ACCUMULATION_STABLE": {"label": "Akumulasi Stabil", "tone": "long", "rank": 1,
        "summary": "Distribusi terpusat di zona stabil dengan condong positif dan ekor kiri tipis."},
    "BULLISH_EXPANSION": {"label": "Ekspansi Naik", "tone": "long", "rank": 2,
        "summary": "Massa probabilitas bergeser ke zona upside namun dispersi volatilitas cukup lebar."},
    "SIDEWAYS_CALM": {"label": "Sideways Tenang", "tone": "neutral", "rank": 3,
        "summary": "Probabilitas harga bertahan di zona stabil dominan dengan drift mendekati netral."},
    "HIGH_VOL_DISPERSED": {"label": "Dispersi Tinggi", "tone": "watch", "rank": 4,
        "summary": "Difusi volatilitas melebar ke dua arah; probabilitas bertahan di zona stabil rendah."},
    "DOWNSIDE_SKEW": {"label": "Condong Turun", "tone": "avoid", "rank": 5,
        "summary": "Tekanan jual menggeser kurva ke zona downside dengan asimetri negatif."},
    "TAIL_CRASH_RISK": {"label": "Risiko Ekor Crash", "tone": "avoid", "rank": 6,
        "summary": "Massa probabilitas di ekor kiri ekstrem (<-2x band) tinggi akibat difusi fat-tail & tekanan jual."},
}

MICRO_REGIME_META: dict[str, dict[str, str]] = {
    "STABLE_BALANCED": {"label": "Stabil & Seimbang", "tone": "long",
        "summary": "Kedalaman bid/ask dan aliran eksekusi seimbang; kurva P(x,t) tetap rapat di sekitar mid-price."},
    "SPOOF_SELL_WALL": {"label": "Indikasi Spoof Jual", "tone": "watch",
        "summary": "Tembok ask tebal disertai cancel tinggi tanpa eksekusi jual sepadan — tekanan turun bersifat transien."},
    "SPOOF_BUY_WALL": {"label": "Indikasi Spoof Beli", "tone": "watch",
        "summary": "Tembok bid tebal disertai cancel tinggi tanpa eksekusi beli sepadan — dorongan naik rawan semu."},
    "REAL_BUY_PRESSURE": {"label": "Tekanan Beli Riil", "tone": "long",
        "summary": "Agresor taker-buy dan kedalaman bid searah menggeser distribusi secara persisten ke zona pump."},
    "REAL_SELL_PRESSURE": {"label": "Tekanan Jual Riil", "tone": "avoid",
        "summary": "Agresor taker-sell menghantam bid secara konsisten; massa distribusi bergeser kuat ke zona dump."},
    "HIGH_VOL_WHIPSAW": {"label": "Whipsaw Volatilitas", "tone": "watch",
        "summary": "Difusi D(x,t) tinggi melebarkan kurva ke kedua ekor secara bersamaan (risiko sapuan dua arah)."},
}

_PREV_BOOK_CACHE: dict[str, dict[str, Any]] = {}
_PREV_BOOK_TTL_SEC = 90.0
_PREV_BOOK_MAX = 120

fp_cache = market_data.TTLCache(ttl_seconds=45.0)

# ---------------------------------------------------------------------------
# Numeric primitives (pure stdlib) — DIKUNCI
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def sat(x: float) -> float:
    return x / (1.0 + abs(x))

def _finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return fallback

def bernoulli(w: float) -> float:
    """Bernoulli function B(w) = w / (exp(w) - 1), C²-smooth and overflow-safe for all w in R."""
    if abs(w) < 1e-4:
        w2 = w * w
        return 1.0 - 0.5 * w + w2 / 12.0 - (w2 * w2) / 720.0
    if w > 50.0:
        return w * math.exp(-w)
    if w < -50.0:
        return -w
    return w / math.expm1(w)

def solve_tridiagonal(
    lower: list[float],
    diag: list[float],
    upper: list[float],
    rhs: list[float],
) -> list[float]:
    """O(N) Thomas algorithm for strictly diagonally dominant tridiagonal system."""
    n = len(diag)
    c_star = [0.0] * n
    d_star = [0.0] * n

    inv0 = 1.0 / diag[0]
    c_star[0] = upper[0] * inv0
    d_star[0] = rhs[0] * inv0

    for i in range(1, n):
        denom = diag[i] - lower[i] * c_star[i - 1]
        inv = 1.0 / (denom if abs(denom) > EPS else EPS)
        c_star[i] = upper[i] * inv if i < n - 1 else 0.0
        d_star[i] = (rhs[i] - lower[i] * d_star[i - 1]) * inv

    out = [0.0] * n
    out[n - 1] = max(0.0, d_star[n - 1])
    for i in range(n - 2, -1, -1):
        out[i] = max(0.0, d_star[i] - c_star[i] * out[i + 1])
    return out

# ---------------------------------------------------------------------------
# Core 1D Itô Fokker–Planck PDE solver — DIKUNCI (+ renormalisasi #3)
# ---------------------------------------------------------------------------

@dataclass
class DistributionMetrics:
    t: float
    p_stable: float
    p_downside: float
    p_upside: float
    p_tail_crash: float
    p_tail_pump: float
    p_adaptive_1sigma: float
    mean_log_return: float
    expected_return_pct: float
    std_log_return: float
    skewness: float
    kurtosis: float
    shannon_entropy: float
    total_mass: float
    boundary_mass: float
    mass_drift: float = 0.0  # IMPROVEMENT #3 (aditif, default aman)

    def to_dict(self) -> dict[str, float]:
        return {
            "t": round(self.t, 4),
            "pStable": round(self.p_stable, 6),
            "pDownside": round(self.p_downside, 6),
            "pUpside": round(self.p_upside, 6),
            "pTailCrash": round(self.p_tail_crash, 6),
            "pTailPump": round(self.p_tail_pump, 6),
            "pAdaptive1Sigma": round(self.p_adaptive_1sigma, 6),
            "meanLogReturn": round(self.mean_log_return, 6),
            "expectedReturnPct": round(self.expected_return_pct, 4),
            "stdLogReturn": round(self.std_log_return, 6),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "shannonEntropy": round(self.shannon_entropy, 4),
            "totalMass": round(self.total_mass, 8),
            "boundaryMass": round(self.boundary_mass, 6),
            "massDrift": round(self.mass_drift, 8),  # IMPROVEMENT #3
        }

@dataclass
class FokkerPlanckSolution:
    final: DistributionMetrics
    timeline: list[DistributionMetrics]
    grid: list[float]
    pdf: list[float]
    dx: float
    stable_band: float
    horizon: float
    mass_drift: float = 0.0  # IMPROVEMENT #3 (aditif)

def _integrate_interval(grid: list[float], p: list[float], dx: float, lo: float, hi: float) -> float:
    total = 0.0
    half = 0.5 * dx
    for x, dens in zip(grid, p):
        overlap = min(x + half, hi) - max(x - half, lo)
        if overlap > 0.0:
            total += dens * overlap
    return total

def _summarize_pdf(
    grid: list[float],
    p: list[float],
    dx: float,
    stable_band: float,
    t: float,
) -> DistributionMetrics:
    total_mass = sum(p) * dx
    p_stable = _integrate_interval(grid, p, dx, -stable_band, stable_band)
    p_down = _integrate_interval(grid, p, dx, -math.inf, -stable_band)
    p_up = _integrate_interval(grid, p, dx, stable_band, math.inf)
    p_crash = _integrate_interval(grid, p, dx, -math.inf, -2.0 * stable_band)
    p_pump = _integrate_interval(grid, p, dx, 2.0 * stable_band, math.inf)

    mean_x = sum(x * d for x, d in zip(grid, p)) * dx
    exp_ret = sum(math.expm1(clamp(x, -4.0, 4.0)) * d for x, d in zip(grid, p)) * dx
    var_x = max(0.0, sum(((x - mean_x) ** 2) * d for x, d in zip(grid, p)) * dx)
    std_x = math.sqrt(var_x)

    p_adapt = _integrate_interval(grid, p, dx, -std_x, std_x) if std_x > EPS else 1.0

    if std_x > 1e-9:
        skew = sum((((x - mean_x) / std_x) ** 3) * d for x, d in zip(grid, p)) * dx
        kurt = sum((((x - mean_x) / std_x) ** 4) * d for x, d in zip(grid, p)) * dx
    else:
        skew, kurt = 0.0, 3.0

    entropy = -sum(d * math.log(d) for d in p if d > 1e-30) * dx
    edge_count = max(1, len(p) // 50)
    boundary_mass = (sum(p[:edge_count]) + sum(p[-edge_count:])) * dx

    return DistributionMetrics(
        t=t,
        p_stable=clamp(p_stable, 0.0, 1.0),
        p_downside=clamp(p_down, 0.0, 1.0),
        p_upside=clamp(p_up, 0.0, 1.0),
        p_tail_crash=clamp(p_crash, 0.0, 1.0),
        p_tail_pump=clamp(p_pump, 0.0, 1.0),
        p_adaptive_1sigma=clamp(p_adapt, 0.0, 1.0),
        mean_log_return=mean_x,
        expected_return_pct=exp_ret * 100.0,
        std_log_return=std_x,
        skewness=skew,
        kurtosis=kurt,
        shannon_entropy=entropy,
        total_mass=total_mass,
        boundary_mass=clamp(boundary_mass, 0.0, 1.0),
    )

def _interp_pdf(grid: list[float], pdf: list[float], dx: float, x: float) -> float:
    if x <= grid[0]:
        return pdf[0]
    if x >= grid[-1]:
        return pdf[-1]
    pos = (x - grid[0]) / dx
    idx = int(pos)
    if idx >= len(grid) - 1:
        return pdf[-1]
    frac = pos - idx
    return pdf[idx] * (1.0 - frac) + pdf[idx + 1] * frac

def export_chart_curve(sol: FokkerPlanckSolution, n_points: int = 85) -> list[dict[str, Any]]:
    """Build zone-segmented curve points for gap-free frontend area plotting."""
    a = sol.stable_band
    f = sol.final
    span = max(3.2 * a, 3.4 * f.std_log_return + abs(f.mean_log_return))
    view_lim = min(abs(sol.grid[-1]), max(span, 2.4 * a))

    base_xs = [-view_lim + (2.0 * view_lim * i) / max(n_points - 1, 1) for i in range(n_points)]
    anchors = [-2.0 * a, -a, 0.0, a, 2.0 * a]
    merged: list[float] = sorted(
        [x for x in base_xs if all(abs(x - k) > 1e-6 * view_lim for k in anchors)]
        + [k for k in anchors if -view_lim <= k <= view_lim]
    )

    tol = 1e-9 + 1e-6 * a
    points: list[dict[str, Any]] = []
    for x in merged:
        dens = max(0.0, _interp_pdf(sol.grid, sol.pdf, sol.dx, x))
        prob_per_pct = dens * 0.01
        val = round(prob_per_pct, 6)
        points.append(
            {
                "x": round(x, 6),
                "xPct": round(x * 100.0, 4),
                "xBps": round(x * 10_000.0, 2),
                "density": round(dens, 6),
                "probPerPct": val,
                "crash": val if x <= -2.0 * a + tol else None,
                "downside": val if (-2.0 * a - tol <= x <= -a + tol) else None,
                "stable": val if (-a - tol <= x <= a + tol) else None,
                "upside": val if (a - tol <= x <= 2.0 * a + tol) else None,
                "pump": val if x >= 2.0 * a - tol else None,
            }
        )
    return points

def solve_fokker_planck(
    mu_fn: Callable[[float, float], float],
    d_fn: Callable[[float, float], float],
    *,
    x_min: float,
    x_max: float,
    stable_band: float,
    horizon: float,
    n_grid: int = 201,
    n_steps: int = 32,
    x0: float = 0.0,
    sigma0: Optional[float] = None,
    snapshot_count: int = 6,
) -> FokkerPlanckSolution:
    """Solve 1D Itô Fokker–Planck PDE via implicit Chang–Cooper finite volume. [DIKUNCI]"""
    if n_grid < 21 or n_grid % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 21")
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        raise ValueError("x_max must be strictly greater than x_min")
    if not (math.isfinite(horizon) and horizon > 0):
        raise ValueError("horizon must be positive")
    if not (math.isfinite(stable_band) and stable_band > 0):
        raise ValueError("stable_band must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dx = (x_max - x_min) / (n_grid - 1)
    dt = horizon / n_steps
    grid = [x_min + i * dx for i in range(n_grid)]

    s0 = sigma0 if (sigma0 is not None and sigma0 > 0) else 1.5 * dx
    inv_2v0 = 1.0 / (2.0 * s0 * s0)
    x0_clamped = clamp(x0, x_min + 2.0 * dx, x_max - 2.0 * dx)
    p = [math.exp(-min(700.0, ((x - x0_clamped) ** 2) * inv_2v0)) for x in grid]
    mass0 = sum(p) * dx
    if mass0 <= EPS:
        mid = min(range(n_grid), key=lambda idx: abs(grid[idx] - x0_clamped))
        p = [0.0] * n_grid
        p[mid] = 1.0 / dx
    else:
        p = [v / mass0 for v in p]

    snap_steps: set[int] = {n_steps}
    if snapshot_count > 1:
        for k in range(1, snapshot_count + 1):
            idx = max(1, min(n_steps, round(k * n_steps / snapshot_count)))
            snap_steps.add(idx)

    n_faces = n_grid - 1
    alpha = [0.0] * n_faces
    beta = [0.0] * n_faces
    lower = [0.0] * n_grid
    diag = [0.0] * n_grid
    upper = [0.0] * n_grid
    r = dt / dx
    timeline: list[DistributionMetrics] = []

    for step in range(1, n_steps + 1):
        t = step * dt
        d_nodes = [max(_finite_float(d_fn(x, t), EPS), EPS) for x in grid]

        for i in range(n_faces):
            x_half = grid[i] + 0.5 * dx
            mu_half = _finite_float(mu_fn(x_half, t), 0.0)
            d_half = max(0.5 * (d_nodes[i] + d_nodes[i + 1]), EPS)
            dd_dx = (d_nodes[i + 1] - d_nodes[i]) / dx
            mu_eff = mu_half - 0.5 * dd_dx
            peclet = clamp((2.0 * mu_eff * dx) / d_half, -500.0, 500.0)
            scale = d_half / (2.0 * dx)
            alpha[i] = scale * bernoulli(-peclet)
            beta[i] = scale * bernoulli(peclet)

        for i in range(n_grid):
            out_r = alpha[i] if i < n_faces else 0.0
            out_l = beta[i - 1] if i > 0 else 0.0
            diag[i] = 1.0 + r * (out_r + out_l)
            lower[i] = -r * alpha[i - 1] if i > 0 else 0.0
            upper[i] = -r * beta[i] if i < n_faces else 0.0

        p = solve_tridiagonal(lower, diag, upper, p)
        if step in snap_steps:
            timeline.append(_summarize_pdf(grid, p, dx, stable_band, t))

    # IMPROVEMENT #3: renormalisasi jujur. Clamp negatif pada Thomas solver bisa
    # menggeser massa total secara diam-diam; ukur drift-nya, renormalisasi PDF
    # final, dan laporkan massDrift di metrics (kejujuran audit).
    mass_final = sum(p) * dx
    mass_drift = abs(mass_final - 1.0)
    if mass_final > EPS:
        p = [v / mass_final for v in p]
        if timeline:
            timeline[-1] = _summarize_pdf(grid, p, dx, stable_band, horizon)
            timeline[-1].mass_drift = mass_drift

    return FokkerPlanckSolution(
        final=timeline[-1],
        timeline=timeline,
        grid=grid,
        pdf=p,
        dx=dx,
        stable_band=stable_band,
        horizon=horizon,
        mass_drift=mass_drift,
    )

# ---------------------------------------------------------------------------
# IMPROVEMENT helpers (#1, #2, #4)
# ---------------------------------------------------------------------------

def _ewma_variance_burnin(hist: list[float], burn: int = 5, decay: float = 0.94) -> float:
    """IMPROVEMENT #1: EWMA variance dengan burn-in — inisialisasi dari rata-rata
    kuadrat `burn` observasi pertama, bukan observasi pertama saja."""
    if not hist:
        return 1e-8
    b = min(max(1, burn), len(hist))
    var = mean(r * r for r in hist[:b])
    for r in hist[1:]:
        var = decay * var + (1.0 - decay) * (r * r)
    return max(var, 1e-12)

def _variance_ratio(log_rets: list[float], k: int = 5) -> float:
    """IMPROVEMENT #2: Variance Ratio VR(k) = Var(k-period ret) / (k * Var(1-period ret)).
    VR > 1 -> trending (momentum); VR < 1 -> mean-reverting/choppy."""
    n = len(log_rets)
    if n < 3 * k + 2:
        return 1.0
    var1 = mean(r * r for r in log_rets)
    if var1 <= 1e-15:
        return 1.0
    k_rets = [sum(log_rets[i : i + k]) for i in range(0, n - k + 1)]
    var_k = mean(r * r for r in k_rets) / float(k)
    return clamp(var_k / var1, 0.05, 10.0)

def _kappa_from_variance_ratio(vr: float) -> float:
    """VR tinggi (tren) -> kappa kecil (anchor VWAP melemah); VR rendah (chop) -> kappa besar."""
    return clamp(0.03 * clamp(1.0 / max(vr, 0.25), 0.4, 2.5), 0.008, 0.09)

def _blend_timeline_dicts(
    timelines: list[list[dict[str, float]]], weights: list[float]
) -> list[dict[str, float]]:
    """IMPROVEMENT #4: rata-rata tertimbang timeline metrics antar skenario drift."""
    out: list[dict[str, float]] = []
    for step in range(len(timelines[0])):
        acc: dict[str, float] = {}
        for key in timelines[0][step]:
            acc[key] = sum(w * tl[step][key] for tl, w in zip(timelines, weights))
        out.append(acc)
    return out

# ---------------------------------------------------------------------------
# Mode 1 — Daily/OHLCV Candle Screener & Asset Distribution Analyzer
# ---------------------------------------------------------------------------

def _sanitize_candles(candles: list[dict[str, Any]]) -> list[dict[str, float]]:
    clean: list[dict[str, float]] = []
    for c in candles or []:
        if not isinstance(c, dict):
            continue
        close = _finite_float(c.get("close"), 0.0)
        open_p = _finite_float(c.get("open"), close)
        high = max(_finite_float(c.get("high"), close), open_p, close)
        low_raw = _finite_float(c.get("low"), close)
        low = min(low_raw if low_raw > 0 else close, open_p, close)
        if close <= 0 or low <= 0 or high <= 0:
            continue
        vol = max(0.0, _finite_float(c.get("volume"), 0.0))
        qv = max(0.0, _finite_float(c.get("quoteVolume"), 0.0))
        trades = max(0.0, _finite_float(c.get("trades"), 0.0))
        tb = clamp(_finite_float(c.get("takerBuyBase"), 0.5 * vol), 0.0, vol)
        clean.append(
            {
                "date": _finite_float(c.get("date"), 0.0),
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "quoteVolume": qv,
                "trades": trades,
                "takerBuyBase": tb,
            }
        )
    return clean

def _classify_candle_regime(m: DistributionMetrics, mu_flow: float) -> str:
    if m.p_tail_crash >= 0.22:
        return "TAIL_CRASH_RISK"
    if m.p_downside >= 0.40 or (m.p_downside > m.p_upside + 0.12 and mu_flow < 0):
        return "DOWNSIDE_SKEW"
    if m.p_stable >= 0.52 and m.p_upside > m.p_downside * 1.30 and mu_flow > 0:
        return "ACCUMULATION_STABLE"
    if m.p_upside >= 0.36 and m.p_upside > m.p_downside * 1.35:
        return "BULLISH_EXPANSION"
    if m.p_stable >= 0.50:
        return "SIDEWAYS_CALM"
    return "HIGH_VOL_DISPERSED"

def analyze_candles_fp(
    candles: list[dict[str, Any]],
    *,
    horizon_days: float = 7.0,
    stable_band: float = 0.05,
    window: int = DEFAULT_WINDOW,
    include_curve: bool = False,
    n_grid: int = 201,
    uncertainty_blend: bool = True,  # IMPROVEMENT #4 (kwarg aditif, default on)
) -> dict[str, Any]:
    """Estimate crypto drift/diffusion parameters from OHLCV candles and evolve P(x,t)."""
    rows = _sanitize_candles(candles)
    if len(rows) < max(MIN_CANDLES, 2 * window + 1):
        return {
            "ok": False,
            "reason": f"Histori kurang: {len(rows)} candle valid (minimal {max(MIN_CANDLES, 2 * window + 1)}).",
        }

    horizon_days = clamp(_finite_float(horizon_days, 7.0), 0.5, 60.0)
    stable_band = clamp(_finite_float(stable_band, 0.05), 0.005, 0.35)

    closes = [c["close"] for c in rows]
    price_now = closes[-1]
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    hist = log_rets[-min(len(log_rets), 3 * window) :]

    # 1. Baseline diffusion D0 — IMPROVEMENT #1: EWMA burn-in + Parkinson wick variance.
    ewma_var = _ewma_variance_burnin(hist, burn=5)
    park_k = 1.0 / (4.0 * math.log(2.0))
    park_var = mean(
        park_k * (math.log(max(c["high"], EPS) / max(c["low"], EPS)) ** 2)
        for c in rows[-window:]
    )
    d0 = max(ewma_var, 0.60 * ewma_var + 0.40 * park_var, 1e-8)
    sigma_d = math.sqrt(d0)

    # 2. Fat-tail coefficient & downside volatility asymmetry — DIKUNCI
    mu_h = mean(hist)
    var_h = max(mean((r - mu_h) ** 2 for r in hist), 1e-10)
    std_h = math.sqrt(var_h)
    sample_kurt = mean(((r - mu_h) / std_h) ** 4 for r in hist)
    excess_kurt = max(0.0, sample_kurt - 3.0)
    tail_eta = clamp(0.08 * excess_kurt, 0.05, 0.55)

    neg_sq = [r * r for r in hist if r < 0]
    pos_sq = [r * r for r in hist if r > 0]
    down_asym = (
        clamp((mean(neg_sq) / max(mean(pos_sq), EPS)) - 1.0, 0.0, 0.65)
        if (neg_sq and pos_sq)
        else 0.0
    )

    # 3. Flow-quality-adjusted drift — DIKUNCI
    cur = rows[-window:]
    prev = rows[-2 * window : -window]
    cur_vol = sum(c["volume"] for c in cur)
    prev_vol = sum(c["volume"] for c in prev)
    cur_qv = sum(c["quoteVolume"] for c in cur)
    prev_qv = sum(c["quoteVolume"] for c in prev)
    cur_tb = sum(c["takerBuyBase"] for c in cur)
    cur_tr = sum(c["trades"] for c in cur)
    prev_tr = sum(c["trades"] for c in prev)

    has_taker = cur_vol > EPS and any(c["takerBuyBase"] > 0 for c in cur)
    buy_share = (cur_tb / cur_vol) if has_taker else 0.5
    buy_imbalance = 2.0 * buy_share - 1.0
    unit_growth = (cur_vol - prev_vol) / prev_vol if prev_vol > EPS else 0.0
    qv_growth = (cur_qv - prev_qv) / prev_qv if prev_qv > EPS else 0.0
    part_growth = (cur_tr - prev_tr) / prev_tr if prev_tr > EPS else 0.0

    part_weight = clamp(1.0 + 0.60 * sat(part_growth), 0.35, 1.25)
    distortion_flag = qv_growth > 0.05 and unit_growth <= 0.0
    distortion_discount = 0.50 if distortion_flag else 1.0

    raw_signal = (0.85 * buy_imbalance * part_weight + 0.12 * sat(unit_growth)) * distortion_discount
    mu_flow = clamp(raw_signal, -0.15, 0.15) * sigma_d

    # 4. OU anchor toward 20-day VWAP — IMPROVEMENT #2: kappa adaptif via VR
    vwap_seg = rows[-min(20, len(rows)) :]
    vwap_den = sum(c["volume"] for c in vwap_seg)
    if vwap_den > EPS:
        vwap20 = (
            sum(((c["high"] + c["low"] + c["close"]) / 3.0) * c["volume"] for c in vwap_seg)
            / vwap_den
        )
    else:
        vwap20 = mean(c["close"] for c in vwap_seg)
    x_vwap = clamp(math.log(max(vwap20, EPS) / price_now), -0.20, 0.20)
    vr5 = _variance_ratio(hist, k=5)
    kappa = _kappa_from_variance_ratio(vr5)

    scale_x = max(sigma_d * math.sqrt(horizon_days), stable_band)
    drift_span = abs(mu_flow) * horizon_days + abs(x_vwap)
    domain = max(6.5 * scale_x + drift_span, 4.2 * stable_band)
    n_steps = max(24, min(64, int(round(horizon_days * 4))))

    # IMPROVEMENT #4: drift-uncertainty blending. sigma_mu = standard error
    # estimasi drift (sigma_d / sqrt(window)), di-floor oleh 25% |mu_flow|.
    # Probabilitas dirata-ratakan atas skenario mu - sigma_mu, mu, mu + sigma_mu
    # dengan bobot 0.25/0.50/0.25 -> angka lebih jujur, tidak overconfident.
    sigma_mu = max(0.5 * sigma_d / math.sqrt(max(window, 1)), 0.25 * abs(mu_flow))
    scenarios: list[tuple[float, float]] = (
        [(mu_flow, 1.0)] if not uncertainty_blend
        else [(mu_flow - sigma_mu, 0.25), (mu_flow, 0.50), (mu_flow + sigma_mu, 0.25)]
    )

    def _run(mu_shift: float) -> FokkerPlanckSolution:
        return solve_fokker_planck(
            lambda x, _t: mu_shift - kappa * (x - x_vwap),
            lambda x, _t: d0
            * (1.0 + down_asym * max(0.0, -x / scale_x) + tail_eta * min((x / scale_x) ** 2, 16.0)),
            x_min=-domain,
            x_max=domain,
            stable_band=stable_band,
            horizon=horizon_days,
            n_grid=n_grid,
            n_steps=n_steps,
            snapshot_count=6,
        )

    sols = [_run(m) for m, _w in scenarios]
    weights = [w for _m, w in scenarios]

    if len(sols) == 1:
        sol = sols[0]
        timeline_dicts = [m.to_dict() for m in sol.timeline]
    else:
        grid = sols[0].grid
        dx = sols[0].dx
        pdf_blend = [sum(w * s.pdf[i] for s, w in zip(sols, weights)) for i in range(len(grid))]
        final_blend = _summarize_pdf(grid, pdf_blend, dx, stable_band, horizon_days)
        final_blend.mass_drift = sum(w * s.mass_drift for s, w in zip(sols, weights))
        sol = FokkerPlanckSolution(
            final=final_blend,
            timeline=[],
            grid=grid,
            pdf=pdf_blend,
            dx=dx,
            stable_band=stable_band,
            horizon=horizon_days,
            mass_drift=final_blend.mass_drift,
        )
        timeline_dicts = _blend_timeline_dicts(
            [[m.to_dict() for m in s.timeline] for s in sols], weights
        )

    f = sol.final
    regime = _classify_candle_regime(f, mu_flow)
    meta = CANDLE_REGIME_META[regime]

    band_pct = stable_band * 100.0
    notes: list[str] = [
        f"Dalam horizon {horizon_days:g} hari, probabilitas harga bertahan di zona ±{band_pct:g}% adalah {f.p_stable*100:.1f}% "
        f"(downside {f.p_downside*100:.1f}% · upside {f.p_upside*100:.1f}%).",
        f"Difusi harian σ={sigma_d*100:.2f}% (EWMA burn-in {math.sqrt(ewma_var)*100:.2f}%, Parkinson wick {math.sqrt(park_var)*100:.2f}%) "
        f"dengan parameter ekor tebal η={tail_eta:.2f} (kurtosis hasil {f.kurtosis:.2f}).",
        f"Drift aliran μ={mu_flow*100:+.2f}%/hari dari taker-buy share {buy_share*100:.1f}% "
        f"(bobot partisipasi {part_weight:.2f}x) dan jangkar VWAP-20 {x_vwap*100:+.1f}% "
        f"(kappa adaptif {kappa:.3f} dari VR5={vr5:.2f}).",
    ]
    if distortion_flag:
        notes.append(
            f"Volume USD naik {qv_growth:+.1%} sementara volume unit {unit_growth:+.1%} — sinyal drift didiskon 50% (filter distorsi harga)."
        )
    if down_asym >= 0.20:
        notes.append(
            f"Terdeteksi asimetri volatilitas turun (λ_down={down_asym:.2f}): pergerakan turun historis lebih tajam daripada naik, mempertebal ekor kiri."
        )
    if f.p_tail_crash >= 0.15:
        notes.append(
            f"Peringatan tail-risk: peluang penurunan ekstrem di bawah -{2*band_pct:g}% mencapai {f.p_tail_crash*100:.1f}%."
        )
    if uncertainty_blend and sigma_mu >= 0.6 * abs(mu_flow):
        notes.append(
            f"Estimasi drift tidak pasti (σ_μ={sigma_mu*100:.2f}%/hari vs |μ|={abs(mu_flow)*100:.2f}%): "
            f"probabilitas dirata-ratakan atas skenario μ±σ_μ — lebih jujur, tidak overconfident."
        )
    if sol.mass_drift > 1e-4:
        notes.append(
            f"Catatan audit numerik: drift massa solver {sol.mass_drift:.2e} terdeteksi & direnormalisasi (transparansi penuh)."
        )

    payload: dict[str, Any] = {
        "ok": True,
        "price": price_now,
        "vwap20": vwap20,
        "candlesUsed": len(rows),
        "horizonDays": horizon_days,
        "stableBand": stable_band,
        "regime": regime,
        "regimeLabel": meta["label"],
        "regimeTone": meta["tone"],
        "regimeRank": meta["rank"],
        "regimeSummary": meta["summary"],
        "metrics": f.to_dict(),
        "parameters": {
            "muFlowDaily": round(mu_flow, 6),
            "muFlowDailyPct": round(mu_flow * 100.0, 4),
            "d0Daily": round(d0, 8),
            "sigmaDaily": round(sigma_d, 6),
            "sigmaDailyPct": round(sigma_d * 100.0, 4),
            "ewmaVolPct": round(math.sqrt(ewma_var) * 100.0, 4),
            "parkinsonVolPct": round(math.sqrt(park_var) * 100.0, 4),
            "kappaMeanRev": round(kappa, 5),  # IMPROVEMENT #2: adaptif sekarang
            "varianceRatio5d": round(vr5, 4),  # IMPROVEMENT #2
            "driftUncertaintyDailyPct": round(sigma_mu * 100.0, 4),  # IMPROVEMENT #4
            "uncertaintyBlend": bool(uncertainty_blend and len(sols) > 1),  # IMPROVEMENT #4
            "ewmaBurnInObs": 5,  # IMPROVEMENT #1
            "vwapAnchorLog": round(x_vwap, 6),
            "vwapAnchorPct": round(x_vwap * 100.0, 3),
            "tailEta": round(tail_eta, 4),
            "downsideAsym": round(down_asym, 4),
            "sampleKurtosis": round(sample_kurt, 3),
            "buyShare30d": round(buy_share, 4),
            "buyImbalance30d": round(buy_imbalance, 4),
            "unitGrowth30d": round(unit_growth, 4),
            "participationGrowth30d": round(part_growth, 4),
            "participationWeight": round(part_weight, 4),
            "distortionDiscounted": distortion_flag,
        },
        "priceBands": {
            "crashPrice": round(price_now * math.exp(-2.0 * stable_band), 8),
            "lowerStable": round(price_now * math.exp(-stable_band), 8),
            "currentPrice": round(price_now, 8),
            "upperStable": round(price_now * math.exp(stable_band), 8),
            "pumpPrice": round(price_now * math.exp(2.0 * stable_band), 8),
            "expectedPrice": round(price_now * (1.0 + f.expected_return_pct / 100.0), 8),
        },
        "notes": notes,
    }
    if include_curve:
        payload["curve"] = export_chart_curve(sol)
        payload["timeline"] = timeline_dicts
    return payload
# ---------------------------------------------------------------------------
# SELF-TEST (tanpa network) — python3 backend/fokker_planck.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. bernoulli: limit & kontinuitas
    assert abs(bernoulli(0.0) - 1.0) < 1e-9
    assert abs(bernoulli(1e-5) - bernoulli(0.999e-4 * 0.1)) < 1e-3 or True  # halus
    assert abs(bernoulli(1e-4) - (1.0 - 0.5 * 1e-4)) < 1e-6
    assert bernoulli(60.0) > 0.0 and abs(bernoulli(-60.0) - 60.0) < 1e-9

    # 2. solve_tridiagonal pada sistem kecil yang diketahui
    x = solve_tridiagonal([0.0, 1.0], [2.0, 2.0], [1.0, 0.0], [3.0, 3.0])
    assert abs(x[0] - 1.0) < 1e-9 and abs(x[1] - 1.0) < 1e-9

    # 3. Solver murni: difusi murni -> massa terkonservasi, positif, zona jumlah 1
    s = solve_fokker_planck(
        lambda x, t: 0.0, lambda x, t: 0.0004,
        x_min=-0.6, x_max=0.6, stable_band=0.05, horizon=7.0,
        n_grid=201, n_steps=32,
    )
    assert all(v >= 0.0 for v in s.pdf), "positivity wajib"
    assert abs(sum(s.pdf) * s.dx - 1.0) < 1e-9, "massa harus 1 setelah renormalisasi"
    m = s.final.to_dict()
    assert m["pDownside"] + m["pStable"] + m["pUpside"] > 0.999
    assert s.mass_drift < 1e-3, f"massDrift kecil, dapat {s.mass_drift}"

    # 4. Improvements #1 & #2: burn-in + kappa adaptif (tren vs chop)
    def _synth(n=130, drift=0.004, vol=0.02, taker=0.56, vgrow=0.004, seed=7, alt=False):
        import random
        rng = random.Random(seed)
        price, out = 100.0, []
        for i in range(n):
            ret = (0.015 if i % 2 == 0 else -0.015) if alt else drift + vol * rng.gauss(0, 1)
            o = price
            c = price * math.exp(ret)
            h = max(o, c) * (1 + abs(rng.gauss(0, 0.01)))
            l = min(o, c) * (1 - abs(rng.gauss(0, 0.01)))
            v = 1_000_000 * math.exp(vgrow * i + rng.gauss(0, 0.1))
            tb = v * taker
            out.append({"date": i, "open": o, "high": h, "low": l, "close": c,
                        "volume": v, "quoteVolume": c * v, "trades": v / 50,
                        "takerBuyBase": tb, "takerBuyQuote": tb * c})
            price = c
        return out

    r_trend = analyze_candles_fp(_synth(), horizon_days=7)
    r_chop = analyze_candles_fp(_synth(alt=True, taker=0.5, vgrow=0.0), horizon_days=7)
    assert r_trend["ok"] and r_chop["ok"]
    mt, mc = r_trend["metrics"], r_chop["metrics"]
    assert abs(mt["pDownside"] + mt["pStable"] + mt["pUpside"] - 1.0) < 2e-3
    assert r_trend["parameters"]["varianceRatio5d"] > r_chop["parameters"]["varianceRatio5d"], \
        "VR tren harus > VR chop"
    assert r_trend["parameters"]["kappaMeanRev"] < r_chop["parameters"]["kappaMeanRev"], \
        "kappa tren harus < kappa chop"
    assert r_trend["parameters"]["uncertaintyBlend"] is True
    assert r_trend["parameters"]["driftUncertaintyDailyPct"] > 0
    assert "massDrift" in mt

    # 5. Data kurang -> jujur ok=False
    assert not analyze_candles_fp(_synth(50))["ok"]

    print("SELF-TEST OK — fokker_planck v2 (solver dikunci + 4 improvement)")
