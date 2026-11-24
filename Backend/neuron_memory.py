"""
Meridian — Neuron Memory v1 (hippocampus + cortical patterns, modul ISOLASI)
============================================================================

Peran biologis:
- Hippocampus : menyimpan EPISODE (pengalaman konkret) — kondisi dendrite saat
                prediksi dibuat, prediksi neuron, dan realized return saat
                episode ditutup. Ini bahan belajar guru offline (refit).
- Cortex      : PATTERN — signature dendrite terkuantisasi yang diakumulasi
                lintas episode. Pattern baru TIDAK dipercaya sebelum
                terkonsolidasi: minimal PATTERN_MIN_EPISODES episode DAN
                Wilson lower bound success rate >= PATTERN_MIN_WILSON.

Aturan konsolidasi (desain sesi sebelumnya):
  "Pattern hanya dipercaya setelah >= 10 episode sukses (Wilson lower bound)."

Vercel-safe: murni in-memory + serialisasi JSON penuh (to_dict/from_dict).
Persistensi nyata (localStorage sync / API / Supabase) di wiring route.

Murni komputasi, tanpa I/O — deterministik, bisa unit-test.
Self-test:  python3 backend/neuron_memory.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Any, Optional
from uuid import uuid4

MEMORY_VERSION = "neuron-memory-v1.0"

# --- Konfigurasi konsolidasi -------------------------------------------------
MAX_EPISODES = 2000          # ring buffer (localStorage/JSON friendly)
PATTERN_MIN_EPISODES = 10    # minimal episode sebelum pattern bisa dipercaya
PATTERN_MIN_WILSON = 0.55    # Wilson LB minimal (10/10 sukses = 0.722 -> lolos;
                             # 9/10 = 0.596 -> lolos; 8/10 = 0.490 -> GAGAL)
Z95 = 1.96
PATTERN_BIN = 0.34           # kuantisasi dendrite [-1,+1] -> bin -3..+3
NEUTRAL_Y = 0.05             # |y| < ini -> episode netral, tidak masuk pattern

def wilson_lower_bound(successes: int, n: int, z: float = Z95) -> float:
    """Wilson score lower bound 95% — ukuran kepercayaan konservatif."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def pattern_signature(dendrites: dict[str, float]) -> str:
    """Dendrite [-1,+1] -> signature terkuantisasi (kunci pattern cortex)."""
    if not dendrites:
        return ""
    return "|".join(
        f"{k}:{int(round(max(-1.0, min(1.0, float(v))) / PATTERN_BIN))}"
        for k, v in sorted(dendrites.items())
    )

class NeuronMemory:
    """Hippocampus (episode) + cortex (pattern trust). JSON-serializable."""

    def __init__(self, state: Optional[dict[str, Any]] = None,
                 max_episodes: Optional[int] = None) -> None:
        self.max_episodes = int(max_episodes or MAX_EPISODES)
        self.episodes: list[dict[str, Any]] = []
        self.patterns: dict[str, dict[str, Any]] = {}
        if state:
            self._load(state)

    # -- hippocampus: episode --------------------------------------------------
    def record_episode(self, symbol: str, dendrites: dict[str, float],
                       y: float, verdict: str) -> dict[str, Any]:
        ep = {
            "id": uuid4().hex[:10],
            "ts": _now(),
            "symbol": symbol.upper().strip(),
            "signature": pattern_signature(dendrites),
            "dendrites": {k: round(float(v), 4) for k, v in dendrites.items()},
            "y": round(float(y), 4),
            "verdict": verdict,
            "status": "PENDING",
            "realizedReturn": None,
            "success": None,
        }
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
        return ep

    def close_episode(self, episode_id: str, realized_return: float) -> dict[str, Any]:
        """Outcome datang -> episode ditutup, pattern cortex diperbarui."""
        for ep in reversed(self.episodes):
            if ep["id"] == episode_id and ep["status"] == "PENDING":
                ep["realizedReturn"] = round(float(realized_return), 6)
                y = float(ep["y"])
                if abs(y) < NEUTRAL_Y:
                    ep["success"] = None      # netral: tidak mengotori pattern
                else:
                    ep["success"] = (y * float(realized_return)) > 0.0
                    if ep["signature"]:
                        self._note_pattern(ep["signature"], ep["success"],
                                           float(realized_return))
                ep["status"] = "CLOSED"
                return ep
        raise KeyError(f"Episode {episode_id} tidak ditemukan / sudah ditutup.")

    def _note_pattern(self, signature: str, success: bool,
                      realized_return: float) -> None:
        pat = self.patterns.setdefault(signature, {
            "n": 0, "successes": 0, "sumReturn": 0.0, "lastSeen": None,
        })
        pat["n"] += 1
        pat["successes"] += 1 if success else 0
        pat["sumReturn"] += realized_return
        pat["lastSeen"] = _now()

    # -- cortex: pattern trust ---------------------------------------------------
    def pattern_trust(self, dendrites: dict[str, float]) -> dict[str, Any]:
        sig = pattern_signature(dendrites)
        pat = self.patterns.get(sig)
        if not pat or pat["n"] < PATTERN_MIN_EPISODES:
            return {"signature": sig, "trusted": False,
                    "n": pat["n"] if pat else 0,
                    "successes": pat["successes"] if pat else 0,
                    "successRate": None, "wilsonLB": None,
                    "reason": f"butuh >= {PATTERN_MIN_EPISODES} episode"}
        rate = pat["successes"] / pat["n"]
        lb = wilson_lower_bound(pat["successes"], pat["n"])
        trusted = lb >= PATTERN_MIN_WILSON
        return {"signature": sig, "trusted": trusted, "n": pat["n"],
                "successes": pat["successes"], "successRate": round(rate, 4),
                "wilsonLB": round(lb, 4),
                "avgReturn": round(pat["sumReturn"] / pat["n"], 6),
                "reason": "terkonsolidasi" if trusted
                          else "wilson LB di bawah ambang"}

    def trusted_patterns(self) -> list[dict[str, Any]]:
        out = []
        for sig, pat in self.patterns.items():
            if pat["n"] >= PATTERN_MIN_EPISODES:
                lb = wilson_lower_bound(pat["successes"], pat["n"])
                if lb >= PATTERN_MIN_WILSON:
                    out.append({"signature": sig, "n": pat["n"],
                                "successes": pat["successes"],
                                "wilsonLB": round(lb, 4),
                                "avgReturn": round(pat["sumReturn"] / pat["n"], 6)})
        return out

    # -- bahan guru offline (refit) ----------------------------------------------
    def closed_episodes(self) -> list[dict[str, Any]]:
        return [e for e in self.episodes if e["status"] == "CLOSED"
                and e["success"] is not None]

    def stats(self) -> dict[str, Any]:
        closed = [e for e in self.episodes if e["status"] == "CLOSED"]
        scored = [e for e in closed if e["success"] is not None]
        wins = sum(1 for e in scored if e["success"])
        return {
            "episodes": len(self.episodes),
            "pending": len(self.episodes) - len(closed),
            "closed": len(closed),
            "scored": len(scored),
            "winRate": round(wins / len(scored), 4) if scored else None,
            "patterns": len(self.patterns),
            "trustedPatterns": len(self.trusted_patterns()),
        }

    # -- serialisasi (Vercel ephemeral-safe) ---------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": MEMORY_VERSION, "episodes": list(self.episodes),
                "patterns": {k: dict(v) for k, v in self.patterns.items()},
                "maxEpisodes": self.max_episodes}

    def _load(self, state: dict[str, Any]) -> None:
        self.episodes = list(state.get("episodes") or [])
        self.patterns = {k: dict(v) for k, v in (state.get("patterns") or {}).items()}
        self.max_episodes = int(state.get("maxEpisodes") or MAX_EPISODES)

    @classmethod
    def from_dict(cls, state: Optional[dict[str, Any]],
                  max_episodes: Optional[int] = None) -> "NeuronMemory":
        return cls(state=state, max_episodes=max_episodes)

# ---------------------------------------------------------------------------
# Self-test (deterministik, tanpa I/O)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # 1) Matematika Wilson benar (10/10=0.72, 7/10=0.40, 0/10=0)
    assert abs(wilson_lower_bound(10, 10) - 0.7225) < 0.01, "1a"
    assert wilson_lower_bound(0, 10) == 0.0, "1b"
    assert abs(wilson_lower_bound(7, 10) - 0.3968) < 0.01, "1c"
    assert wilson_lower_bound(5, 0) == 0.0, "1d"

    # 2) Signature stabil & urutan tak penting
    d = {"screenSignal": 0.8, "rvol": 0.1, "vwapDist": -0.9}
    assert pattern_signature(d) == pattern_signature(
        dict(reversed(list(d.items())))), "2a"
    assert pattern_signature({}) == "", "2b"

    # 3) Episode: record -> close -> success dihitung benar
    m = NeuronMemory()
    ep = m.record_episode("BTCUSDT", d, y=0.5, verdict="LONG")
    assert ep["status"] == "PENDING", "3a"
    assert m.close_episode(ep["id"], 0.04)["success"] is True, "3b"
    ep2 = m.record_episode("ETHUSDT", d, y=0.5, verdict="LONG")
    assert m.close_episode(ep2["id"], -0.02)["success"] is False, "3c"
    ep3 = m.record_episode("SOLUSDT", d, y=0.01, verdict="NEUTRAL")
    assert m.close_episode(ep3["id"], 0.09)["success"] is None, "3d"

    # 4) Pattern belum dipercaya sebelum 10 episode
    t = m.pattern_trust(d)
    assert t["trusted"] is False and t["n"] == 2, "4"

    # 5) Konsolidasi: 10 episode sukses -> Wilson 0.72 >= 0.55 -> TRUSTED
    m2 = NeuronMemory()
    for _ in range(10):
        e = m2.record_episode("BTCUSDT", d, y=0.6, verdict="LONG")
        m2.close_episode(e["id"], 0.03)
    t2 = m2.pattern_trust(d)
    assert t2["trusted"] is True and t2["n"] == 10 and t2["wilsonLB"] > 0.55, f"5: {t2}"

    # 6) 8/10 sukses -> Wilson 0.49 < 0.55 -> TIDAK dipercaya (konservatif)
    m3 = NeuronMemory()
    for i in range(10):
        e = m3.record_episode("BTCUSDT", d, y=0.6, verdict="LONG")
        m3.close_episode(e["id"], 0.03 if i < 8 else -0.03)
    t3 = m3.pattern_trust(d)
    assert t3["trusted"] is False and t3["successes"] == 8, f"6: {t3}"

    # 7) Ring buffer: cap kecil -> episode lama dibuang
    m4 = NeuronMemory(max_episodes=5)
    for _ in range(8):
        m4.record_episode("X", d, y=0.5, verdict="LONG")
    assert len(m4.episodes) == 5, "7"

    # 8) Serialisasi roundtrip identik
    snap = m2.to_dict()
    m5 = NeuronMemory.from_dict(snap)
    assert m5.pattern_trust(d)["trusted"] is True, "8a"
    assert m5.stats() == m2.stats(), "8b"

    # 9) closed_episodes siap untuk guru offline (refit)
    refit = m2.closed_episodes()
    assert len(refit) == 10 and all(e["realizedReturn"] is not None for e in refit), "9"

    print("SELF-TEST OK — neuron_memory (episode + Wilson consolidation + serialisasi)")

if __name__ == "__main__":
    _self_test()
