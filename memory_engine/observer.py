"""메모리 셀 Observer — Ω 5레이어 관측 + 판정 + 진단.

Ω 공식 (DRAM)
─────────────
  Ω_retention : cell.q                                   — 전하 보존
  Ω_margin    : clamp(bl_swing / (bl_swing_max) → [0,1]) — 읽기 마진
  Ω_endurance : 1 − n_cycles / max_cycles                — 내구성 여유
  Ω_speed     : 1 − t_access / t_access_ref              — 접근 속도
  Ω_power     : 1 − power / power_ref                    — 소비전력 효율

  Ω_global = 0.30·Ω_ret + 0.25·Ω_margin + 0.20·Ω_end + 0.15·Ω_speed + 0.10·Ω_power

Ω 공식 (SRAM)
─────────────
  Ω_retention : |v_q − v_qb| / Vdd                       — 노드 전압차
  Ω_margin    : SNM / (Vdd/2)                             — 안정성 마진
  Ω_endurance : 1 − n_cycles / max_cycles                — 내구성 여유
  Ω_speed     : 1 − t_access / t_access_ref              — 접근 속도
  Ω_power     : 1 − leakage_nA / leakage_ref             — 누설 전력 효율

판정:
  HEALTHY  (Ω ≥ 0.75)
  STABLE   (0.55 ≤ Ω < 0.75)
  FRAGILE  (0.35 ≤ Ω < 0.55)
  CRITICAL (Ω < 0.35)
"""

from __future__ import annotations

from typing import List

from .schema import DRAMCellState, DRAMDesignParams, MemoryObservation, SRAMCellState, SRAMDesignParams
from .dram_physics import bitline_swing_fraction, read_margin as _dram_read_margin
from .sram_physics import static_noise_margin, read_noise_margin, write_margin, stability_index

# ── 기준값 상수 ──────────────────────────────────────────────────────────────
_DRAM_T_ACCESS_REF_NS: float = 20.0   # 느린 DRAM 기준 접근 시간 [ns]
_DRAM_POWER_REF_MW: float = 1e-5       # 소비전력 기준 [mW/cell]

_SRAM_T_ACCESS_REF_NS: float = 2.0    # 느린 SRAM 기준 접근 시간 [ns]
_SRAM_LEAK_REF_NA: float = 10.0       # 누설 전류 기준 [nA/cell]

# ── 판정 임계 ────────────────────────────────────────────────────────────────
_VERDICTS: list[tuple[float, str]] = [
    (0.75, "HEALTHY"),
    (0.55, "STABLE"),
    (0.35, "FRAGILE"),
]


def _verdict(omega: float) -> str:
    for thr, v in _VERDICTS:
        if omega >= thr:
            return v
    return "CRITICAL"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# ══════════════════════════════════════════════════════════════════════════════
# DRAM Observer
# ══════════════════════════════════════════════════════════════════════════════

def observe_dram(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> MemoryObservation:
    """DRAM 셀 Ω 5레이어 관측.

    Args:
        cell  : 현재 DRAM 셀 상태.
        params: DRAM 설계 파라미터.

    Returns:
        MemoryObservation.
    """
    # ── Ω_retention: 전하 보존율 ─────────────────────────────────────────
    omega_ret = _clamp(cell.q)

    # ── Ω_margin: 읽기 마진 ───────────────────────────────────────────────
    bl = bitline_swing_fraction(cell, params)
    # bl_max = Cs/(Cs+Cbl) (완전 충전 시)
    cs = max(1e-9, params.C_s_fF)
    cbl = max(1e-9, params.C_bl_fF)
    bl_max = cs / (cs + cbl)
    omega_margin = _clamp(bl / max(1e-9, bl_max))

    # ── Ω_endurance: 내구성 여유 ─────────────────────────────────────────
    cycle_ratio = cell.n_cycles / max(1, params.max_cycles)
    omega_end = _clamp(1.0 - cycle_ratio)

    # ── Ω_speed: 접근 속도 효율 ──────────────────────────────────────────
    omega_speed = _clamp(1.0 - params.t_access_ns / _DRAM_T_ACCESS_REF_NS)

    # ── Ω_power: 소비전력 효율 ───────────────────────────────────────────
    omega_power = _clamp(1.0 - params.power_mW_per_cell / _DRAM_POWER_REF_MW)

    # ── Ω_global ─────────────────────────────────────────────────────────
    omega = (
        0.30 * omega_ret
        + 0.25 * omega_margin
        + 0.20 * omega_end
        + 0.15 * omega_speed
        + 0.10 * omega_power
    )
    omega = _clamp(omega)

    # ── 플래그 ───────────────────────────────────────────────────────────
    flags: List[str] = []
    rm = _dram_read_margin(cell, params)
    if cell.q < 0.20:
        flags.append("critical_charge_loss")
    elif cell.q < 0.50:
        flags.append("low_charge")
    if rm < 0.0:
        flags.append("read_margin_fail")
    elif rm < 0.02:
        flags.append("read_margin_marginal")
    if cycle_ratio > 0.90:
        flags.append("endurance_critical")
    elif cycle_ratio > 0.70:
        flags.append("endurance_warning")
    if cell.n_reads > 10_000:
        flags.append("high_read_disturb_risk")

    return MemoryObservation(
        omega_global=round(omega, 6),
        verdict=_verdict(omega),
        omega_retention=round(omega_ret, 6),
        omega_margin=round(omega_margin, 6),
        omega_endurance=round(omega_end, 6),
        omega_speed=round(omega_speed, 6),
        omega_power=round(omega_power, 6),
        flags=flags,
        notes="; ".join(flags),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SRAM Observer
# ══════════════════════════════════════════════════════════════════════════════

def observe_sram(
    cell: SRAMCellState,
    params: SRAMDesignParams,
) -> MemoryObservation:
    """SRAM 셀 Ω 5레이어 관측.

    Args:
        cell  : 현재 SRAM 셀 상태.
        params: SRAM 설계 파라미터.

    Returns:
        MemoryObservation.
    """
    # ── Ω_retention: 노드 전압차 정규화 ─────────────────────────────────
    v_diff = abs(cell.v_q - cell.v_qb)
    omega_ret = _clamp(v_diff / max(1e-9, params.vdd_v))

    # ── Ω_margin: SNM 기반 안정성 마진 ───────────────────────────────────
    snm = static_noise_margin(params)
    snm_ref = params.vdd_v * 0.5
    omega_margin = _clamp(snm / max(1e-9, snm_ref))

    # ── Ω_endurance: 내구성 여유 (SRAM은 매우 큼) ────────────────────────
    cycle_ratio = cell.n_cycles / max(1, params.max_cycles)
    omega_end = _clamp(1.0 - cycle_ratio)

    # ── Ω_speed: 접근 속도 효율 ──────────────────────────────────────────
    omega_speed = _clamp(1.0 - params.t_access_ns / _SRAM_T_ACCESS_REF_NS)

    # ── Ω_power: 누설 전류 기반 전력 효율 ────────────────────────────────
    omega_power = _clamp(1.0 - params.leakage_nA / _SRAM_LEAK_REF_NA)

    # ── Ω_global ─────────────────────────────────────────────────────────
    omega = (
        0.30 * omega_ret
        + 0.25 * omega_margin
        + 0.20 * omega_end
        + 0.15 * omega_speed
        + 0.10 * omega_power
    )
    omega = _clamp(omega)

    # ── 플래그 ───────────────────────────────────────────────────────────
    flags: List[str] = []
    rnm = read_noise_margin(params)
    wm = write_margin(params)
    si = stability_index(params)

    if snm < 0.03:
        flags.append("snm_critical")
    elif snm < 0.06:
        flags.append("snm_low")
    if rnm < 0.02:
        flags.append("read_margin_fail")
    elif rnm < 0.04:
        flags.append("read_margin_marginal")
    if wm < 0.03:
        flags.append("write_margin_fail")
    elif wm < 0.06:
        flags.append("write_margin_marginal")
    if si < 0.30:
        flags.append("cell_unstable")
    if v_diff < params.vdd_v * 0.20:
        flags.append("weak_cell_state")

    return MemoryObservation(
        omega_global=round(omega, 6),
        verdict=_verdict(omega),
        omega_retention=round(omega_ret, 6),
        omega_margin=round(omega_margin, 6),
        omega_endurance=round(omega_end, 6),
        omega_speed=round(omega_speed, 6),
        omega_power=round(omega_power, 6),
        flags=flags,
        notes="; ".join(flags),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 공통 진단
# ══════════════════════════════════════════════════════════════════════════════

def diagnose(obs: MemoryObservation) -> List[str]:
    """관측 결과 기반 설계 개선 권고 생성.

    Args:
        obs: MemoryObservation 결과.

    Returns:
        권고 문자열 리스트.
    """
    advice: List[str] = []

    if "critical_charge_loss" in obs.flags or "low_charge" in obs.flags:
        advice.append("Refresh 간격 단축 또는 저장 커패시터(Cs) 증가를 검토하세요.")
    if "read_margin_fail" in obs.flags:
        advice.append("센스앰프 감지 임계 완화 또는 Cs/Cbl 비율 개선이 필요합니다.")
    if "read_margin_marginal" in obs.flags:
        advice.append("읽기 마진이 임계 근처입니다. 공정 변동 여유를 확인하세요.")
    if "endurance_critical" in obs.flags:
        advice.append("셀 내구성 한계에 근접했습니다. 마모 평준화(wear leveling) 적용을 권장합니다.")
    if "snm_critical" in obs.flags or "cell_unstable" in obs.flags:
        advice.append("SNM 위험 — beta_ratio 증가 또는 Vdd 상향이 필요합니다.")
    if "write_margin_fail" in obs.flags:
        advice.append("쓰기 마진 부족 — wl_strength 증가 또는 beta_ratio 감소를 검토하세요.")
    if "high_read_disturb_risk" in obs.flags:
        advice.append("읽기 교란 누적이 높습니다. Periodic refresh 또는 read scrub을 적용하세요.")

    if not advice:
        advice.append(f"현재 상태는 모든 마진 기준을 충족합니다 (Ω = {obs.omega_global:.3f}, {obs.verdict}).")

    return advice
