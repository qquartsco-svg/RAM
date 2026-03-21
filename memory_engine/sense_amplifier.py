"""DRAM 센스앰프(Sense Amplifier) 모델 — differential latch 기반.

물리 동작 순서
──────────────
  1. Precharge (EQ on):
       BL = BLB = Vdd/2

  2. Equalize off → Word Line enable:
       저장 커패시터(Cs)가 비트라인(Cbl)에 전하 공유.
       ΔV = q · Vdd · Cs / (Cs + Cbl)

       BL  = Vdd/2 + ΔV   (저장 '1' 읽기 시)
       BLB = Vdd/2         (상보 비트라인)

  3. SA Latch 동작:
       교차결합 인버터 쌍이 ΔV를 증폭 → full swing (0 / Vdd).
       조건: ΔV > sa_offset_v   (오프셋보다 신호가 커야 정확히 감지)

  4. Restore:
       증폭된 Vdd/0가 WL을 통해 커패시터를 재충전.

핵심 파라미터
──────────────
  SA Margin = ΔV − sa_offset_v
    양수 → 정확 감지 (read_success=True)
    음수 → 비트 오류 (read_success=False)

  BER(Bit Error Rate) — 오프셋이 정규분포 N(0, σ)를 따를 때:
    P(error) = Q(ΔV / σ) = 0.5 · erfc(ΔV / (σ·√2))

수정 이력:
  v0.2.0 — 신규.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .schema import DRAMCellState, DRAMDesignParams, SAObservation, SAParams
from .dram_physics import bitline_swing_fraction


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _q_function(x: float) -> float:
    """Q(x) = 0.5 · erfc(x / √2) — 정규 분포 우측 꼬리 확률."""
    if x >= 40.0:
        return 0.0
    if x <= -40.0:
        return 1.0
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _sigma(sa_params: SAParams) -> float:
    """오프셋 표준편차 σ [V]. None이면 sa_offset_v/3 사용."""
    if sa_params.sigma_offset_v is not None:
        return max(1e-9, float(sa_params.sigma_offset_v))
    return max(1e-9, sa_params.sa_offset_v / 3.0)


# ══════════════════════════════════════════════════════════════════════════════
# 핵심 함수
# ══════════════════════════════════════════════════════════════════════════════

def sense_op(
    cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
) -> SAObservation:
    """센스앰프 1회 동작 시뮬레이션.

    단계:
      1. ΔV = q · Vdd · Cs/(Cs+Cbl)  계산.
      2. SA 마진 = ΔV − sa_offset_v.
      3. BER = Q(ΔV / σ).
      4. ω_SA = clamp(ΔV / (Vdd/2)).

    Args:
        cell       : 현재 DRAM 셀 상태.
        dram_params: DRAM 설계 파라미터.
        sa_params  : 센스앰프 파라미터.

    Returns:
        SAObservation.
    """
    # ── ΔV 계산 ───────────────────────────────────────────────────────────
    bl_frac = bitline_swing_fraction(cell, dram_params)
    delta_v = bl_frac * dram_params.vdd_v      # ΔV [V]

    # ── SA 마진 ───────────────────────────────────────────────────────────
    sa_margin = delta_v - sa_params.sa_offset_v
    read_success = sa_margin > 0.0

    # ── BER (Q-function 기반) ─────────────────────────────────────────────
    sigma = _sigma(sa_params)
    ber = _q_function(delta_v / sigma)

    # ── ω_SA ──────────────────────────────────────────────────────────────
    half_vdd = max(1e-9, dram_params.vdd_v / 2.0)
    omega_sa = max(0.0, min(1.0, delta_v / half_vdd))

    # ── 접근 시간 ─────────────────────────────────────────────────────────
    t_total = sa_params.t_sense_ns + sa_params.t_restore_ns

    # ── 플래그 ────────────────────────────────────────────────────────────
    flags: List[str] = []
    if not read_success:
        flags.append("sa_bit_error")
    elif delta_v < sa_params.sa_sensitivity_v:
        flags.append("sa_margin_marginal")

    return SAObservation(
        delta_v=round(delta_v, 7),
        sa_margin_v=round(sa_margin, 7),
        read_success=read_success,
        ber=ber,
        omega_sa=round(omega_sa, 6),
        t_total_ns=t_total,
        flags=flags,
    )


def sa_bit_error_rate(
    cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
) -> float:
    """비트 오류율 BER = Q(ΔV / σ).

    오프셋 분포: N(0, σ), σ = sa_params.sigma_offset_v (또는 sa_offset_v/3).

    Returns:
        BER ∈ [0, 0.5].
    """
    bl_frac = bitline_swing_fraction(cell, dram_params)
    delta_v = bl_frac * dram_params.vdd_v
    sigma = _sigma(sa_params)
    return _q_function(delta_v / sigma)


def sa_row_ber(
    cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
) -> float:
    """행(row) 단위 오류율 = 1 − (1 − BER_bit)^n_bits_per_row.

    한 row 내 임의 비트라도 오류가 발생할 확률.
    """
    ber_bit = sa_bit_error_rate(cell, dram_params, sa_params)
    n = sa_params.n_bits_per_row
    return 1.0 - (1.0 - ber_bit) ** n


def sa_min_delta_v_for_ber(
    target_ber: float,
    sa_params: SAParams,
) -> float:
    """목표 BER을 달성하기 위한 최소 ΔV [V].

       BER = Q(ΔV/σ) = target_ber
       → ΔV = σ · Q⁻¹(target_ber)

    Q⁻¹(x) ≈ √2 · erfinv(1 − 2x)  (scipy 없이 bisection으로 해결)

    Returns:
        최소 ΔV [V].
    """
    sigma = _sigma(sa_params)
    if target_ber <= 0.0:
        return float("inf")
    if target_ber >= 0.5:
        return 0.0

    # bisection으로 Q⁻¹ 계산
    lo, hi = 0.0, 40.0 * sigma
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _q_function(mid / sigma) > target_ber:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 프리셋 SA 파라미터
# ══════════════════════════════════════════════════════════════════════════════

SA_7NM = SAParams(
    sa_offset_v=0.008,
    sa_sensitivity_v=0.011,
    t_sense_ns=0.8,
    t_restore_ns=1.2,
    sigma_offset_v=None,      # → 0.008/3 ≈ 2.7mV
    n_bits_per_row=2048,
)

SA_28NM = SAParams(
    sa_offset_v=0.020,
    sa_sensitivity_v=0.026,
    t_sense_ns=2.0,
    t_restore_ns=3.0,
    sigma_offset_v=None,      # → 0.020/3 ≈ 6.7mV
    n_bits_per_row=1024,
)

SA_65NM = SAParams(
    sa_offset_v=0.035,
    sa_sensitivity_v=0.046,
    t_sense_ns=3.5,
    t_restore_ns=5.0,
    sigma_offset_v=None,      # → 0.035/3 ≈ 11.7mV
    n_bits_per_row=512,
)
