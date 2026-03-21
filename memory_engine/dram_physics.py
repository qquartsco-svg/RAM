"""DRAM 셀 물리 모델.

핵심 물리 방정식
─────────────────
1. RC 지수 방전 (retention decay):
       Q(t) = Q0 × exp(−t / τ)
   τ 는 Arrhenius 보정으로 온도 의존.

2. Arrhenius 온도 가속 (retention_tau):
       τ(T) = τ_ref × exp( Ea/kB × (1/T − 1/T_ref) )
   Ea 는 활성화 에너지 [eV], kB = 8.617×10⁻⁵ eV/K.

3. 비트라인 스윙 (bitline_swing_fraction):
       ΔVbl / Vdd = q × Cs / (Cs + Cbl)
   Cs: 저장 커패시터, Cbl: 비트라인 부하 커패시터.
   센스앰프가 감지하려면 이 값 ≥ sense_threshold_fraction.

4. 읽기 교란 (read_disturb):
       q_new = q − read_disturb_factor × q
   읽기 전압이 접근 트랜지스터를 통해 저장 노드에 미소 방전.

5. Refresh:
       q_new = min(1, q + refresh_recovery × (1 − q))
   부스트 펌프로 커패시터를 재충전.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .schema import DRAMCellState, DRAMDesignParams

# Boltzmann 상수 [eV/K]
_KB_EV: float = 8.617_333e-5


# ══════════════════════════════════════════════════════════════════════════════
# 핵심 물리 함수
# ══════════════════════════════════════════════════════════════════════════════

def retention_tau(params: DRAMDesignParams) -> float:
    """Arrhenius 보정된 전하 보존 시정수 τ [s].

    고온일수록 τ 감소 (열 가속 열화).

    Args:
        params: DRAM 설계 파라미터.

    Returns:
        τ [s].
    """
    if params.T_k <= 0.0:
        return params.t_ret_ref_s
    exponent = (params.Ea_eV / _KB_EV) * (1.0 / params.T_k - 1.0 / params.T_ref_k)
    # 수치 안정: 지수 클램핑 (±50 → 배율 ~10^±22)
    exponent = max(-50.0, min(50.0, exponent))
    return params.t_ret_ref_s * math.exp(exponent)


def retention_decay(
    cell: DRAMCellState,
    params: DRAMDesignParams,
    dt_s: float,
) -> DRAMCellState:
    """시간 경과에 따른 전하 지수 방전.

       Q(t+dt) = Q(t) × exp(−dt / τ)

    Args:
        cell  : 현재 셀 상태.
        params: DRAM 설계 파라미터.
        dt_s  : 경과 시간 [s].

    Returns:
        갱신된 셀 상태.
    """
    dt = max(0.0, float(dt_s))
    tau = retention_tau(params)
    q_new = max(0.0, cell.q * math.exp(-dt / tau))
    return DRAMCellState(
        q=q_new,
        t_since_refresh_s=cell.t_since_refresh_s + dt,
        n_reads=cell.n_reads,
        n_cycles=cell.n_cycles,
    )


def read_disturb(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> DRAMCellState:
    """읽기 교란 1회 적용: 접근 트랜지스터를 통한 미소 방전.

       q_new = q − read_disturb_factor × q

    Args:
        cell  : 현재 셀 상태.
        params: DRAM 설계 파라미터.

    Returns:
        갱신된 셀 상태 (n_reads += 1).
    """
    delta = params.read_disturb_factor * cell.q
    return DRAMCellState(
        q=max(0.0, cell.q - delta),
        t_since_refresh_s=cell.t_since_refresh_s,
        n_reads=cell.n_reads + 1,
        n_cycles=cell.n_cycles,
    )


def refresh(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> DRAMCellState:
    """Refresh 1회: 부스트 펌프로 저장 전하 회복.

       q_new = min(1, q + refresh_recovery × (1 − q))
    t_since_refresh_s 초기화.

    Args:
        cell  : 현재 셀 상태.
        params: DRAM 설계 파라미터.

    Returns:
        갱신된 셀 상태.
    """
    q_new = min(1.0, cell.q + params.refresh_recovery * (1.0 - cell.q))
    return DRAMCellState(
        q=max(0.0, q_new),
        t_since_refresh_s=0.0,
        n_reads=cell.n_reads,
        n_cycles=cell.n_cycles,
    )


def write(
    cell: DRAMCellState,
    params: DRAMDesignParams,
    value: float = 1.0,
) -> DRAMCellState:
    """데이터 쓰기: 전하를 value 수준으로 설정하고 사이클 카운터 증가.

    Args:
        cell  : 현재 셀 상태.
        params: DRAM 설계 파라미터 (미사용, 일관성 유지).
        value : 기록할 정규화 전하 [0, 1]. 기본 1.0 ('1' 기록).

    Returns:
        갱신된 셀 상태 (n_reads=0 초기화, n_cycles += 1).
    """
    v = max(0.0, min(1.0, float(value)))
    return DRAMCellState(
        q=v,
        t_since_refresh_s=0.0,
        n_reads=0,
        n_cycles=cell.n_cycles + 1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 파생 지표
# ══════════════════════════════════════════════════════════════════════════════

def bitline_swing_fraction(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> float:
    """비트라인 전압 스윙 비율 ΔVbl / Vdd.

       ΔVbl / Vdd = q × Cs / (Cs + Cbl)

    센스앰프는 이 값이 sense_threshold_fraction 이상일 때 감지 성공.

    Returns:
        ΔVbl / Vdd ∈ [0, 1].
    """
    cs = max(1e-9, params.C_s_fF)
    cbl = max(1e-9, params.C_bl_fF)
    return cell.q * cs / (cs + cbl)


def read_margin(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> float:
    """읽기 마진 = ΔVbl/Vdd − sense_threshold_fraction.

    양수: 감지 성공.  음수: 감지 실패 (refresh 필요).
    """
    return bitline_swing_fraction(cell, params) - params.sense_threshold_fraction


def refresh_needed(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> bool:
    """읽기 마진이 0 이하면 refresh 필요 판정."""
    return read_margin(cell, params) < 0.0


def time_to_fail(
    params: DRAMDesignParams,
) -> float:
    """완전 충전(q=1) 상태에서 읽기 마진이 0이 되는 시간 [s] (retention time 한계).

       t_fail = τ × ln(Cs / ((Cs + Cbl) × threshold))
    """
    tau = retention_tau(params)
    cs = max(1e-9, params.C_s_fF)
    cbl = max(1e-9, params.C_bl_fF)
    thr = max(1e-9, params.sense_threshold_fraction)
    ratio = cs / ((cs + cbl) * thr)
    if ratio <= 0.0:
        return 0.0
    return tau * math.log(ratio)
