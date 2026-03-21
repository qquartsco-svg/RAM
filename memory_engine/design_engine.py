"""메모리 설계 엔진 — 시뮬레이션 / 파라미터 스윕 / 검증.

API 개요
─────────
DRAM:
  simulate_dram_retention(cell, params, total_s, dt_s)
    → 시간별 전하 방전 이력.
  simulate_dram_read_disturb(cell, params, n_reads)
    → 읽기 누적에 따른 전하 손실 이력.
  simulate_dram_refresh_cycle(cell, params, n_cycles, period_s)
    → refresh 주기별 전하 유지 이력.
  sweep_dram_temperature(params, T_range)
    → 온도별 τ, Ω 분석.
  sweep_dram_vdd(params, vdd_range)
    → 전원 전압별 bitline swing, read margin, Ω 분석.
  sweep_dram_cs(params, cs_range)
    → 저장 커패시터 크기별 특성 분석.
  verify_dram(cell, params) → VerificationReport

SRAM:
  simulate_sram_leakage(cell, params, total_s, dt_s)
    → 누설에 의한 노드 전압 열화 이력.
  sweep_sram_beta(params, beta_range)
    → beta ratio별 SNM / WM 트레이드오프.
  sweep_sram_vdd(params, vdd_range)
    → 전원 전압별 SNM, RNM, WM, Ω.
  sweep_sram_temperature(params, T_range)
    → 온도가 Vth에 미치는 영향 (단순 선형 근사) 분석.
  verify_sram(cell, params) → VerificationReport
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, Dict, List

from .schema import (
    DRAMCellState,
    DRAMDesignParams,
    MemoryObservation,
    SRAMCellState,
    SRAMDesignParams,
    VerificationReport,
)
from .dram_physics import (
    bitline_swing_fraction,
    read_margin as _dram_read_margin,
    read_disturb as _read_disturb,
    refresh as _refresh,
    retention_decay as _retention_decay,
    retention_tau,
    time_to_fail,
    write as _dram_write,
    row_hammer_disturb as _row_hammer_disturb,
    row_hammer_failure_threshold,
)
from .sense_amplifier import (
    sense_op as _sense_op,
    sa_bit_error_rate as _sa_ber,
    sa_row_ber as _sa_row_ber,
    sa_min_delta_v_for_ber,
)
from .schema import SAParams, SAObservation
from .observer import diagnose, observe_dram, observe_sram
from .sram_physics import (
    read_noise_margin,
    sram_leakage_decay,
    sram_write as _sram_write,
    stability_index,
    static_noise_margin,
    write_margin,
    snm_by_mode as _snm_by_mode,
    read_node_disturb_v,
    read_snm_physical,
)


# ══════════════════════════════════════════════════════════════════════════════
# DRAM 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

def simulate_dram_retention(
    cell: DRAMCellState,
    params: DRAMDesignParams,
    total_s: float,
    dt_s: float = 1e-3,
) -> List[Dict[str, Any]]:
    """DRAM retention 시뮬레이션.

    자연 방전 Q(t) = Q0·exp(−t/τ) 를 dt_s 간격으로 추적.

    Args:
        cell   : 초기 셀 상태.
        params : DRAM 설계 파라미터.
        total_s: 총 시뮬레이션 시간 [s].
        dt_s   : 시간 스텝 [s].

    Returns:
        각 시점의 {t_s, q, bitline_swing, read_margin, omega_global, verdict} 딕트 리스트.
    """
    history: List[Dict[str, Any]] = []
    c = DRAMCellState(q=cell.q, t_since_refresh_s=cell.t_since_refresh_s,
                      n_reads=cell.n_reads, n_cycles=cell.n_cycles)
    t = 0.0
    dt = max(1e-12, float(dt_s))
    total = max(0.0, float(total_s))

    while t <= total + dt * 0.5:
        obs = observe_dram(c, params)
        history.append({
            "t_s":          round(t, 9),
            "q":            round(c.q, 6),
            "bitline_swing": round(bitline_swing_fraction(c, params), 6),
            "read_margin":  round(_dram_read_margin(c, params), 6),
            "omega_global": obs.omega_global,
            "verdict":      obs.verdict,
            "flags":        list(obs.flags),
        })
        if t >= total:
            break
        c = _retention_decay(c, params, dt)
        t = min(t + dt, total)

    return history


def simulate_dram_read_disturb(
    cell: DRAMCellState,
    params: DRAMDesignParams,
    n_reads: int,
) -> List[Dict[str, Any]]:
    """읽기 교란 누적 시뮬레이션.

    Args:
        cell   : 초기 셀 상태.
        params : DRAM 설계 파라미터.
        n_reads: 읽기 횟수.

    Returns:
        각 읽기 후 {n_reads, q, read_margin, omega_global} 딕트 리스트.
    """
    history: List[Dict[str, Any]] = []
    c = DRAMCellState(q=cell.q, t_since_refresh_s=cell.t_since_refresh_s,
                      n_reads=cell.n_reads, n_cycles=cell.n_cycles)

    for i in range(n_reads + 1):
        obs = observe_dram(c, params)
        history.append({
            "n_reads":     i,
            "q":           round(c.q, 6),
            "read_margin": round(_dram_read_margin(c, params), 6),
            "omega_global": obs.omega_global,
            "verdict":     obs.verdict,
        })
        if i < n_reads:
            c = _read_disturb(c, params)

    return history


def simulate_dram_refresh_cycle(
    cell: DRAMCellState,
    params: DRAMDesignParams,
    n_cycles: int,
    period_s: float = 64e-3,
) -> List[Dict[str, Any]]:
    """Refresh 주기 시뮬레이션.

    각 주기: period_s 동안 방전 → refresh → 다음 주기.

    Args:
        cell    : 초기 셀 상태.
        params  : DRAM 설계 파라미터.
        n_cycles: refresh 주기 수.
        period_s: refresh 주기 [s]. DRAM 표준 64ms.

    Returns:
        각 주기 후 {cycle, q_before_refresh, q_after_refresh, omega_before, omega_after}.
    """
    history: List[Dict[str, Any]] = []
    c = DRAMCellState(q=cell.q, t_since_refresh_s=0.0,
                      n_reads=cell.n_reads, n_cycles=cell.n_cycles)

    for cycle in range(n_cycles):
        # 방전
        c = _retention_decay(c, params, float(period_s))
        obs_before = observe_dram(c, params)
        q_before = c.q
        # refresh
        c = _refresh(c, params)
        obs_after = observe_dram(c, params)
        history.append({
            "cycle":             cycle + 1,
            "q_before_refresh":  round(q_before, 6),
            "q_after_refresh":   round(c.q, 6),
            "read_margin_before": round(_dram_read_margin(
                DRAMCellState(q=q_before), params), 6),
            "omega_before":      obs_before.omega_global,
            "omega_after":       obs_after.omega_global,
            "verdict_before":    obs_before.verdict,
            "verdict_after":     obs_after.verdict,
        })

    return history


# ══════════════════════════════════════════════════════════════════════════════
# DRAM 파라미터 스윕
# ══════════════════════════════════════════════════════════════════════════════

def sweep_dram_temperature(
    params: DRAMDesignParams,
    T_range: List[float],
) -> List[Dict[str, Any]]:
    """온도 스윕: Arrhenius 가속에 따른 τ, time_to_fail, Ω 분석.

    Args:
        params : DRAM 기본 파라미터 (T_k 덮어씀).
        T_range: 온도 목록 [K].

    Returns:
        각 온도에서의 {T_k, T_c, tau_s, tau_ms, time_to_fail_s, omega_global, verdict}.
    """
    results: List[Dict[str, Any]] = []
    for T in T_range:
        p = _dram_replace(params, T_k=float(T))
        cell = DRAMCellState(q=1.0)
        tau = retention_tau(p)
        ttf = time_to_fail(p)
        obs = observe_dram(cell, p)
        results.append({
            "T_k":           round(T, 2),
            "T_c":           round(T - 273.15, 2),
            "tau_s":         round(tau, 6),
            "tau_ms":        round(tau * 1e3, 4),
            "time_to_fail_s": round(ttf, 6),
            "omega_global":  obs.omega_global,
            "verdict":       obs.verdict,
        })
    return results


def sweep_dram_vdd(
    params: DRAMDesignParams,
    vdd_range: List[float],
) -> List[Dict[str, Any]]:
    """전원 전압 스윕: bitline swing, read margin, Ω 분석.

    Args:
        params   : DRAM 기본 파라미터 (vdd_v 덮어씀).
        vdd_range: 전압 목록 [V].

    Returns:
        각 Vdd에서의 {vdd_v, bitline_swing, read_margin, omega_global, verdict}.
    """
    results: List[Dict[str, Any]] = []
    for vdd in vdd_range:
        p = _dram_replace(params, vdd_v=float(vdd))
        cell = DRAMCellState(q=1.0)
        bl = bitline_swing_fraction(cell, p)
        rm = _dram_read_margin(cell, p)
        obs = observe_dram(cell, p)
        results.append({
            "vdd_v":         round(vdd, 4),
            "bitline_swing": round(bl, 6),
            "read_margin":   round(rm, 6),
            "omega_global":  obs.omega_global,
            "verdict":       obs.verdict,
        })
    return results


def sweep_dram_cs(
    params: DRAMDesignParams,
    cs_range: List[float],
) -> List[Dict[str, Any]]:
    """저장 커패시터(Cs) 스윕: bitline swing, 면적-성능 트레이드오프 분석.

    Args:
        params  : DRAM 기본 파라미터 (C_s_fF 덮어씀).
        cs_range: Cs 목록 [fF].

    Returns:
        각 Cs에서의 {C_s_fF, bitline_swing, read_margin, omega_global}.
    """
    results: List[Dict[str, Any]] = []
    for cs in cs_range:
        p = _dram_replace(params, C_s_fF=float(cs))
        cell = DRAMCellState(q=1.0)
        bl = bitline_swing_fraction(cell, p)
        rm = _dram_read_margin(cell, p)
        obs = observe_dram(cell, p)
        results.append({
            "C_s_fF":        round(cs, 3),
            "bitline_swing": round(bl, 6),
            "read_margin":   round(rm, 6),
            "omega_global":  obs.omega_global,
            "verdict":       obs.verdict,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DRAM 검증
# ══════════════════════════════════════════════════════════════════════════════

def verify_dram(
    cell: DRAMCellState,
    params: DRAMDesignParams,
) -> VerificationReport:
    """DRAM 설계 검증 보고서 생성.

    검증 항목:
      - 읽기 마진 (read_margin ≥ 0.02 → PASS, ≥ 0 → MARGINAL)
      - Vdd 범위 (vdd_min ≤ Vdd ≤ vdd_max)
      - 보존 마진 (τ / t_ret_ref ≥ 1.5 → PASS)
      - 내구성 여유 (n_cycles < 80% max → PASS)

    Returns:
        VerificationReport.
    """
    rm = _dram_read_margin(cell, params)
    # 쓰기 마진: DRAM은 full-swing → 1 − threshold (구조적)
    wm = 1.0 - params.sense_threshold_fraction

    tau = retention_tau(params)
    ret_margin = tau / max(1e-12, params.t_ret_ref_s)

    cycle_ratio = cell.n_cycles / max(1, params.max_cycles)
    end_margin = 1.0 - cycle_ratio

    obs = observe_dram(cell, params)
    notes: List[str] = []

    # Vdd 범위
    vdd_ok = params.vdd_min_v <= params.vdd_v <= params.vdd_max_v
    if not vdd_ok:
        notes.append(
            f"Vdd {params.vdd_v:.3f}V 가 허용 범위 "
            f"[{params.vdd_min_v:.3f}, {params.vdd_max_v:.3f}]V 를 벗어남."
        )

    # 읽기 마진
    if rm < 0.0:
        notes.append(f"읽기 마진 부족: {rm:.4f} (센스앰프 감지 불가).")
    elif rm < 0.02:
        notes.append(f"읽기 마진 임계 근처: {rm:.4f} — 공정 변동 여유 부족.")

    # 보존 마진
    if ret_margin < 1.0:
        notes.append(f"retention τ({tau*1e3:.1f}ms) < 기준({params.t_ret_ref_s*1e3:.0f}ms).")
    elif ret_margin < 1.5:
        notes.append(f"retention 마진 {ret_margin:.2f}× — 고온 마진 재확인 필요.")

    # 내구성
    if cycle_ratio > 0.80:
        notes.append(f"내구성 {cycle_ratio*100:.1f}% 소진 — 마모 평준화 필요.")

    # 종합 판정
    if rm >= 0.02 and vdd_ok and ret_margin >= 1.0 and end_margin >= 0.20:
        verdict = "PASS"
    elif rm >= 0.0 and end_margin >= 0.05:
        verdict = "MARGINAL"
        if "마진 여유" not in " ".join(notes):
            notes.append("마진 여유 부족 — 공정 변동 및 온도 리스크 재점검 권장.")
    else:
        verdict = "FAIL"

    return VerificationReport(
        cell_type="DRAM",
        verdict=verdict,
        read_margin=rm,
        write_margin=wm,
        retention_margin=ret_margin,
        endurance_margin=end_margin,
        omega_global=obs.omega_global,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SRAM 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

def simulate_sram_leakage(
    cell: SRAMCellState,
    params: SRAMDesignParams,
    total_s: float,
    dt_s: float = 1e-3,
) -> List[Dict[str, Any]]:
    """SRAM 누설 전류에 의한 노드 전압 열화 시뮬레이션.

    Args:
        cell   : 초기 셀 상태.
        params : SRAM 설계 파라미터.
        total_s: 총 시뮬레이션 시간 [s].
        dt_s   : 시간 스텝 [s].

    Returns:
        각 시점의 {t_s, v_q, v_qb, v_diff, omega_global, verdict} 딕트 리스트.
    """
    history: List[Dict[str, Any]] = []
    c = SRAMCellState(v_q=cell.v_q, v_qb=cell.v_qb, n_cycles=cell.n_cycles)
    t = 0.0
    dt = max(1e-12, float(dt_s))
    total = max(0.0, float(total_s))

    while t <= total + dt * 0.5:
        obs = observe_sram(c, params)
        history.append({
            "t_s":     round(t, 9),
            "v_q":     round(c.v_q, 6),
            "v_qb":    round(c.v_qb, 6),
            "v_diff":  round(abs(c.v_q - c.v_qb), 6),
            "omega_global": obs.omega_global,
            "verdict": obs.verdict,
        })
        if t >= total:
            break
        c = sram_leakage_decay(c, params, dt)
        t = min(t + dt, total)

    return history


# ══════════════════════════════════════════════════════════════════════════════
# SRAM 파라미터 스윕
# ══════════════════════════════════════════════════════════════════════════════

def sweep_sram_beta(
    params: SRAMDesignParams,
    beta_range: List[float],
) -> List[Dict[str, Any]]:
    """beta_ratio 스윕: SNM ↑ vs WM ↓ 트레이드오프.

    Args:
        params    : SRAM 기본 파라미터 (beta_ratio 덮어씀).
        beta_range: beta_ratio 목록.

    Returns:
        각 beta에서의 {beta_ratio, snm_v, rnm_v, wm_v, stability, omega_global}.
    """
    results: List[Dict[str, Any]] = []
    for beta in beta_range:
        p = _sram_replace(params, beta_ratio=float(beta))
        cell = SRAMCellState(v_q=p.vdd_v, v_qb=0.0)
        obs = observe_sram(cell, p)
        results.append({
            "beta_ratio": round(beta, 3),
            "snm_v":      round(static_noise_margin(p), 6),
            "rnm_v":      round(read_noise_margin(p), 6),
            "wm_v":       round(write_margin(p), 6),
            "stability":  round(stability_index(p), 6),
            "omega_global": obs.omega_global,
            "verdict":    obs.verdict,
        })
    return results


def sweep_sram_vdd(
    params: SRAMDesignParams,
    vdd_range: List[float],
) -> List[Dict[str, Any]]:
    """Vdd 스윕: SNM, RNM, WM의 전압 의존성.

    Args:
        params   : SRAM 기본 파라미터 (vdd_v 덮어씀).
        vdd_range: 전압 목록 [V].

    Returns:
        각 Vdd에서의 {vdd_v, snm_v, rnm_v, wm_v, stability, omega_global}.
    """
    results: List[Dict[str, Any]] = []
    for vdd in vdd_range:
        p = _sram_replace(params, vdd_v=float(vdd))
        cell = SRAMCellState(v_q=vdd, v_qb=0.0)
        obs = observe_sram(cell, p)
        results.append({
            "vdd_v":   round(vdd, 4),
            "snm_v":   round(static_noise_margin(p), 6),
            "rnm_v":   round(read_noise_margin(p), 6),
            "wm_v":    round(write_margin(p), 6),
            "stability": round(stability_index(p), 6),
            "omega_global": obs.omega_global,
            "verdict": obs.verdict,
        })
    return results


def sweep_sram_temperature(
    params: SRAMDesignParams,
    T_range: List[float],
    dvth_dT_mV_per_K: float = -0.7,
) -> List[Dict[str, Any]]:
    """온도 스윕: Vth 온도 계수에 따른 SNM 변화.

    단순 선형 근사: ΔVth = dvth_dT_mV_per_K × (T − T_ref) × 1e-3

    Args:
        params            : SRAM 기본 파라미터.
        T_range           : 온도 목록 [K].
        dvth_dT_mV_per_K  : Vth 온도 계수 [mV/K]. 일반적으로 −0.5~−1.0 mV/K.

    Returns:
        각 온도에서의 {T_k, T_c, Vth_n_v, snm_v, rnm_v, omega_global}.
    """
    results: List[Dict[str, Any]] = []
    T_ref = params.T_k
    for T in T_range:
        dVth = dvth_dT_mV_per_K * (T - T_ref) * 1e-3
        Vth_n_new = max(0.05, params.Vth_n_v + dVth)
        Vth_p_new = max(0.05, params.Vth_p_v + dVth)
        p = _sram_replace(params, T_k=float(T), Vth_n_v=Vth_n_new, Vth_p_v=Vth_p_new)
        cell = SRAMCellState(v_q=p.vdd_v, v_qb=0.0)
        obs = observe_sram(cell, p)
        results.append({
            "T_k":     round(T, 2),
            "T_c":     round(T - 273.15, 2),
            "Vth_n_v": round(Vth_n_new, 6),
            "snm_v":   round(static_noise_margin(p), 6),
            "rnm_v":   round(read_noise_margin(p), 6),
            "omega_global": obs.omega_global,
            "verdict": obs.verdict,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SRAM 검증
# ══════════════════════════════════════════════════════════════════════════════

def verify_sram(
    cell: SRAMCellState,
    params: SRAMDesignParams,
) -> VerificationReport:
    """SRAM 설계 검증 보고서 생성.

    검증 항목:
      - SNM ≥ 0.08V → PASS / ≥ 0.04V → MARGINAL
      - RNM ≥ 0.04V → PASS / ≥ 0.02V → MARGINAL
      - WM  ≥ 0.06V → PASS / ≥ 0.03V → MARGINAL
      - 안정성 지수 ≥ 0.40 → PASS

    Returns:
        VerificationReport.
    """
    snm = static_noise_margin(params)
    rnm = read_noise_margin(params)
    wm = write_margin(params)
    si = stability_index(params)

    cycle_ratio = cell.n_cycles / max(1, params.max_cycles)
    end_margin = 1.0 - cycle_ratio

    obs = observe_sram(cell, params)
    notes: List[str] = []

    if snm < 0.04:
        notes.append(f"SNM 위험: {snm*1000:.1f}mV (< 40mV).")
    elif snm < 0.08:
        notes.append(f"SNM 임계: {snm*1000:.1f}mV — 공정 변동 여유 부족.")

    if rnm < 0.02:
        notes.append(f"읽기 마진 부족: {rnm*1000:.1f}mV (< 20mV).")
    elif rnm < 0.04:
        notes.append(f"읽기 마진 임계: {rnm*1000:.1f}mV.")

    if wm < 0.03:
        notes.append(f"쓰기 마진 부족: {wm*1000:.1f}mV (< 30mV).")
    elif wm < 0.06:
        notes.append(f"쓰기 마진 임계: {wm*1000:.1f}mV.")

    if si < 0.30:
        notes.append(f"셀 안정성 위험: stability_index={si:.3f}.")

    # 종합 판정
    if snm >= 0.08 and rnm >= 0.04 and wm >= 0.06 and si >= 0.40:
        verdict = "PASS"
    elif snm >= 0.04 and rnm >= 0.02 and wm >= 0.03 and si >= 0.20:
        verdict = "MARGINAL"
        if not notes:
            notes.append("마진 여유 부족 — 공정 변동 및 온도 리스크 재점검 권장.")
    else:
        verdict = "FAIL"

    return VerificationReport(
        cell_type="SRAM",
        verdict=verdict,
        read_margin=rnm,
        write_margin=wm,
        retention_margin=si,   # SRAM: stability index 재사용
        endurance_margin=end_margin,
        omega_global=obs.omega_global,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 내부 유틸 — frozen dataclass replace 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# A1. Sense Amplifier 시뮬레이션 / 스윕
# ══════════════════════════════════════════════════════════════════════════════

def simulate_sa_vs_retention(
    cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
    total_s: float,
    dt_s: float = 1e-3,
) -> List[Dict[str, Any]]:
    """시간 경과에 따른 DRAM 전하 방전 + 센스앰프 마진 동시 추적.

    retention_decay와 sense_op를 함께 시뮬레이션:
      각 시점마다 [q, ΔV, sa_margin, BER, omega_sa] 기록.

    Args:
        cell        : 초기 셀 상태.
        dram_params : DRAM 설계 파라미터.
        sa_params   : 센스앰프 파라미터.
        total_s     : 총 시뮬레이션 시간 [s].
        dt_s        : 시간 스텝 [s].

    Returns:
        시점별 딕트 리스트.
    """
    history: List[Dict[str, Any]] = []
    c = DRAMCellState(q=cell.q, t_since_refresh_s=cell.t_since_refresh_s,
                      n_reads=cell.n_reads, n_cycles=cell.n_cycles)
    t = 0.0
    dt = max(1e-12, float(dt_s))
    total = max(0.0, float(total_s))

    while t <= total + dt * 0.5:
        sa_obs = _sense_op(c, dram_params, sa_params)
        history.append({
            "t_s":          round(t, 9),
            "q":            round(c.q, 6),
            "delta_v":      sa_obs.delta_v,
            "sa_margin_v":  sa_obs.sa_margin_v,
            "read_success": sa_obs.read_success,
            "ber":          sa_obs.ber,
            "omega_sa":     sa_obs.omega_sa,
            "flags":        list(sa_obs.flags),
        })
        if t >= total:
            break
        c = _retention_decay(c, dram_params, dt)
        t = min(t + dt, total)

    return history


def sweep_sa_offset(
    cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params_base: SAParams,
    offset_range: List[float],
) -> List[Dict[str, Any]]:
    """SA 오프셋 전압 스윕: 오프셋 크기에 따른 마진·BER 변화.

    Args:
        cell          : 현재 DRAM 셀 상태.
        dram_params   : DRAM 설계 파라미터.
        sa_params_base: 기본 SA 파라미터 (sa_offset_v 덮어씀).
        offset_range  : sa_offset_v 목록 [V].

    Returns:
        각 오프셋에서의 {sa_offset_v, sa_margin_v, read_success, ber, omega_sa}.
    """
    results: List[Dict[str, Any]] = []
    from dataclasses import replace as _replace
    for offset in offset_range:
        sa = _replace(sa_params_base, sa_offset_v=float(offset))
        obs = _sense_op(cell, dram_params, sa)
        results.append({
            "sa_offset_v":  round(offset, 6),
            "sa_margin_v":  obs.sa_margin_v,
            "read_success": obs.read_success,
            "ber":          obs.ber,
            "omega_sa":     obs.omega_sa,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# A2. Row Hammer 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════════

def simulate_row_hammer(
    victim_cell: DRAMCellState,
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
    total_hammers: int,
    step_hammers: int = 1000,
    rh_charge_loss_per_event: float = 5e-6,
) -> List[Dict[str, Any]]:
    """Row Hammer 누적 시뮬레이션.

    step_hammers 간격으로 hammer를 적용하며 피해 셀 전하·SA 마진 추적.

    Args:
        victim_cell              : 피해 셀 초기 상태.
        dram_params              : DRAM 설계 파라미터.
        sa_params                : 센스앰프 파라미터.
        total_hammers            : 총 hammer 횟수.
        step_hammers             : 기록 간격 (n_hammers 단위).
        rh_charge_loss_per_event : hammer 1회당 전하 손실 비율.

    Returns:
        각 체크포인트의 {n_hammers, q, sa_margin_v, read_success, ber, omega_sa} 리스트.
    """
    history: List[Dict[str, Any]] = []
    c = DRAMCellState(q=victim_cell.q, t_since_refresh_s=victim_cell.t_since_refresh_s,
                      n_reads=victim_cell.n_reads, n_cycles=victim_cell.n_cycles)
    n = 0

    while n <= total_hammers:
        sa_obs = _sense_op(c, dram_params, sa_params)
        history.append({
            "n_hammers":    n,
            "q":            round(c.q, 6),
            "sa_margin_v":  sa_obs.sa_margin_v,
            "read_success": sa_obs.read_success,
            "ber":          sa_obs.ber,
            "omega_sa":     sa_obs.omega_sa,
            "flags":        list(sa_obs.flags),
        })
        if n >= total_hammers:
            break
        step = min(step_hammers, total_hammers - n)
        c = _row_hammer_disturb(c, dram_params, step, rh_charge_loss_per_event)
        n += step

    return history


def find_row_hammer_threshold(
    dram_params: DRAMDesignParams,
    sa_params: SAParams,
    rh_charge_loss_per_event: float = 5e-6,
    q_init: float = 1.0,
) -> Dict[str, Any]:
    """Row Hammer 실패 임계 분석.

    SA 마진 실패와 물리적 전하 실패 두 기준에서 각각 임계 hammer 수 계산.

    Returns:
        {n_hammer_sa_fail, n_hammer_q_fail, sa_margin_at_threshold, safer_limit}.
    """
    # SA 실패 임계: ΔV < sa_offset_v
    # ΔV = q × Vdd × Cs/(Cs+Cbl)
    cs  = max(1e-9, dram_params.C_s_fF)
    cbl = max(1e-9, dram_params.C_bl_fF)
    vdd = dram_params.vdd_v
    # q 수준에서 SA가 실패하는 q_sa_fail
    q_sa_fail = sa_params.sa_offset_v * (cs + cbl) / (cs * vdd)

    loss = max(1e-12, rh_charge_loss_per_event)
    import math as _math

    if q_sa_fail >= q_init:
        n_sa = 0
    else:
        n_sa = max(0, int(_math.ceil(
            _math.log(q_sa_fail / q_init) / _math.log(1.0 - loss)
        )))

    n_q = row_hammer_failure_threshold(dram_params, loss)

    return {
        "n_hammer_sa_fail":      n_sa,
        "n_hammer_q_fail":       n_q,
        "q_sa_fail_threshold":   round(q_sa_fail, 6),
        "safer_limit":           min(n_sa, n_q),
        "rh_factor_per_event":   rh_charge_loss_per_event,
    }


# ══════════════════════════════════════════════════════════════════════════════
# A3. SRAM 3모드 SNM 분석
# ══════════════════════════════════════════════════════════════════════════════

def sram_mode_analysis(params: SRAMDesignParams) -> Dict[str, Any]:
    """SRAM 3모드(Hold / Read / Write) 마진 통합 분석.

    Returns:
        snm_by_mode() 결과 + 추가 해석:
          mode_verdict     : 3모드 최약점 기준 종합 판정.
          read_write_tradeoff : SNM_hold vs WM 트레이드오프 설명.
          node_disturb_risk   : 읽기 시 내부 노드 교란 위험도 ('LOW'/'MED'/'HIGH').
    """
    modes = _snm_by_mode(params)

    # 노드 교란 위험도
    dv = read_node_disturb_v(params)
    snm_hold = static_noise_margin(params)
    disturb_ratio = dv / max(1e-9, snm_hold)
    if disturb_ratio < 0.30:
        node_risk = "LOW"
    elif disturb_ratio < 0.60:
        node_risk = "MEDIUM"
    else:
        node_risk = "HIGH"

    # 트레이드오프 설명
    beta = params.beta_ratio
    if beta < 2.0:
        tradeoff = "beta 낮음 → SNM 작음·WM 큼 (쓰기 유리·읽기 위험)"
    elif beta <= 3.0:
        tradeoff = "beta 균형 → SNM/WM 모두 적정 (설계 권장 구간)"
    else:
        tradeoff = "beta 높음 → SNM 큼·WM 작음 (읽기 안전·쓰기 어려움)"

    return {
        **modes,
        "node_disturb_risk":    node_risk,
        "disturb_ratio":        round(disturb_ratio, 4),
        "read_write_tradeoff":  tradeoff,
    }


def sweep_sram_mode_beta(
    params: SRAMDesignParams,
    beta_range: List[float],
) -> List[Dict[str, Any]]:
    """beta_ratio 스윕 + 3모드 SNM 분석.

    Args:
        params    : SRAM 기본 파라미터.
        beta_range: beta_ratio 목록.

    Returns:
        각 beta에서의 {beta_ratio, hold_snm, read_snm_physical, write_margin, ...}.
    """
    results: List[Dict[str, Any]] = []
    for beta in beta_range:
        p = _sram_replace(params, beta_ratio=float(beta))
        analysis = sram_mode_analysis(p)
        results.append({"beta_ratio": round(beta, 3), **analysis})
    return results


def _dram_replace(params: DRAMDesignParams, **kwargs: Any) -> DRAMDesignParams:
    """DRAMDesignParams(frozen) 일부 필드 대체 사본 생성."""
    return replace(params, **kwargs)


def _sram_replace(params: SRAMDesignParams, **kwargs: Any) -> SRAMDesignParams:
    """SRAMDesignParams(frozen) 일부 필드 대체 사본 생성."""
    return replace(params, **kwargs)
