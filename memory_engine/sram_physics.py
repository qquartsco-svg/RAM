"""SRAM 6T 셀 물리 모델.

핵심 물리
─────────
1. CMOS 인버터 VTC (Voltage Transfer Characteristic):
   3구간 piece-wise 모델.
     - Vin < Vth_n          : Vout ≈ Vdd  (NMOS off)
     - Vth_n ≤ Vin ≤ Vdd−Vth_p : 전이 구간 (beta_ratio에 의해 스위칭 포인트 결정)
     - Vin > Vdd−Vth_p      : Vout ≈ 0   (PMOS off)

   스위칭 포인트 (Seevinck 1987 기반):
       Vm = (Vdd + Vth_n·√β − Vth_p) / (1 + √β)
   β = beta_ratio (pull-down NMOS / pull-up PMOS 구동비).

2. SNM (Static Noise Margin):
   butterfly curve = VTC1(Vin) + VTC2(Vout→Vin 반전).
   최대 내접 정사각형 한 변 길이.
   단순화 공식 (Seevinck):
       SNM ≈ 0.5 × V_swing × (β−1)/(β+1)
       V_swing = Vdd − Vth_n − Vth_p
   수치 VTC 그리드 보완으로 정확도 향상.

3. 읽기 노이즈 마진 (RNM):
       RNM = SNM × read_margin_factor
   읽기 시 비트라인이 Vdd로 프리차지 → 내부 노드 전압 상승 → SNM 감소.

4. 쓰기 마진 (WM):
       WM = Vdd × write_margin_factor × wl_strength / β
   워드라인 트랜지스터가 내부 노드를 뒤집는 능력. β 클수록 어려워짐.

5. 안정성 지수 (stability_index):
       SI = SNM / (Vdd / 2)   ∈ [0, 1]
"""

from __future__ import annotations

import math

from .schema import SRAMCellState, SRAMDesignParams


# ══════════════════════════════════════════════════════════════════════════════
# VTC 계산
# ══════════════════════════════════════════════════════════════════════════════

def _vtc_vout(vin: float, params: SRAMDesignParams) -> float:
    """CMOS 인버터 VTC: Vin → Vout.

    3구간 piece-wise 모델:
      0) Vin ≤ Vth_n          → Vout = Vdd
      1) Vth_n < Vin < Vdd−Vth_p → 전이 (선형 + beta 보정)
      2) Vin ≥ Vdd−Vth_p      → Vout = 0
    """
    Vdd = params.vdd_v
    Vth_n = params.Vth_n_v
    Vth_p = params.Vth_p_v
    beta = max(1e-3, params.beta_ratio)

    vin = max(0.0, min(Vdd, vin))

    low = Vth_n
    high = Vdd - Vth_p

    if high <= low:
        # 전이 구간이 없는 공정 파라미터 → 중간에서 급격히 전환
        return Vdd if vin < Vdd * 0.5 else 0.0

    if vin <= low:
        return Vdd
    if vin >= high:
        return 0.0

    # 전이 구간: Seevinck 스위칭 포인트
    sqrt_beta = math.sqrt(beta)
    Vm = (Vdd + Vth_n * sqrt_beta - Vth_p) / (1.0 + sqrt_beta)
    Vm = max(low, min(high, Vm))

    # 전이 구간 내 정규화
    span = high - low
    t = (vin - low) / span  # 0 → 1

    # beta가 클수록 전이가 Vm 쪽으로 치우쳐 급격해짐
    # piece-wise: [low, Vm] → Vdd→0  /  [Vm, high] → 0
    if vin <= Vm:
        t_half = (vin - low) / max(1e-12, Vm - low)
        return Vdd * (1.0 - t_half)
    else:
        return 0.0


def vtc_curve(params: SRAMDesignParams, n_points: int = 201) -> list[tuple[float, float]]:
    """VTC 커브 전체 샘플링: [(Vin, Vout), ...].

    Args:
        params  : SRAM 설계 파라미터.
        n_points: 샘플 수.

    Returns:
        (Vin, Vout) 튜플 리스트.
    """
    Vdd = params.vdd_v
    return [
        (Vdd * i / (n_points - 1), _vtc_vout(Vdd * i / (n_points - 1), params))
        for i in range(n_points)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SNM / 마진 계산
# ══════════════════════════════════════════════════════════════════════════════

def static_noise_margin(params: SRAMDesignParams) -> float:
    """SNM: butterfly curve 내접 정사각형 한 변 [V].

    Seevinck 단순화 공식 + 수치 VTC 보완의 평균:
      SNM_analytic = 0.5 × (Vdd − Vth_n − Vth_p) × (β−1)/(β+1)
      SNM_numeric  = max over Vin of |Vout(Vin) − Vin| × 0.35

    Returns:
        SNM [V] ≥ 0.
    """
    Vdd = params.vdd_v
    Vth_n = params.Vth_n_v
    Vth_p = params.Vth_p_v
    beta = max(1e-3, params.beta_ratio)

    # ── 1) Seevinck 해석적 공식 ──────────────────────────────────────────
    v_swing = max(0.0, Vdd - Vth_n - Vth_p)
    snm_analytic = 0.5 * v_swing * (beta - 1.0) / (beta + 1.0)
    snm_analytic = max(0.0, snm_analytic)

    # ── 2) 수치 VTC 그리드 추정 ──────────────────────────────────────────
    n = 400
    snm_numeric = 0.0
    for i in range(n + 1):
        vin = Vdd * i / n
        vout = _vtc_vout(vin, params)
        # butterfly 내접 추정: |Vout − Vin| 의 0.35 배 (90도 내접 정사각형 근사)
        candidate = abs(vout - vin) * 0.35
        if candidate > snm_numeric:
            snm_numeric = candidate

    # ── 3) 두 추정값 평균 (보수적) ──────────────────────────────────────
    snm = (snm_analytic + snm_numeric) * 0.5
    return max(0.0, min(Vdd * 0.5, snm))


def read_noise_margin(params: SRAMDesignParams) -> float:
    """RNM = SNM × read_margin_factor.

    읽기 시 비트라인 프리차지로 내부 노드가 교란되어 SNM이 감소함을 반영.

    Returns:
        RNM [V].
    """
    return static_noise_margin(params) * params.read_margin_factor


def write_margin(params: SRAMDesignParams) -> float:
    """WM = Vdd × write_margin_factor × wl_strength / beta_ratio.

    워드라인 트랜지스터가 크로스커플 인버터를 뒤집는 구동력.
    beta 클수록 저장 셀이 강해 쓰기 어려워짐.

    Returns:
        WM [V].
    """
    beta = max(1e-3, params.beta_ratio)
    return params.vdd_v * params.write_margin_factor * params.wl_strength / beta


def stability_index(params: SRAMDesignParams) -> float:
    """안정성 지수 = SNM / (Vdd/2) ∈ [0, 1].

    0: 불안정(메타스테이블), 1: 이상적 안정.
    """
    snm = static_noise_margin(params)
    half_vdd = max(1e-9, params.vdd_v / 2.0)
    return min(1.0, snm / half_vdd)


def hold_snm(params: SRAMDesignParams) -> float:
    """Hold(대기) 모드 SNM = static_noise_margin (동일).
    읽기 모드와 구별을 명확히 하기 위한 별칭.
    """
    return static_noise_margin(params)


# ══════════════════════════════════════════════════════════════════════════════
# 셀 조작
# ══════════════════════════════════════════════════════════════════════════════

def sram_initial_state(params: SRAMDesignParams, stored_high: bool = True) -> SRAMCellState:
    """완전 쓰기 직후 초기 셀 상태.

    Args:
        params      : SRAM 설계 파라미터.
        stored_high : True → '1' 저장 (v_q=Vdd, v_qb=0).
    """
    if stored_high:
        return SRAMCellState(v_q=params.vdd_v, v_qb=0.0, n_cycles=0)
    return SRAMCellState(v_q=0.0, v_qb=params.vdd_v, n_cycles=0)


def sram_write(
    cell: SRAMCellState,
    params: SRAMDesignParams,
    value: bool,
) -> SRAMCellState:
    """SRAM 셀에 새 값 기록.

    쓰기 마진 > 0 이면 성공, 그렇지 않으면 셀 상태 변경 없음.

    Args:
        cell  : 현재 셀 상태.
        params: SRAM 설계 파라미터.
        value : 기록할 논리값 (True='1', False='0').

    Returns:
        갱신된 셀 상태.
    """
    wm = write_margin(params)
    if wm <= 0.0:
        # 쓰기 실패: 상태 보존, 사이클은 소모된 것으로 처리
        return SRAMCellState(v_q=cell.v_q, v_qb=cell.v_qb, n_cycles=cell.n_cycles + 1)
    if value:
        return SRAMCellState(v_q=params.vdd_v, v_qb=0.0, n_cycles=cell.n_cycles + 1)
    return SRAMCellState(v_q=0.0, v_qb=params.vdd_v, n_cycles=cell.n_cycles + 1)


def sram_read(
    cell: SRAMCellState,
    params: SRAMDesignParams,
) -> tuple[bool, float]:
    """SRAM 셀 읽기.

    읽기 마진 ≥ 0 이면 노드 전압차로 값 판독.

    Returns:
        (값: bool, 읽기 마진: float [V]).
    """
    rnm = read_noise_margin(params)
    # 노드 전압차로 저장 값 결정
    value = cell.v_q > cell.v_qb
    return value, rnm


def sram_leakage_decay(
    cell: SRAMCellState,
    params: SRAMDesignParams,
    dt_s: float,
) -> SRAMCellState:
    """누설 전류에 의한 미소 노드 전압 열화 (서브 문턱 누설 유비).

       dV/dt ≈ −leakage_nA × 1e−9 / C_gate
    C_gate ≈ 1 fF (단순화 — 상대적 열화 추적용).

    Returns:
        갱신된 셀 상태.
    """
    dt = max(0.0, float(dt_s))
    C_gate_F = 1e-15  # 1 fF (정규화 기준)
    I_leak_A = params.leakage_nA * 1e-9
    dV = I_leak_A * dt / C_gate_F
    # v_q, v_qb 모두 미소 감소 (크로스커플이기 때문에 차동으로 작용)
    v_q_new = max(0.0, cell.v_q - dV)
    v_qb_new = max(0.0, cell.v_qb - dV)
    return SRAMCellState(v_q=v_q_new, v_qb=v_qb_new, n_cycles=cell.n_cycles)
