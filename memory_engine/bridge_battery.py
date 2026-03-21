"""배터리 방전 → Vdd 강하 → 메모리 연쇄 실패 연동 브리지.

아키텍처
─────────
Battery (ECM) → PMIC 변환 효율 → Vdd → Memory_Engine 캐스케이드 분석

    V_term(SOC) = V0 + k_ocv × SOC − I × R0
    Vdd = V_term × n_cells × η_pmic          (n_cells: 직렬 셀 수)

battery_dynamics 패키지 의존성 없음 — ECM을 최소 내장 모델로 구현.
외부에서 이미 계산된 V_term 목록을 전달하는 오버로드도 지원.

주요 API
─────────
  simulate_battery_discharge(current_a, dt_s, n_steps, params) -> List[BatteryStep]
  vterm_to_vdd(v_term, n_cells, pmic_efficiency) -> float
  battery_memory_cascade(
      battery_steps, n_cells, pmic_efficiency,
      dram_params, sram_params, sa_params
  ) -> Dict
  sweep_soc_memory_health(soc_range, params_batt, n_cells, pmic_eff,
                           dram_params, sram_params, sa_params) -> List[Dict]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import DRAMCellState, DRAMDesignParams, SRAMCellState, SRAMDesignParams
from .design_engine import scenario_vdd_drop_cascade


# ══════════════════════════════════════════════════════════════════════════════
# 최소 Battery ECM (내장)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatteryECMParams:
    """단일 RC ECM 배터리 파라미터.

    Attributes:
        q_ah              : 정격 용량 [Ah].
        soc_v0            : SOC=0 시 개회로 전압 [V].
        soc_ocv_v_per_unit: SOC=1 → V0 + k_ocv [V/SOC].
        r0_ohm            : 내부 저항 (순시) [Ω].
        r1_ohm            : RC 분극 저항 [Ω].
        c1_farad          : RC 분극 커패시터 [F].
        soh               : 건강 상태 (State of Health) ∈ (0, 1].
        t_cutoff_v        : 방전 종지 전압 [V] (셀 단위).
    """
    q_ah:               float = 5.0
    soc_v0:             float = 2.8
    soc_ocv_v_per_unit: float = 1.0   # OCV range: 2.8 ~ 3.8 V
    r0_ohm:             float = 0.08
    r1_ohm:             float = 0.04
    c1_farad:           float = 2000.0
    soh:                float = 1.0
    t_cutoff_v:         float = 2.7   # 방전 종지 전압 (셀)


@dataclass
class BatteryStep:
    """단일 시뮬레이션 스텝 결과."""
    t_s:       float    # 경과 시간 [s]
    soc:       float    # 충전 상태
    v_term:    float    # 단자 전압 [V] (셀)
    current_a: float    # 방전 전류 [A]
    v_rc:      float    # RC 분극 전압 [V]
    discharged: bool    # 종지 전압 도달 여부


def _ocv(soc: float, p: BatteryECMParams) -> float:
    """SOC → 개회로 전압 (선형 근사)."""
    return p.soc_v0 + p.soc_ocv_v_per_unit * max(0.0, min(1.0, soc))


def _vterm(soc: float, I_a: float, v_rc: float, p: BatteryECMParams) -> float:
    """단자 전압: V_term = OCV − I·R0 − V_rc."""
    return _ocv(soc, p) - I_a * p.r0_ohm - v_rc


def simulate_battery_discharge(
    current_a: float,
    dt_s: float,
    n_steps: int,
    params: BatteryECMParams,
    soc_init: float = 1.0,
    v_rc_init: float = 0.0,
) -> List[BatteryStep]:
    """정전류 방전 시뮬레이션.

    Args:
        current_a : 방전 전류 [A] (양수 = 방전).
        dt_s      : 시뮬레이션 스텝 [s].
        n_steps   : 총 스텝 수.
        params    : ECM 파라미터.
        soc_init  : 초기 SOC.
        v_rc_init : 초기 분극 전압.

    Returns:
        BatteryStep 리스트 (방전 완료 후 종료).
    """
    I = max(0.0, float(current_a))
    dt = max(1e-9, float(dt_s))
    q_eff = max(1e-9, params.q_ah * 3600.0 * max(0.01, params.soh))  # [As]
    tau = max(1e-9, params.r1_ohm * params.c1_farad)

    soc   = max(0.0, min(1.0, float(soc_init)))
    v_rc  = float(v_rc_init)
    t     = 0.0
    steps: List[BatteryStep] = []

    for _ in range(int(n_steps) + 1):
        vt = _vterm(soc, I, v_rc, params)
        discharged = vt < params.t_cutoff_v
        steps.append(BatteryStep(
            t_s=round(t, 6),
            soc=round(soc, 6),
            v_term=round(vt, 6),
            current_a=round(I, 6),
            v_rc=round(v_rc, 6),
            discharged=discharged,
        ))
        if discharged:
            break
        # ECM 적분
        soc   = max(0.0, min(1.0, soc - (I / q_eff) * dt))
        v_rc  = v_rc + dt * (-v_rc / tau + I / params.c1_farad)
        t    += dt

    return steps


# ══════════════════════════════════════════════════════════════════════════════
# Vdd 변환 및 캐스케이드 브리지
# ══════════════════════════════════════════════════════════════════════════════

def vterm_to_vdd(
    v_term_cell: float,
    n_cells: int = 1,
    pmic_efficiency: float = 0.85,
) -> float:
    """배터리 단자 전압 → 메모리 공급 전압 변환.

    V_dd = V_term_cell × n_cells × η_pmic

    실제 PMIC(Power Management IC)는 배터리 팩 전압을 메모리 Vdd로
    강압(Buck) 변환. 효율 η_pmic ∈ (0, 1].

    Args:
        v_term_cell  : 셀 단자 전압 [V].
        n_cells      : 직렬 셀 수.
        pmic_efficiency: PMIC 변환 효율 (기본 85%).

    Returns:
        추정 메모리 Vdd [V].
    """
    pack_v = max(0.0, float(v_term_cell)) * max(1, int(n_cells))
    return pack_v * max(0.01, min(1.0, float(pmic_efficiency)))


def battery_memory_cascade(
    battery_steps: List[BatteryStep],
    dram_params: DRAMDesignParams,
    sram_params: SRAMDesignParams,
    sa_params: Any,
    n_cells: int = 1,
    pmic_efficiency: float = 0.85,
) -> Dict[str, Any]:
    """배터리 방전 궤적 → 메모리 Vdd 연쇄 실패 분석.

    각 배터리 스텝의 V_term을 Vdd로 변환하고
    scenario_vdd_drop_cascade를 호출해 DRAM + SRAM 상태를 평가.

    Args:
        battery_steps  : simulate_battery_discharge() 결과.
        dram_params    : DRAM 설계 파라미터.
        sram_params    : SRAM 설계 파라미터.
        sa_params      : 센스앰프 파라미터.
        n_cells        : 직렬 셀 수.
        pmic_efficiency: PMIC 변환 효율.

    Returns:
        {
          "trajectory": [ {t_s, soc, v_term, vdd,
                            dram_read_ok, sram_verdict, combined_verdict} ],
          "first_dram_fail_t_s": DRAM SA 실패 첫 시점 (None이면 없음),
          "first_sram_fail_t_s": SRAM FRAGILE 시작 첫 시점,
          "first_cascade_t_s":   둘 중 먼저 발생한 시점,
          "min_vdd":             시뮬레이션 중 최소 Vdd,
          "summary":             str,
        }
    """
    if not battery_steps:
        return {"trajectory": [], "first_dram_fail_t_s": None,
                "first_sram_fail_t_s": None, "first_cascade_t_s": None,
                "min_vdd": None, "summary": "배터리 스텝 없음."}

    # Vdd 목록 추출 (중복 제거 불필요 — 시간순 유지)
    vdd_list = [
        vterm_to_vdd(step.v_term, n_cells, pmic_efficiency)
        for step in battery_steps
    ]

    # scenario_vdd_drop_cascade 는 Vdd 목록을 받아 per_vdd 딕트 반환
    cascade = scenario_vdd_drop_cascade(
        dram_params, sram_params, sa_params,
        vdd_range=vdd_list,
    )

    # per_vdd는 Vdd 오름차순 정렬 → 시간순과 다를 수 있음
    # 시간 축 유지를 위해 battery_steps 순서대로 재구성
    vdd_to_row = {r["vdd_v"]: r for r in cascade["per_vdd"]}

    trajectory: List[Dict[str, Any]] = []
    first_dram_fail_t = None
    first_sram_fail_t = None

    for step, vdd in zip(battery_steps, vdd_list):
        vdd_r = round(vdd, 4)
        # 가장 근접한 Vdd 행 매핑 (소수점 반올림 오차 허용)
        row = vdd_to_row.get(vdd_r)
        if row is None:
            # 근접값 탐색
            closest = min(vdd_to_row.keys(), key=lambda v: abs(v - vdd_r))
            row = vdd_to_row[closest]

        dram_ok = row["dram_read_ok"]
        sram_v  = row["sram_verdict"]
        comb    = row["combined_verdict"]

        traj_row = {
            "t_s":             step.t_s,
            "soc":             step.soc,
            "v_term":          step.v_term,
            "vdd":             vdd_r,
            "dram_read_ok":    dram_ok,
            "sram_verdict":    sram_v,
            "combined_verdict": comb,
            "dram_sa_margin_v": row["dram_sa_margin_v"],
            "sram_snm_v":      row["sram_snm_v"],
        }
        trajectory.append(traj_row)

        if not dram_ok and first_dram_fail_t is None:
            first_dram_fail_t = step.t_s
        if sram_v in ("FRAGILE", "CRITICAL") and first_sram_fail_t is None:
            first_sram_fail_t = step.t_s

    # 연쇄 시작 시점 (두 실패 중 먼저)
    fails_t = [v for v in [first_dram_fail_t, first_sram_fail_t] if v is not None]
    first_cascade_t = min(fails_t) if fails_t else None

    min_vdd = min(vdd_list) if vdd_list else None

    # 요약
    parts = []
    if first_dram_fail_t is not None:
        parts.append(f"DRAM SA 실패: t={first_dram_fail_t:.1f}s (Vdd 부족)")
    else:
        parts.append("DRAM: 방전 완료까지 안전")
    if first_sram_fail_t is not None:
        parts.append(f"SRAM 불안정: t={first_sram_fail_t:.1f}s")
    else:
        parts.append("SRAM: 방전 완료까지 안정")
    if min_vdd is not None:
        parts.append(f"Vdd 최소: {min_vdd:.4f}V")

    return {
        "trajectory":          trajectory,
        "first_dram_fail_t_s": first_dram_fail_t,
        "first_sram_fail_t_s": first_sram_fail_t,
        "first_cascade_t_s":   first_cascade_t,
        "min_vdd":             round(min_vdd, 4) if min_vdd else None,
        "summary":             " | ".join(parts),
    }


def sweep_soc_memory_health(
    soc_range: List[float],
    batt_params: BatteryECMParams,
    dram_params: DRAMDesignParams,
    sram_params: SRAMDesignParams,
    sa_params: Any,
    current_a: float = 1.0,
    n_cells: int = 1,
    pmic_efficiency: float = 0.85,
) -> List[Dict[str, Any]]:
    """SOC 목록 스윕 → 각 SOC에서 V_term → Vdd → 메모리 건강 지표.

    배터리 방전 시나리오의 '스냅샷' 분석:
    각 SOC 수준에서 순간 V_term을 계산하고 메모리 마진을 평가.

    Args:
        soc_range      : SOC 목록 (0~1).
        batt_params    : ECM 파라미터.
        dram_params    : DRAM 파라미터.
        sram_params    : SRAM 파라미터.
        sa_params      : SA 파라미터.
        current_a      : 방전 전류 [A].
        n_cells        : 직렬 셀 수.
        pmic_efficiency: PMIC 효율.

    Returns:
        각 SOC에서의 {soc, v_term, vdd, dram_read_ok, sram_verdict,
                       dram_sa_margin_v, sram_snm_v, combined_verdict} 리스트.
    """
    results: List[Dict[str, Any]] = []

    for soc in soc_range:
        soc_c = max(0.0, min(1.0, float(soc)))
        # 순간 V_term (정상상태 근사: v_rc ≈ I·R1)
        v_rc_ss = float(current_a) * batt_params.r1_ohm
        vt = _vterm(soc_c, current_a, v_rc_ss, batt_params)
        vdd = vterm_to_vdd(vt, n_cells, pmic_efficiency)

        # 단일 Vdd 포인트 캐스케이드
        cascade = scenario_vdd_drop_cascade(
            dram_params, sram_params, sa_params,
            vdd_range=[vdd],
        )
        row_c = cascade["per_vdd"][0] if cascade["per_vdd"] else {}

        results.append({
            "soc":              round(soc_c, 4),
            "v_term":           round(vt, 4),
            "vdd":              round(vdd, 4),
            "dram_read_ok":     row_c.get("dram_read_ok", None),
            "sram_verdict":     row_c.get("sram_verdict", None),
            "dram_sa_margin_v": row_c.get("dram_sa_margin_v", None),
            "sram_snm_v":       row_c.get("sram_snm_v", None),
            "combined_verdict": row_c.get("combined_verdict", None),
        })

    return results
