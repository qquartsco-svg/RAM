"""메모리 엔진 핵심 스키마 — 셀 상태 / 설계 파라미터 / 관측 결과 / 검증 보고서.

메모리 유형:
  DRAM : 1T1C 커패시터 셀. 전하 보존·방전·refresh 주기·read disturb 모형화.
  SRAM : 6T 크로스커플 인버터 셀. SNM(Static Noise Margin)·읽기·쓰기 마진 모형화.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ── 메모리 유형 ────────────────────────────────────────────────────────────────
class MemoryCellType(str, Enum):
    DRAM = "DRAM"
    SRAM = "SRAM"


# ══════════════════════════════════════════════════════════════════════════════
# DRAM 스키마
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DRAMCellState:
    """단일 DRAM 1T1C 셀의 동적 상태.

    q                : 정규화 전하 [0, 1].  1 = 완전 충전(논리 '1'), 0 = 방전.
    t_since_refresh_s: 마지막 refresh 이후 경과 시간 [s].
    n_reads          : 마지막 쓰기 이후 누적 읽기 횟수 (read disturb 누적용).
    n_cycles         : 누적 쓰기 사이클 수 (내구성 추적).
    """
    q: float = 1.0
    t_since_refresh_s: float = 0.0
    n_reads: int = 0
    n_cycles: int = 0


@dataclass(frozen=True)
class DRAMDesignParams:
    """DRAM 설계·공정 파라미터 (변경 불가 값 객체).

    공정 파라미터:
      node_nm     : 공정 노드 [nm].
      vdd_v       : 전원 전압 [V].
      T_k         : 동작 온도 [K].

    셀 물리:
      C_s_fF      : 저장 커패시터 [fF]. DRAM 셀 용량.
      C_bl_fF     : 비트라인 커패시터 [fF]. 감지 부하.
      t_ret_ref_s : 기준 온도(T_ref_k)에서 전하 보존 시정수 τ [s].
                    τ ≈ 64ms (DRAM 표준 refresh 주기 기준).
      T_ref_k     : 기준 온도 [K].
      Ea_eV       : Arrhenius 활성화 에너지 [eV].
                    Si DRAM 전형값 ≈ 0.5~0.7 eV.

    읽기 / 쓰기:
      sense_threshold_fraction : 최소 비트라인 스윙 비율 ΔVbl/Vdd.
                                 이 값 이하면 센스앰프 감지 실패.
      read_disturb_factor      : 읽기 1회당 전하 손실 비율.
      refresh_recovery         : refresh 1회로 회복되는 전하 비율 (0~1).
      t_access_ns              : 셀 접근 사이클 시간 [ns] (t_RCD + t_CL 근사).
      power_mW_per_cell        : 셀당 동적 소비전력 [mW].

    한계:
      max_cycles  : 허용 최대 쓰기 사이클 수 (DRAM ≈ 10^8).
      vdd_min_v   : 최소 허용 전원 전압 [V].
      vdd_max_v   : 최대 허용 전원 전압 [V].
    """
    # 공정
    node_nm: float = 28.0
    vdd_v: float = 1.2
    T_k: float = 300.0
    # 셀 물리
    C_s_fF: float = 25.0
    C_bl_fF: float = 100.0
    t_ret_ref_s: float = 64e-3
    T_ref_k: float = 300.0
    Ea_eV: float = 0.55
    # 읽기/쓰기
    sense_threshold_fraction: float = 0.12
    read_disturb_factor: float = 1.2e-4
    refresh_recovery: float = 0.40
    t_access_ns: float = 10.0
    power_mW_per_cell: float = 5e-6
    # 한계
    max_cycles: int = 100_000_000
    vdd_min_v: float = 1.10
    vdd_max_v: float = 1.30


# ══════════════════════════════════════════════════════════════════════════════
# SRAM 스키마
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SRAMCellState:
    """단일 SRAM 6T 셀의 동적 상태.

    v_q     : 저장 노드 전압 [V]. 논리 '1' 저장 시 Vdd에 수렴.
    v_qb    : 상보 노드 전압 [V]. 논리 '1' 저장 시 0에 수렴.
    n_cycles: 누적 쓰기 사이클 수.
    """
    v_q: float = 1.0
    v_qb: float = 0.0
    n_cycles: int = 0


@dataclass(frozen=True)
class SRAMDesignParams:
    """SRAM 설계·공정 파라미터 (변경 불가 값 객체).

    공정 파라미터:
      node_nm     : 공정 노드 [nm].
      vdd_v       : 전원 전압 [V].
      T_k         : 동작 온도 [K].
      Vth_n_v     : NMOS 문턱 전압 [V].
      Vth_p_v     : PMOS 문턱 전압 절댓값 [V].

    셀 파라미터:
      beta_ratio      : pull-down(저장 NMOS) / pull-up(부하 PMOS) 전류 구동비.
                        beta > 1 → 읽기 SNM 증가, 쓰기 마진 감소.
      wl_strength     : 워드라인(접근) 트랜지스터 상대 구동력 (정규화, 1.0 = 기준).

    마진:
      read_margin_factor  : RNM = SNM × factor. 읽기 시 SNM 감소 비율.
      write_margin_factor : WM = Vdd × factor × wl_strength / beta.

    접근/전력:
      t_access_ns : 접근 사이클 시간 [ns].
      leakage_nA  : 셀당 정적 누설 전류 [nA].

    한계:
      max_cycles  : SRAM은 사실상 무제한(~10^12).
    """
    # 공정
    node_nm: float = 28.0
    vdd_v: float = 1.0
    T_k: float = 300.0
    Vth_n_v: float = 0.35
    Vth_p_v: float = 0.35
    # 셀 파라미터
    beta_ratio: float = 2.5
    wl_strength: float = 1.0
    # 마진
    read_margin_factor: float = 0.60
    write_margin_factor: float = 0.42
    # 접근/전력
    t_access_ns: float = 1.0
    leakage_nA: float = 2.0
    # 한계
    max_cycles: int = 1_000_000_000_000


# ══════════════════════════════════════════════════════════════════════════════
# 공통 관측/검증 스키마
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryObservation:
    """메모리 셀 관측 결과 — Ω 5레이어 + 종합 판정.

    Ω 구성 (DRAM):
      omega_retention : 전하 보존율 (q 직접 반영).
      omega_margin    : 읽기 마진 비율 (bitline swing / sense_threshold 기반).
      omega_endurance : 내구성 여유 (1 - n_cycles / max_cycles).
      omega_speed     : 접근 속도 효율 (1 - t_access / t_ref).
      omega_power     : 소비전력 효율 (낮을수록 좋음).

    Ω 구성 (SRAM):
      omega_retention : 노드 전압차 정규화 (|Vq - Vqb| / Vdd).
      omega_margin    : SNM / (Vdd/2) — 안정성 마진.
      omega_endurance : 내구성 여유.
      omega_speed     : 접근 속도 효율.
      omega_power     : 누설 전류 기반 전력 효율.

    판정:
      HEALTHY  (Ω ≥ 0.75)
      STABLE   (0.55 ≤ Ω < 0.75)
      FRAGILE  (0.35 ≤ Ω < 0.55)
      CRITICAL (Ω < 0.35)
    """
    omega_global: float
    verdict: str                            # HEALTHY / STABLE / FRAGILE / CRITICAL
    omega_retention: float
    omega_margin: float
    omega_endurance: float
    omega_speed: float
    omega_power: float
    flags: List[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "omega_global":    self.omega_global,
            "verdict":         self.verdict,
            "omega_retention": self.omega_retention,
            "omega_margin":    self.omega_margin,
            "omega_endurance": self.omega_endurance,
            "omega_speed":     self.omega_speed,
            "omega_power":     self.omega_power,
            "flags":           list(self.flags),
            "notes":           self.notes,
        }


@dataclass
class VerificationReport:
    """메모리 셀 설계 검증 보고서.

    read_margin      : 읽기 마진 (DRAM: ΔVbl/Vdd − threshold; SRAM: RNM [V]).
                       양수 = 통과, 음수 = 실패.
    write_margin     : 쓰기 마진 (DRAM: 1−threshold 근사; SRAM: WM [V]).
    retention_margin : 보존 마진 (DRAM: τ/τ_ref 배율; SRAM: stability index 0~1).
    endurance_margin : 내구성 여유 (0~1).
    omega_global     : 관측 Ω 종합값.
    verdict          : PASS / MARGINAL / FAIL.
    notes            : 경고·실패 사유 목록.
    """
    cell_type: str
    verdict: str                    # PASS / MARGINAL / FAIL
    read_margin: float
    write_margin: float
    retention_margin: float
    endurance_margin: float
    omega_global: float
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cell_type":        self.cell_type,
            "verdict":          self.verdict,
            "read_margin":      self.read_margin,
            "write_margin":     self.write_margin,
            "retention_margin": self.retention_margin,
            "endurance_margin": self.endurance_margin,
            "omega_global":     self.omega_global,
            "notes":            list(self.notes),
        }
