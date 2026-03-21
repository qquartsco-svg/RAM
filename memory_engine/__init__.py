"""memory_engine — DRAM / SRAM 메모리 설계·시뮬레이션·검증 엔진.

계층 구조
─────────
  schema.py           — 셀 상태, 설계 파라미터, 관측·검증 스키마 (SAParams, SAObservation 포함)
  dram_physics.py     — DRAM 물리 (RC 방전, Arrhenius, read disturb, refresh, Row Hammer)
  sram_physics.py     — SRAM 물리 (VTC, SNM, RNM, WM, 3모드 분석)
  sense_amplifier.py  — 센스앰프 (differential latch, BER, Q-function)
  observer.py         — Ω 5레이어 관측 (DRAM/SRAM) + 진단
  design_engine.py    — 시뮬레이션, 파라미터 스윕, 검증 보고서
  presets.py          — 공정 프리셋 (LPDDR5/DDR5/DDR4/DDR3, SRAM 7~65nm, SA 7/28/65nm)

빠른 시작
─────────
  from memory_engine import (
      DRAMCellState, DDR4_PARAMS,
      retention_decay, read_disturb, refresh,
      observe_dram, verify_dram,
      sweep_dram_temperature,
  )

  cell = DRAMCellState(q=1.0)
  cell = retention_decay(cell, DDR4_PARAMS, dt_s=32e-3)
  obs  = observe_dram(cell, DDR4_PARAMS)
  print(obs.verdict, obs.omega_global)

  report = verify_dram(cell, DDR4_PARAMS)
  print(report.verdict, report.notes)
"""

from .schema import (
    MemoryCellType,
    DRAMCellState,
    DRAMDesignParams,
    SRAMCellState,
    SRAMDesignParams,
    MemoryObservation,
    VerificationReport,
    SAParams,
    SAObservation,
)
from .dram_physics import (
    retention_tau,
    retention_decay,
    read_disturb,
    refresh,
    write as dram_write,
    bitline_swing_fraction,
    read_margin as dram_read_margin,
    refresh_needed,
    time_to_fail,
    row_hammer_disturb,
    row_hammer_failure_threshold,
)
from .sense_amplifier import (
    sense_op,
    sa_bit_error_rate,
    sa_row_ber,
    sa_min_delta_v_for_ber,
    SA_7NM as SA_PARAMS_7NM,
    SA_28NM as SA_PARAMS_28NM,
    SA_65NM as SA_PARAMS_65NM,
)
from .sram_physics import (
    vtc_curve,
    static_noise_margin,
    read_noise_margin,
    write_margin,
    stability_index,
    hold_snm,
    sram_initial_state,
    sram_write,
    sram_read,
    sram_leakage_decay,
    read_node_disturb_v,
    read_snm_physical,
    snm_by_mode,
)
from .observer import (
    observe_dram,
    observe_sram,
    diagnose,
)
from .design_engine import (
    simulate_dram_retention,
    simulate_dram_read_disturb,
    simulate_dram_refresh_cycle,
    sweep_dram_temperature,
    sweep_dram_vdd,
    sweep_dram_cs,
    verify_dram,
    simulate_sram_leakage,
    sweep_sram_beta,
    sweep_sram_vdd,
    sweep_sram_temperature,
    verify_sram,
    # A1: Sense Amplifier
    simulate_sa_vs_retention,
    sweep_sa_offset,
    # A2: Row Hammer
    simulate_row_hammer,
    find_row_hammer_threshold,
    # A3: SRAM 3모드
    sram_mode_analysis,
    sweep_sram_mode_beta,
)
from .presets import (
    LPDDR5_PARAMS,
    DDR5_PARAMS,
    DDR4_PARAMS,
    DDR3_PARAMS,
    SRAM_7NM,
    SRAM_14NM,
    SRAM_28NM,
    SRAM_65NM,
    get_dram_preset,
    get_sram_preset,
    list_presets,
)

__version__ = "0.2.0"

__all__ = [
    # 스키마
    "MemoryCellType",
    "DRAMCellState",
    "DRAMDesignParams",
    "SRAMCellState",
    "SRAMDesignParams",
    "MemoryObservation",
    "VerificationReport",
    # 센스앰프
    "SAParams",
    "SAObservation",
    "sense_op",
    "sa_bit_error_rate",
    "sa_row_ber",
    "sa_min_delta_v_for_ber",
    "SA_PARAMS_7NM",
    "SA_PARAMS_28NM",
    "SA_PARAMS_65NM",
    # DRAM 물리
    "retention_tau",
    "retention_decay",
    "read_disturb",
    "refresh",
    "dram_write",
    "bitline_swing_fraction",
    "dram_read_margin",
    "refresh_needed",
    "time_to_fail",
    "row_hammer_disturb",
    "row_hammer_failure_threshold",
    # SRAM 물리
    "vtc_curve",
    "static_noise_margin",
    "read_noise_margin",
    "write_margin",
    "stability_index",
    "hold_snm",
    "sram_initial_state",
    "sram_write",
    "sram_read",
    "sram_leakage_decay",
    "read_node_disturb_v",
    "read_snm_physical",
    "snm_by_mode",
    # Observer
    "observe_dram",
    "observe_sram",
    "diagnose",
    # 설계 엔진
    "simulate_dram_retention",
    "simulate_dram_read_disturb",
    "simulate_dram_refresh_cycle",
    "sweep_dram_temperature",
    "sweep_dram_vdd",
    "sweep_dram_cs",
    "verify_dram",
    "simulate_sram_leakage",
    "sweep_sram_beta",
    "sweep_sram_vdd",
    "sweep_sram_temperature",
    "verify_sram",
    # A1 Sense Amplifier 엔진
    "simulate_sa_vs_retention",
    "sweep_sa_offset",
    # A2 Row Hammer 엔진
    "simulate_row_hammer",
    "find_row_hammer_threshold",
    # A3 SRAM 3모드 분석
    "sram_mode_analysis",
    "sweep_sram_mode_beta",
    # 프리셋
    "LPDDR5_PARAMS",
    "DDR5_PARAMS",
    "DDR4_PARAMS",
    "DDR3_PARAMS",
    "SRAM_7NM",
    "SRAM_14NM",
    "SRAM_28NM",
    "SRAM_65NM",
    "get_dram_preset",
    "get_sram_preset",
    "list_presets",
    "__version__",
]
