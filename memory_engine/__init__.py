"""memory_engine — DRAM / SRAM 메모리 설계·시뮬레이션·검증 엔진.

계층 구조
─────────
  schema.py        — 셀 상태, 설계 파라미터, 관측·검증 스키마
  dram_physics.py  — DRAM 물리 (RC 방전, Arrhenius 보정, read disturb, refresh)
  sram_physics.py  — SRAM 물리 (VTC, SNM, RNM, WM, stability index)
  observer.py      — Ω 5레이어 관측 (DRAM/SRAM) + 진단
  design_engine.py — 시뮬레이션, 파라미터 스윕, 검증 보고서
  presets.py       — 공정 프리셋 (LPDDR5/DDR5/DDR4/DDR3, SRAM 7~65nm)

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

__version__ = "0.1.0"

__all__ = [
    # 스키마
    "MemoryCellType",
    "DRAMCellState",
    "DRAMDesignParams",
    "SRAMCellState",
    "SRAMDesignParams",
    "MemoryObservation",
    "VerificationReport",
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
