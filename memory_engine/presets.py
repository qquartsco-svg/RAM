"""메모리 공정 프리셋 — 실제 공정 규격 기반 파라미터 초깃값.

DRAM 프리셋:
  LPDDR5_PARAMS  : 모바일 LPDDR5  (12nm, Vdd=1.05V, t_access=3.75ns)
  DDR5_PARAMS    : 서버/데스크탑 DDR5  (15nm, Vdd=1.1V, t_access=5ns)
  DDR4_PARAMS    : 레거시 DDR4    (20nm, Vdd=1.2V, t_access=10ns)
  DDR3_PARAMS    : 레거시 DDR3    (30nm, Vdd=1.5V, t_access=13ns)

SRAM 프리셋:
  SRAM_7NM       : 최첨단 7nm (Vdd=0.75V, β=2.0, SNM 작음 → 속도 우선)
  SRAM_14NM      : 14nm FinFET (Vdd=0.8V, β=2.2)
  SRAM_28NM      : 28nm HKMG  (Vdd=1.0V, β=2.5)
  SRAM_65NM      : 65nm Planar (Vdd=1.2V, β=3.0, SNM 큼 → 안정성 우선)

함수:
  get_dram_preset(name, **overrides) → DRAMDesignParams
  get_sram_preset(name, **overrides) → SRAMDesignParams
  list_presets()                     → dict[str, list[str]]
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from .schema import DRAMDesignParams, SRAMDesignParams


# ══════════════════════════════════════════════════════════════════════════════
# DRAM 프리셋
# ══════════════════════════════════════════════════════════════════════════════

LPDDR5_PARAMS = DRAMDesignParams(
    node_nm=12.0,
    vdd_v=1.05,
    T_k=300.0,
    C_s_fF=10.0,
    C_bl_fF=150.0,
    t_ret_ref_s=32e-3,         # 고속 공정 → refresh 주기 단축
    T_ref_k=300.0,
    Ea_eV=0.55,
    sense_threshold_fraction=0.08,
    read_disturb_factor=2.0e-4,
    refresh_recovery=0.45,
    t_access_ns=3.75,
    power_mW_per_cell=2.5e-6,
    max_cycles=200_000_000,
    vdd_min_v=0.90,
    vdd_max_v=1.10,
)

DDR5_PARAMS = DRAMDesignParams(
    node_nm=15.0,
    vdd_v=1.1,
    T_k=300.0,
    C_s_fF=15.0,
    C_bl_fF=130.0,
    t_ret_ref_s=64e-3,
    T_ref_k=300.0,
    Ea_eV=0.55,
    sense_threshold_fraction=0.10,
    read_disturb_factor=1.8e-4,
    refresh_recovery=0.42,
    t_access_ns=5.0,
    power_mW_per_cell=3.5e-6,
    max_cycles=150_000_000,
    vdd_min_v=1.00,
    vdd_max_v=1.15,
)

DDR4_PARAMS = DRAMDesignParams(
    node_nm=20.0,
    vdd_v=1.2,
    T_k=300.0,
    C_s_fF=20.0,
    C_bl_fF=120.0,
    t_ret_ref_s=64e-3,
    T_ref_k=300.0,
    Ea_eV=0.55,
    sense_threshold_fraction=0.12,
    read_disturb_factor=1.2e-4,
    refresh_recovery=0.40,
    t_access_ns=10.0,
    power_mW_per_cell=5.0e-6,
    max_cycles=100_000_000,
    vdd_min_v=1.10,
    vdd_max_v=1.30,
)

DDR3_PARAMS = DRAMDesignParams(
    node_nm=30.0,
    vdd_v=1.5,
    T_k=300.0,
    C_s_fF=25.0,
    C_bl_fF=100.0,
    t_ret_ref_s=64e-3,
    T_ref_k=300.0,
    Ea_eV=0.55,
    sense_threshold_fraction=0.13,
    read_disturb_factor=1.0e-4,
    refresh_recovery=0.38,
    t_access_ns=13.5,
    power_mW_per_cell=8.0e-6,
    max_cycles=80_000_000,
    vdd_min_v=1.35,
    vdd_max_v=1.60,
)


# ══════════════════════════════════════════════════════════════════════════════
# SRAM 프리셋
# ══════════════════════════════════════════════════════════════════════════════

SRAM_7NM = SRAMDesignParams(
    node_nm=7.0,
    vdd_v=0.75,
    T_k=300.0,
    Vth_n_v=0.20,
    Vth_p_v=0.20,
    beta_ratio=2.0,
    wl_strength=1.2,
    read_margin_factor=0.55,
    write_margin_factor=0.38,
    t_access_ns=0.3,
    leakage_nA=0.5,
    max_cycles=10 ** 12,
)

SRAM_14NM = SRAMDesignParams(
    node_nm=14.0,
    vdd_v=0.8,
    T_k=300.0,
    Vth_n_v=0.25,
    Vth_p_v=0.25,
    beta_ratio=2.2,
    wl_strength=1.1,
    read_margin_factor=0.58,
    write_margin_factor=0.40,
    t_access_ns=0.5,
    leakage_nA=0.8,
    max_cycles=10 ** 12,
)

SRAM_28NM = SRAMDesignParams(
    node_nm=28.0,
    vdd_v=1.0,
    T_k=300.0,
    Vth_n_v=0.35,
    Vth_p_v=0.35,
    beta_ratio=2.5,
    wl_strength=1.0,
    read_margin_factor=0.60,
    write_margin_factor=0.42,
    t_access_ns=1.0,
    leakage_nA=2.0,
    max_cycles=10 ** 12,
)

SRAM_65NM = SRAMDesignParams(
    node_nm=65.0,
    vdd_v=1.2,
    T_k=300.0,
    Vth_n_v=0.45,
    Vth_p_v=0.45,
    beta_ratio=3.0,
    wl_strength=1.0,
    read_margin_factor=0.65,
    write_margin_factor=0.45,
    t_access_ns=2.5,
    leakage_nA=5.0,
    max_cycles=10 ** 12,
)


# ══════════════════════════════════════════════════════════════════════════════
# 조회 인터페이스
# ══════════════════════════════════════════════════════════════════════════════

_DRAM_REGISTRY: Dict[str, DRAMDesignParams] = {
    "lpddr5": LPDDR5_PARAMS,
    "ddr5":   DDR5_PARAMS,
    "ddr4":   DDR4_PARAMS,
    "ddr3":   DDR3_PARAMS,
}

_SRAM_REGISTRY: Dict[str, SRAMDesignParams] = {
    "sram_7nm":  SRAM_7NM,
    "sram_14nm": SRAM_14NM,
    "sram_28nm": SRAM_28NM,
    "sram_65nm": SRAM_65NM,
}


def get_dram_preset(name: str, **overrides: Any) -> DRAMDesignParams:
    """DRAM 프리셋 반환. overrides로 일부 파라미터 덮어쓰기.

    Args:
        name     : 프리셋 이름 ('lpddr5' / 'ddr5' / 'ddr4' / 'ddr3').
        **overrides: DRAMDesignParams 필드 덮어쓰기.

    Returns:
        DRAMDesignParams 인스턴스.

    Raises:
        ValueError: 알 수 없는 프리셋 이름.
    """
    key = name.lower()
    if key not in _DRAM_REGISTRY:
        raise ValueError(f"DRAM 프리셋 '{name}' 없음. 사용 가능: {list(_DRAM_REGISTRY)}")
    base = _DRAM_REGISTRY[key]
    return replace(base, **overrides) if overrides else base


def get_sram_preset(name: str, **overrides: Any) -> SRAMDesignParams:
    """SRAM 프리셋 반환. overrides로 일부 파라미터 덮어쓰기.

    Args:
        name     : 프리셋 이름 ('sram_7nm' / 'sram_14nm' / 'sram_28nm' / 'sram_65nm').
        **overrides: SRAMDesignParams 필드 덮어쓰기.

    Returns:
        SRAMDesignParams 인스턴스.

    Raises:
        ValueError: 알 수 없는 프리셋 이름.
    """
    key = name.lower()
    if key not in _SRAM_REGISTRY:
        raise ValueError(f"SRAM 프리셋 '{name}' 없음. 사용 가능: {list(_SRAM_REGISTRY)}")
    base = _SRAM_REGISTRY[key]
    return replace(base, **overrides) if overrides else base


def list_presets() -> Dict[str, List[str]]:
    """사용 가능한 프리셋 이름 목록."""
    return {
        "dram": list(_DRAM_REGISTRY.keys()),
        "sram": list(_SRAM_REGISTRY.keys()),
    }
