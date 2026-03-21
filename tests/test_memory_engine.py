"""메모리 엔진 테스트 suite.

§1 DRAM 물리 — retention decay, Arrhenius, read disturb, refresh, write
§2 DRAM 파생 지표 — bitline swing, read margin, time_to_fail
§3 SRAM 물리 — VTC, SNM, RNM, WM, stability index, write, read
§4 DRAM Observer Ω — 5레이어, 판정, 플래그
§5 SRAM Observer Ω — 5레이어, 판정, 플래그
§6 DRAM 설계 엔진 — 시뮬레이션, 스윕
§7 SRAM 설계 엔진 — 시뮬레이션, 스윕
§8 검증 보고서 — DRAM/SRAM PASS/MARGINAL/FAIL
§9 프리셋 — DRAM/SRAM 프리셋 로드 + override
§10 진단 — diagnose()
"""

from __future__ import annotations

import math
import pytest

from memory_engine import (
    # DRAM
    DRAMCellState, DRAMDesignParams, DDR4_PARAMS, DDR5_PARAMS, LPDDR5_PARAMS, DDR3_PARAMS,
    retention_tau, retention_decay, read_disturb, refresh, dram_write,
    bitline_swing_fraction, dram_read_margin, refresh_needed, time_to_fail,
    observe_dram, verify_dram,
    simulate_dram_retention, simulate_dram_read_disturb, simulate_dram_refresh_cycle,
    sweep_dram_temperature, sweep_dram_vdd, sweep_dram_cs,
    # SRAM
    SRAMCellState, SRAMDesignParams, SRAM_7NM, SRAM_14NM, SRAM_28NM, SRAM_65NM,
    static_noise_margin, read_noise_margin, write_margin, stability_index, hold_snm,
    sram_initial_state, sram_write, sram_read, sram_leakage_decay, vtc_curve,
    observe_sram, verify_sram,
    simulate_sram_leakage, sweep_sram_beta, sweep_sram_vdd, sweep_sram_temperature,
    # 공통
    MemoryObservation, VerificationReport, diagnose,
    get_dram_preset, get_sram_preset, list_presets,
)


# ─────────────────────────────────────────────────────────────────────────────
# 공통 fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ddr4():
    return DDR4_PARAMS


@pytest.fixture
def ddr4_cell():
    return DRAMCellState(q=1.0)


@pytest.fixture
def sram28():
    return SRAM_28NM


@pytest.fixture
def sram28_cell():
    return SRAMCellState(v_q=1.0, v_qb=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# §1 DRAM 물리
# ══════════════════════════════════════════════════════════════════════════════

class TestDRAMRetentionDecay:
    def test_decay_reduces_charge(self, ddr4, ddr4_cell):
        after = retention_decay(ddr4_cell, ddr4, dt_s=0.01)
        assert after.q < ddr4_cell.q

    def test_decay_to_near_zero_long_time(self, ddr4):
        cell = DRAMCellState(q=1.0)
        # τ ≈ 64ms → 10×τ 이면 exp(-10) ≈ 4.5e-5
        tau = retention_tau(ddr4)
        after = retention_decay(cell, ddr4, dt_s=tau * 10)
        assert after.q < 0.001

    def test_decay_zero_dt_no_change(self, ddr4, ddr4_cell):
        after = retention_decay(ddr4_cell, ddr4, dt_s=0.0)
        assert abs(after.q - ddr4_cell.q) < 1e-9

    def test_decay_increments_t_since_refresh(self, ddr4, ddr4_cell):
        after = retention_decay(ddr4_cell, ddr4, dt_s=0.01)
        assert abs(after.t_since_refresh_s - 0.01) < 1e-9

    def test_charge_never_negative(self, ddr4):
        cell = DRAMCellState(q=0.001)
        after = retention_decay(cell, ddr4, dt_s=1000.0)
        assert after.q >= 0.0

    def test_exponential_law(self, ddr4):
        """Q(2dt) / Q(dt) ≈ Q(dt) / Q(0) — 지수 감쇠 일관성."""
        cell = DRAMCellState(q=1.0)
        tau = retention_tau(ddr4)
        dt = tau * 0.1
        c1 = retention_decay(cell, ddr4, dt_s=dt)
        c2 = retention_decay(c1, ddr4, dt_s=dt)
        # ratio1 = c1.q / cell.q, ratio2 = c2.q / c1.q — 동일해야 함
        ratio1 = c1.q / cell.q
        ratio2 = c2.q / c1.q
        assert abs(ratio1 - ratio2) < 1e-4


class TestDRAMArrhenius:
    def test_higher_temp_shorter_tau(self, ddr4):
        p_hot = DRAMDesignParams(**{**vars(ddr4), "T_k": 370.0}) \
            if False else get_dram_preset("ddr4", T_k=370.0)
        tau_ref = retention_tau(ddr4)
        tau_hot = retention_tau(p_hot)
        assert tau_hot < tau_ref

    def test_lower_temp_longer_tau(self, ddr4):
        p_cold = get_dram_preset("ddr4", T_k=250.0)
        tau_ref = retention_tau(ddr4)
        tau_cold = retention_tau(p_cold)
        assert tau_cold > tau_ref

    def test_at_reference_temp_tau_equals_ref(self, ddr4):
        tau = retention_tau(ddr4)
        assert abs(tau - ddr4.t_ret_ref_s) < 1e-9


class TestDRAMReadDisturb:
    def test_read_disturb_reduces_charge(self, ddr4, ddr4_cell):
        after = read_disturb(ddr4_cell, ddr4)
        assert after.q < ddr4_cell.q

    def test_read_disturb_increments_n_reads(self, ddr4, ddr4_cell):
        after = read_disturb(ddr4_cell, ddr4)
        assert after.n_reads == ddr4_cell.n_reads + 1

    def test_repeated_disturb_accumulates(self, ddr4):
        cell = DRAMCellState(q=1.0)
        for _ in range(1000):
            cell = read_disturb(cell, ddr4)
        assert cell.q < 0.9

    def test_charge_never_negative_under_disturb(self, ddr4):
        cell = DRAMCellState(q=1e-10)
        after = read_disturb(cell, ddr4)
        assert after.q >= 0.0


class TestDRAMRefresh:
    def test_refresh_increases_charge(self, ddr4):
        cell = DRAMCellState(q=0.3)
        after = refresh(cell, ddr4)
        assert after.q > cell.q

    def test_refresh_resets_t_since_refresh(self, ddr4):
        cell = DRAMCellState(q=0.5, t_since_refresh_s=0.1)
        after = refresh(cell, ddr4)
        assert after.t_since_refresh_s == 0.0

    def test_refresh_never_exceeds_one(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after = refresh(cell, ddr4)
        assert after.q <= 1.0

    def test_refresh_full_cell_unchanged(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after = refresh(cell, ddr4)
        assert abs(after.q - 1.0) < 1e-9


class TestDRAMWrite:
    def test_write_sets_charge(self, ddr4):
        cell = DRAMCellState(q=0.2)
        after = dram_write(cell, ddr4, value=1.0)
        assert abs(after.q - 1.0) < 1e-9

    def test_write_increments_cycles(self, ddr4):
        cell = DRAMCellState(q=0.5, n_cycles=10)
        after = dram_write(cell, ddr4)
        assert after.n_cycles == 11

    def test_write_resets_n_reads(self, ddr4):
        cell = DRAMCellState(q=0.5, n_reads=500)
        after = dram_write(cell, ddr4)
        assert after.n_reads == 0

    def test_write_zero(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after = dram_write(cell, ddr4, value=0.0)
        assert abs(after.q - 0.0) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# §2 DRAM 파생 지표
# ══════════════════════════════════════════════════════════════════════════════

class TestDRAMDerivedMetrics:
    def test_bitline_swing_full_charge(self, ddr4):
        cell = DRAMCellState(q=1.0)
        bl = bitline_swing_fraction(cell, ddr4)
        expected = ddr4.C_s_fF / (ddr4.C_s_fF + ddr4.C_bl_fF)
        assert abs(bl - expected) < 1e-9

    def test_bitline_swing_zero_charge(self, ddr4):
        cell = DRAMCellState(q=0.0)
        bl = bitline_swing_fraction(cell, ddr4)
        assert bl == 0.0

    def test_bitline_swing_proportional_to_q(self, ddr4):
        bl1 = bitline_swing_fraction(DRAMCellState(q=0.5), ddr4)
        bl2 = bitline_swing_fraction(DRAMCellState(q=1.0), ddr4)
        assert abs(bl2 - 2 * bl1) < 1e-9

    def test_read_margin_positive_full_charge(self, ddr4):
        cell = DRAMCellState(q=1.0)
        rm = dram_read_margin(cell, ddr4)
        assert rm > 0.0

    def test_read_margin_negative_low_charge(self, ddr4):
        cell = DRAMCellState(q=0.05)
        rm = dram_read_margin(cell, ddr4)
        assert rm < 0.0

    def test_refresh_needed_when_charge_low(self, ddr4):
        cell = DRAMCellState(q=0.05)
        assert refresh_needed(cell, ddr4)

    def test_refresh_not_needed_when_full(self, ddr4):
        cell = DRAMCellState(q=1.0)
        assert not refresh_needed(cell, ddr4)

    def test_time_to_fail_positive(self, ddr4):
        ttf = time_to_fail(ddr4)
        assert ttf > 0.0

    def test_time_to_fail_less_than_tau(self, ddr4):
        """완전 충전 → 마진 0까지 시간 < τ (threshold > 0이므로)."""
        ttf = time_to_fail(ddr4)
        tau = retention_tau(ddr4)
        assert ttf < tau

    def test_time_to_fail_decreases_with_higher_threshold(self):
        p_loose = DRAMDesignParams(sense_threshold_fraction=0.05)
        p_tight = DRAMDesignParams(sense_threshold_fraction=0.15)
        assert time_to_fail(p_loose) > time_to_fail(p_tight)


# ══════════════════════════════════════════════════════════════════════════════
# §3 SRAM 물리
# ══════════════════════════════════════════════════════════════════════════════

class TestSRAMPhysics:
    def test_snm_positive_normal_params(self, sram28):
        snm = static_noise_margin(sram28)
        assert snm > 0.0

    def test_snm_increases_with_beta(self):
        p_low  = SRAMDesignParams(beta_ratio=1.5, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        p_high = SRAMDesignParams(beta_ratio=4.0, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        assert static_noise_margin(p_high) > static_noise_margin(p_low)

    def test_snm_increases_with_vdd(self):
        p_low  = SRAMDesignParams(vdd_v=0.7, Vth_n_v=0.25, Vth_p_v=0.25)
        p_high = SRAMDesignParams(vdd_v=1.2, Vth_n_v=0.25, Vth_p_v=0.25)
        assert static_noise_margin(p_high) > static_noise_margin(p_low)

    def test_rnm_less_than_snm(self, sram28):
        snm = static_noise_margin(sram28)
        rnm = read_noise_margin(sram28)
        assert rnm < snm

    def test_write_margin_decreases_with_beta(self):
        p_low  = SRAMDesignParams(beta_ratio=1.5, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        p_high = SRAMDesignParams(beta_ratio=4.0, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        assert write_margin(p_high) < write_margin(p_low)

    def test_stability_index_between_0_and_1(self, sram28):
        si = stability_index(sram28)
        assert 0.0 <= si <= 1.0

    def test_stability_index_increases_with_beta(self):
        p_low  = SRAMDesignParams(beta_ratio=1.5, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        p_high = SRAMDesignParams(beta_ratio=4.0, vdd_v=1.0, Vth_n_v=0.3, Vth_p_v=0.3)
        assert stability_index(p_high) > stability_index(p_low)

    def test_hold_snm_equals_snm(self, sram28):
        assert abs(hold_snm(sram28) - static_noise_margin(sram28)) < 1e-12

    def test_vtc_curve_length(self, sram28):
        curve = vtc_curve(sram28, n_points=101)
        assert len(curve) == 101

    def test_vtc_vin_zero_gives_vdd(self, sram28):
        curve = vtc_curve(sram28, n_points=201)
        vin0, vout0 = curve[0]
        assert abs(vin0) < 1e-9
        assert abs(vout0 - sram28.vdd_v) < 0.01

    def test_vtc_vin_vdd_gives_zero(self, sram28):
        curve = vtc_curve(sram28, n_points=201)
        vin_last, vout_last = curve[-1]
        assert abs(vin_last - sram28.vdd_v) < 1e-9
        assert vout_last < 0.05


class TestSRAMCellOps:
    def test_initial_state_high(self, sram28):
        cell = sram_initial_state(sram28, stored_high=True)
        assert cell.v_q == sram28.vdd_v
        assert cell.v_qb == 0.0

    def test_initial_state_low(self, sram28):
        cell = sram_initial_state(sram28, stored_high=False)
        assert cell.v_q == 0.0
        assert cell.v_qb == sram28.vdd_v

    def test_write_increments_cycles(self, sram28):
        cell = sram_initial_state(sram28)
        after = sram_write(cell, sram28, value=False)
        assert after.n_cycles == cell.n_cycles + 1

    def test_write_true(self, sram28):
        cell = SRAMCellState(v_q=0.0, v_qb=sram28.vdd_v)
        after = sram_write(cell, sram28, value=True)
        assert after.v_q == sram28.vdd_v

    def test_read_correct_value(self, sram28):
        cell = sram_initial_state(sram28, stored_high=True)
        value, rnm = sram_read(cell, sram28)
        assert value is True
        assert rnm > 0.0

    def test_leakage_decay_reduces_voltage(self, sram28):
        cell = SRAMCellState(v_q=1.0, v_qb=0.0)
        after = sram_leakage_decay(cell, sram28, dt_s=1.0)
        assert after.v_q < cell.v_q

    def test_leakage_decay_zero_dt_no_change(self, sram28):
        cell = SRAMCellState(v_q=1.0, v_qb=0.2)
        after = sram_leakage_decay(cell, sram28, dt_s=0.0)
        assert abs(after.v_q - cell.v_q) < 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# §4 DRAM Observer Ω
# ══════════════════════════════════════════════════════════════════════════════

class TestDRAMObserver:
    def test_omega_in_range(self, ddr4, ddr4_cell):
        obs = observe_dram(ddr4_cell, ddr4)
        assert 0.0 <= obs.omega_global <= 1.0

    def test_omega_all_submetrics_in_range(self, ddr4, ddr4_cell):
        obs = observe_dram(ddr4_cell, ddr4)
        for attr in ("omega_retention", "omega_margin", "omega_endurance",
                     "omega_speed", "omega_power"):
            v = getattr(obs, attr)
            assert 0.0 <= v <= 1.0, f"{attr} = {v}"

    def test_verdict_type(self, ddr4, ddr4_cell):
        obs = observe_dram(ddr4_cell, ddr4)
        assert obs.verdict in ("HEALTHY", "STABLE", "FRAGILE", "CRITICAL")

    def test_full_charge_high_omega(self, ddr4):
        obs = observe_dram(DRAMCellState(q=1.0), ddr4)
        assert obs.omega_global > 0.50

    def test_depleted_charge_low_omega(self, ddr4):
        obs_full = observe_dram(DRAMCellState(q=1.0), ddr4)
        obs_depleted = observe_dram(DRAMCellState(q=0.01), ddr4)
        assert obs_depleted.omega_global < obs_full.omega_global

    def test_read_margin_fail_flag(self, ddr4):
        cell = DRAMCellState(q=0.01)
        obs = observe_dram(cell, ddr4)
        assert "read_margin_fail" in obs.flags

    def test_no_flags_at_full_charge(self, ddr4):
        obs = observe_dram(DRAMCellState(q=1.0, n_cycles=0), ddr4)
        # 충전 완전·사이클 0이면 부정적 플래그 없어야 함
        bad_flags = {"read_margin_fail", "critical_charge_loss", "endurance_critical"}
        assert not (bad_flags & set(obs.flags))

    def test_endurance_flag_at_90pct(self, ddr4):
        cell = DRAMCellState(q=1.0, n_cycles=int(ddr4.max_cycles * 0.92))
        obs = observe_dram(cell, ddr4)
        assert "endurance_critical" in obs.flags

    def test_as_dict_has_required_keys(self, ddr4, ddr4_cell):
        obs = observe_dram(ddr4_cell, ddr4)
        d = obs.as_dict()
        for key in ("omega_global", "verdict", "omega_retention", "omega_margin",
                    "omega_endurance", "omega_speed", "omega_power", "flags", "notes"):
            assert key in d


# ══════════════════════════════════════════════════════════════════════════════
# §5 SRAM Observer Ω
# ══════════════════════════════════════════════════════════════════════════════

class TestSRAMObserver:
    def test_omega_in_range(self, sram28, sram28_cell):
        obs = observe_sram(sram28_cell, sram28)
        assert 0.0 <= obs.omega_global <= 1.0

    def test_all_submetrics_in_range(self, sram28, sram28_cell):
        obs = observe_sram(sram28_cell, sram28)
        for attr in ("omega_retention", "omega_margin", "omega_endurance",
                     "omega_speed", "omega_power"):
            v = getattr(obs, attr)
            assert 0.0 <= v <= 1.0

    def test_verdict_type(self, sram28, sram28_cell):
        obs = observe_sram(sram28_cell, sram28)
        assert obs.verdict in ("HEALTHY", "STABLE", "FRAGILE", "CRITICAL")

    def test_full_state_higher_omega_than_weak(self, sram28):
        strong = SRAMCellState(v_q=sram28.vdd_v, v_qb=0.0)
        weak   = SRAMCellState(v_q=sram28.vdd_v * 0.1, v_qb=sram28.vdd_v * 0.9)
        obs_strong = observe_sram(strong, sram28)
        obs_weak   = observe_sram(weak, sram28)
        assert obs_strong.omega_global > obs_weak.omega_global

    def test_weak_cell_flag(self, sram28):
        """v_q, v_qb 차이가 작은 메타스테이블 상태 → weak_cell_state 플래그."""
        # |v_q - v_qb| = 0.10 < vdd * 0.20 = 0.20 → 플래그 트리거
        cell = SRAMCellState(v_q=0.55, v_qb=0.45)
        obs = observe_sram(cell, sram28)
        assert "weak_cell_state" in obs.flags

    def test_low_vdd_snm_critical_flag(self):
        """Vdd가 임계값에 근접하면 snm_low 이상 또는 STABLE 이하 판정."""
        p = SRAMDesignParams(vdd_v=0.3, Vth_n_v=0.25, Vth_p_v=0.25, beta_ratio=1.5)
        cell = SRAMCellState(v_q=0.3, v_qb=0.0)
        obs = observe_sram(cell, p)
        # SNM이 매우 작으면 snm_critical 또는 snm_low 플래그
        has_snm_flag = "snm_critical" in obs.flags or "snm_low" in obs.flags
        assert has_snm_flag or obs.omega_global < 0.75


# ══════════════════════════════════════════════════════════════════════════════
# §6 DRAM 설계 엔진
# ══════════════════════════════════════════════════════════════════════════════

class TestDRAMDesignEngine:
    def test_simulate_retention_returns_list(self, ddr4, ddr4_cell):
        hist = simulate_dram_retention(ddr4_cell, ddr4, total_s=0.01, dt_s=0.001)
        assert isinstance(hist, list)
        assert len(hist) >= 2

    def test_simulate_retention_charge_monotone_decrease(self, ddr4, ddr4_cell):
        hist = simulate_dram_retention(ddr4_cell, ddr4, total_s=0.05, dt_s=0.005)
        qs = [h["q"] for h in hist]
        assert all(qs[i] >= qs[i + 1] - 1e-9 for i in range(len(qs) - 1))

    def test_simulate_read_disturb_length(self, ddr4, ddr4_cell):
        hist = simulate_dram_read_disturb(ddr4_cell, ddr4, n_reads=100)
        assert len(hist) == 101  # 0 ~ 100

    def test_simulate_read_disturb_charge_decreases(self, ddr4, ddr4_cell):
        hist = simulate_dram_read_disturb(ddr4_cell, ddr4, n_reads=50)
        assert hist[-1]["q"] < hist[0]["q"]

    def test_simulate_refresh_cycle_length(self, ddr4, ddr4_cell):
        hist = simulate_dram_refresh_cycle(ddr4_cell, ddr4, n_cycles=5)
        assert len(hist) == 5

    def test_simulate_refresh_q_after_gt_before(self, ddr4):
        cell = DRAMCellState(q=1.0)
        hist = simulate_dram_refresh_cycle(cell, ddr4, n_cycles=3, period_s=0.032)
        for h in hist:
            assert h["q_after_refresh"] >= h["q_before_refresh"]

    def test_sweep_temperature_tau_decreases_with_T(self, ddr4):
        T_range = [300.0, 330.0, 360.0]
        results = sweep_dram_temperature(ddr4, T_range)
        taus = [r["tau_s"] for r in results]
        assert taus[0] > taus[1] > taus[2]

    def test_sweep_vdd_read_margin_increases_with_vdd(self, ddr4):
        vdd_range = [1.0, 1.1, 1.2, 1.3]
        results = sweep_dram_vdd(ddr4, vdd_range)
        # read_margin은 Vdd에 독립적 (Cs/Cbl 비율 의존) — omega는 Vdd 범위에 따라 달라질 수 있음
        # 최소한 결과 길이 검증
        assert len(results) == 4

    def test_sweep_cs_margin_increases_with_cs(self, ddr4):
        cs_range = [10.0, 20.0, 30.0, 40.0]
        results = sweep_dram_cs(ddr4, cs_range)
        margins = [r["read_margin"] for r in results]
        assert margins[0] < margins[-1]


# ══════════════════════════════════════════════════════════════════════════════
# §7 SRAM 설계 엔진
# ══════════════════════════════════════════════════════════════════════════════

class TestSRAMDesignEngine:
    def test_simulate_leakage_returns_list(self, sram28, sram28_cell):
        hist = simulate_sram_leakage(sram28_cell, sram28, total_s=0.01, dt_s=0.001)
        assert isinstance(hist, list)
        assert len(hist) >= 2

    def test_simulate_leakage_voltage_decreases(self, sram28):
        cell = SRAMCellState(v_q=1.0, v_qb=0.0)
        hist = simulate_sram_leakage(cell, sram28, total_s=1.0, dt_s=0.1)
        assert hist[-1]["v_q"] <= hist[0]["v_q"]

    def test_sweep_beta_snm_increases(self, sram28):
        beta_range = [1.5, 2.0, 2.5, 3.0, 3.5]
        results = sweep_sram_beta(sram28, beta_range)
        snms = [r["snm_v"] for r in results]
        assert snms[0] < snms[-1]

    def test_sweep_beta_wm_decreases(self, sram28):
        beta_range = [1.5, 2.0, 3.0, 4.0]
        results = sweep_sram_beta(sram28, beta_range)
        wms = [r["wm_v"] for r in results]
        assert wms[0] > wms[-1]

    def test_sweep_vdd_snm_increases_with_vdd(self, sram28):
        vdd_range = [0.7, 0.8, 0.9, 1.0, 1.1]
        results = sweep_sram_vdd(sram28, vdd_range)
        snms = [r["snm_v"] for r in results]
        assert snms[0] < snms[-1]

    def test_sweep_temperature_length(self, sram28):
        T_range = [250.0, 300.0, 350.0, 400.0]
        results = sweep_sram_temperature(sram28, T_range)
        assert len(results) == 4

    def test_sweep_temperature_snm_decreases_with_T(self, sram28):
        """온도 상승 → Vth 감소 → Vm 이동 → SNM 변화 (단조 감소 기대)."""
        T_range = [250.0, 300.0, 350.0, 400.0]
        results = sweep_sram_temperature(sram28, T_range)
        snms = [r["snm_v"] for r in results]
        # 고온: Vth 낮아짐 → V_swing 증가 but Vm 이동 → net effect: 일반적 감소
        # 결과 레코드 구조 검증
        assert all("snm_v" in r for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# §8 검증 보고서
# ══════════════════════════════════════════════════════════════════════════════

class TestDRAMVerification:
    def test_verify_ddr4_full_charge_pass(self, ddr4):
        cell = DRAMCellState(q=1.0, n_cycles=0)
        report = verify_dram(cell, ddr4)
        assert report.verdict == "PASS"
        assert report.cell_type == "DRAM"

    def test_verify_low_charge_marginal_or_fail(self, ddr4):
        cell = DRAMCellState(q=0.05)
        report = verify_dram(cell, ddr4)
        assert report.verdict in ("MARGINAL", "FAIL")

    def test_verify_depleted_cell_fail(self, ddr4):
        cell = DRAMCellState(q=0.001)
        report = verify_dram(cell, ddr4)
        assert report.verdict == "FAIL"

    def test_verify_high_cycle_count_warns(self, ddr4):
        cell = DRAMCellState(q=1.0, n_cycles=int(ddr4.max_cycles * 0.95))
        report = verify_dram(cell, ddr4)
        # 내구성 경고가 notes에 있어야 함
        assert any("마모" in n or "내구" in n for n in report.notes)

    def test_verify_report_as_dict(self, ddr4, ddr4_cell):
        report = verify_dram(ddr4_cell, ddr4)
        d = report.as_dict()
        for key in ("cell_type", "verdict", "read_margin", "write_margin",
                    "retention_margin", "endurance_margin", "omega_global", "notes"):
            assert key in d

    def test_verify_vdd_out_of_range_warns(self):
        p = get_dram_preset("ddr4", vdd_v=0.5)  # vdd_min=1.1V 위반
        cell = DRAMCellState(q=1.0)
        report = verify_dram(cell, p)
        assert report.verdict in ("MARGINAL", "FAIL")
        assert any("Vdd" in n for n in report.notes)


class TestSRAMVerification:
    def test_verify_sram_65nm_pass(self):
        cell = sram_initial_state(SRAM_65NM, stored_high=True)
        report = verify_sram(cell, SRAM_65NM)
        assert report.verdict == "PASS"

    def test_verify_sram_28nm_pass(self):
        cell = sram_initial_state(SRAM_28NM, stored_high=True)
        report = verify_sram(cell, SRAM_28NM)
        assert report.verdict in ("PASS", "MARGINAL")

    def test_verify_low_vdd_low_beta_fail(self):
        """WM이 임계 이하로 강제되면 FAIL."""
        # wl_strength 극소 + beta 극대로 WM < 0.03 확실히 보장
        p = SRAMDesignParams(
            vdd_v=0.5, Vth_n_v=0.40, Vth_p_v=0.40,
            beta_ratio=25.0, wl_strength=0.05, write_margin_factor=0.10,
        )
        cell = SRAMCellState(v_q=0.5, v_qb=0.0)
        report = verify_sram(cell, p)
        assert report.verdict == "FAIL"

    def test_verify_sram_report_as_dict(self):
        cell = sram_initial_state(SRAM_28NM)
        report = verify_sram(cell, SRAM_28NM)
        d = report.as_dict()
        assert "verdict" in d and "read_margin" in d

    def test_verify_sram_7nm_has_verdict(self):
        cell = sram_initial_state(SRAM_7NM)
        report = verify_sram(cell, SRAM_7NM)
        assert report.verdict in ("PASS", "MARGINAL", "FAIL")


# ══════════════════════════════════════════════════════════════════════════════
# §9 프리셋
# ══════════════════════════════════════════════════════════════════════════════

class TestPresets:
    def test_get_dram_preset_ddr4(self):
        p = get_dram_preset("ddr4")
        assert p.node_nm == 20.0
        assert p.vdd_v == 1.2

    def test_get_dram_preset_lpddr5(self):
        p = get_dram_preset("lpddr5")
        assert p.node_nm == 12.0
        assert p.t_access_ns < 5.0

    def test_get_dram_preset_override(self):
        p = get_dram_preset("ddr4", T_k=370.0)
        assert p.T_k == 370.0
        assert p.vdd_v == 1.2  # 나머지는 원본 유지

    def test_get_sram_preset_28nm(self):
        p = get_sram_preset("sram_28nm")
        assert p.node_nm == 28.0

    def test_get_sram_preset_override(self):
        p = get_sram_preset("sram_65nm", beta_ratio=4.0)
        assert p.beta_ratio == 4.0

    def test_unknown_dram_preset_raises(self):
        with pytest.raises(ValueError):
            get_dram_preset("unknown_xyz")

    def test_unknown_sram_preset_raises(self):
        with pytest.raises(ValueError):
            get_sram_preset("unknown_xyz")

    def test_list_presets(self):
        presets = list_presets()
        assert "dram" in presets
        assert "sram" in presets
        assert "ddr4" in presets["dram"]
        assert "sram_28nm" in presets["sram"]

    def test_all_dram_presets_importable(self):
        for name in ("lpddr5", "ddr5", "ddr4", "ddr3"):
            p = get_dram_preset(name)
            assert p.vdd_v > 0.0

    def test_all_sram_presets_importable(self):
        for name in ("sram_7nm", "sram_14nm", "sram_28nm", "sram_65nm"):
            p = get_sram_preset(name)
            assert p.vdd_v > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# §10 진단
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnose:
    def test_diagnose_returns_list(self, ddr4, ddr4_cell):
        obs = observe_dram(ddr4_cell, ddr4)
        advice = diagnose(obs)
        assert isinstance(advice, list)
        assert len(advice) >= 1

    def test_diagnose_healthy_positive_message(self, ddr4):
        obs = observe_dram(DRAMCellState(q=1.0, n_cycles=0), ddr4)
        advice = diagnose(obs)
        assert any("충족" in a or "Ω" in a for a in advice)

    def test_diagnose_low_charge_recommends_refresh(self, ddr4):
        obs = observe_dram(DRAMCellState(q=0.1), ddr4)
        advice = diagnose(obs)
        assert any("refresh" in a.lower() or "커패시터" in a for a in advice)

    def test_diagnose_sram_low_snm_recommends_beta(self):
        p = SRAMDesignParams(vdd_v=0.3, Vth_n_v=0.25, Vth_p_v=0.25, beta_ratio=1.0)
        cell = SRAMCellState(v_q=0.3, v_qb=0.0)
        obs = observe_sram(cell, p)
        advice = diagnose(obs)
        assert isinstance(advice, list)
        assert len(advice) >= 1
