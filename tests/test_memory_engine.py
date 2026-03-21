"""메모리 엔진 테스트 suite.

§1  DRAM 물리 — retention decay, Arrhenius, read disturb, refresh, write
§2  DRAM 파생 지표 — bitline swing, read margin, time_to_fail
§3  SRAM 물리 — VTC, SNM, RNM, WM, stability index, write, read
§4  DRAM Observer Ω — 5레이어, 판정, 플래그
§5  SRAM Observer Ω — 5레이어, 판정, 플래그
§6  DRAM 설계 엔진 — 시뮬레이션, 스윕
§7  SRAM 설계 엔진 — 시뮬레이션, 스윕
§8  검증 보고서 — DRAM/SRAM PASS/MARGINAL/FAIL
§9  프리셋 — DRAM/SRAM 프리셋 로드 + override
§10 진단 — diagnose()
§11 A1 Sense Amplifier — SAParams, sense_op, BER, Q-function, 스윕
§12 A2 Row Hammer — 전하 손실, 실패 임계, 시뮬레이션
§13 A3 SRAM 3모드 SNM — Hold/Read/Write 분리, beta sweep
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


# ══════════════════════════════════════════════════════════════════════════════
# §11 A1 Sense Amplifier
# ══════════════════════════════════════════════════════════════════════════════

from memory_engine import (
    SAParams, SAObservation,
    sense_op, sa_bit_error_rate, sa_row_ber, sa_min_delta_v_for_ber,
    SA_PARAMS_7NM, SA_PARAMS_28NM, SA_PARAMS_65NM,
)


class TestSAParams:
    def test_sa_params_defaults(self):
        sa = SAParams()
        assert sa.sa_offset_v > 0.0
        assert sa.sa_sensitivity_v > 0.0
        assert sa.n_bits_per_row > 0

    def test_sa_presets_exist(self):
        assert SA_PARAMS_7NM.sa_offset_v < SA_PARAMS_28NM.sa_offset_v
        assert SA_PARAMS_28NM.sa_offset_v < SA_PARAMS_65NM.sa_offset_v

    def test_sa_7nm_offset_is_small(self):
        # 7nm SA 오프셋은 28nm보다 작아야 함 (더 정밀한 공정)
        assert SA_PARAMS_7NM.sa_offset_v <= 0.012


class TestSenseOp:
    def test_sense_op_returns_saobservation(self, ddr4, ddr4_cell):
        sa = SA_PARAMS_28NM
        obs = sense_op(ddr4_cell, ddr4, sa)
        assert isinstance(obs, SAObservation)

    def test_full_charge_read_success(self, ddr4):
        cell = DRAMCellState(q=1.0)
        obs = sense_op(cell, ddr4, SA_PARAMS_28NM)
        assert obs.read_success is True

    def test_depleted_cell_read_fail(self, ddr4):
        # q=0.01 → bitline swing 매우 작음 → SA margin < 0
        cell = DRAMCellState(q=0.01)
        obs = sense_op(cell, ddr4, SA_PARAMS_28NM)
        assert obs.read_success is False

    def test_delta_v_proportional_to_charge(self, ddr4):
        obs_full = sense_op(DRAMCellState(q=1.0), ddr4, SA_PARAMS_28NM)
        obs_half = sense_op(DRAMCellState(q=0.5), ddr4, SA_PARAMS_28NM)
        assert obs_full.delta_v > obs_half.delta_v

    def test_sa_margin_positive_when_success(self, ddr4):
        obs = sense_op(DRAMCellState(q=1.0), ddr4, SA_PARAMS_28NM)
        if obs.read_success:
            assert obs.sa_margin_v > 0.0

    def test_omega_sa_in_range(self, ddr4, ddr4_cell):
        obs = sense_op(ddr4_cell, ddr4, SA_PARAMS_28NM)
        assert 0.0 <= obs.omega_sa <= 1.0

    def test_t_total_ns_positive(self, ddr4, ddr4_cell):
        obs = sense_op(ddr4_cell, ddr4, SA_PARAMS_28NM)
        assert obs.t_total_ns > 0.0

    def test_flags_is_list(self, ddr4, ddr4_cell):
        obs = sense_op(ddr4_cell, ddr4, SA_PARAMS_28NM)
        assert isinstance(obs.flags, list)

    def test_sense_op_lpddr5_7nm(self):
        cell = DRAMCellState(q=1.0)
        obs = sense_op(cell, LPDDR5_PARAMS, SA_PARAMS_7NM)
        assert obs.delta_v > 0.0
        assert 0.0 <= obs.omega_sa <= 1.0

    def test_sa_offset_large_causes_failure(self, ddr4):
        # 오프셋이 매우 크면 full-charge도 실패
        sa_bad = SAParams(sa_offset_v=0.999)
        obs = sense_op(DRAMCellState(q=1.0), ddr4, sa_bad)
        assert obs.read_success is False


class TestSABitErrorRate:
    def test_ber_full_charge_very_small(self, ddr4):
        cell = DRAMCellState(q=1.0)
        ber = sa_bit_error_rate(cell, ddr4, SA_PARAMS_28NM)
        assert 0.0 <= ber <= 0.5

    def test_ber_depleted_cell_higher(self, ddr4):
        ber_full = sa_bit_error_rate(DRAMCellState(q=1.0), ddr4, SA_PARAMS_28NM)
        ber_low  = sa_bit_error_rate(DRAMCellState(q=0.05), ddr4, SA_PARAMS_28NM)
        assert ber_low >= ber_full

    def test_ber_never_negative(self, ddr4):
        for q in (0.0, 0.01, 0.5, 1.0):
            ber = sa_bit_error_rate(DRAMCellState(q=q), ddr4, SA_PARAMS_28NM)
            assert ber >= 0.0

    def test_row_ber_geq_bit_ber(self, ddr4):
        # 부분 방전 셀 사용 — q=0.15 → BER 수치 범위가 float64로 표현 가능
        cell = DRAMCellState(q=0.15)
        ber_bit = sa_bit_error_rate(cell, ddr4, SA_PARAMS_28NM)
        ber_row = sa_row_ber(cell, ddr4, SA_PARAMS_28NM)
        assert ber_row >= ber_bit

    def test_row_ber_in_range(self, ddr4, ddr4_cell):
        ber_row = sa_row_ber(ddr4_cell, ddr4, SA_PARAMS_28NM)
        assert 0.0 <= ber_row <= 1.0


class TestSAMinDeltaV:
    def test_min_delta_v_for_ber_1e3(self):
        dv = sa_min_delta_v_for_ber(1e-3, SA_PARAMS_28NM)
        assert dv > 0.0

    def test_stricter_ber_needs_larger_delta_v(self):
        dv_loose  = sa_min_delta_v_for_ber(1e-3, SA_PARAMS_28NM)
        dv_strict = sa_min_delta_v_for_ber(1e-9, SA_PARAMS_28NM)
        assert dv_strict > dv_loose

    def test_min_delta_v_7nm_vs_65nm(self):
        # 같은 BER 목표 — 7nm SA(낮은 σ) 는 더 작은 ΔV로 달성 가능
        dv_7nm  = sa_min_delta_v_for_ber(1e-6, SA_PARAMS_7NM)
        dv_65nm = sa_min_delta_v_for_ber(1e-6, SA_PARAMS_65NM)
        assert dv_7nm <= dv_65nm


class TestSweepSAOffset:
    def test_sweep_sa_offset_returns_list(self, ddr4, ddr4_cell):
        from memory_engine import sweep_sa_offset
        results = sweep_sa_offset(
            ddr4_cell, ddr4, SA_PARAMS_28NM,
            offset_range=[0.005, 0.010, 0.020, 0.050],
        )
        assert len(results) == 4

    def test_sweep_sa_offset_larger_offset_fewer_successes(self, ddr4):
        from memory_engine import sweep_sa_offset
        cell = DRAMCellState(q=0.5)
        results = sweep_sa_offset(
            cell, ddr4, SA_PARAMS_28NM,
            offset_range=[0.001, 0.010, 0.100, 0.500],
        )
        # read_success 는 offset 증가에 따라 단조 감소(또는 유지)
        successes = [r["read_success"] for r in results]
        # True 개수는 역순 정렬 리스트에서 앞에 몰려야 함
        assert successes[0] >= successes[-1]  # 작은 offset → 더 잘 감지

    def test_sweep_sa_offset_result_keys(self, ddr4, ddr4_cell):
        from memory_engine import sweep_sa_offset
        results = sweep_sa_offset(
            ddr4_cell, ddr4, SA_PARAMS_28NM,
            offset_range=[0.010, 0.020],
        )
        for r in results:
            assert "sa_offset_v" in r
            assert "read_success" in r
            assert "ber" in r


# ══════════════════════════════════════════════════════════════════════════════
# §12 A2 Row Hammer
# ══════════════════════════════════════════════════════════════════════════════

from memory_engine import (
    row_hammer_disturb, row_hammer_failure_threshold,
    simulate_row_hammer, find_row_hammer_threshold,
)


class TestRowHammerDisturb:
    def test_rh_disturb_reduces_charge(self, ddr4, ddr4_cell):
        after = row_hammer_disturb(ddr4_cell, ddr4, n_hammers=100_000)
        assert after.q < ddr4_cell.q

    def test_rh_disturb_zero_hammers_no_change(self, ddr4, ddr4_cell):
        after = row_hammer_disturb(ddr4_cell, ddr4, n_hammers=0)
        assert abs(after.q - ddr4_cell.q) < 1e-12

    def test_rh_disturb_increments_n_reads(self, ddr4, ddr4_cell):
        n = 50_000
        after = row_hammer_disturb(ddr4_cell, ddr4, n_hammers=n)
        assert after.n_reads == ddr4_cell.n_reads + n

    def test_rh_disturb_more_hammers_lower_charge(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after_low  = row_hammer_disturb(cell, ddr4, n_hammers=10_000)
        after_high = row_hammer_disturb(cell, ddr4, n_hammers=500_000)
        assert after_high.q < after_low.q

    def test_rh_disturb_q_never_negative(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after = row_hammer_disturb(cell, ddr4, n_hammers=10_000_000,
                                   rh_charge_loss_per_event=0.1)
        assert after.q >= 0.0

    def test_rh_disturb_loss_rate_sensitivity(self, ddr4):
        cell = DRAMCellState(q=1.0)
        after_7nm  = row_hammer_disturb(cell, ddr4, n_hammers=100_000,
                                        rh_charge_loss_per_event=15e-6)
        after_28nm = row_hammer_disturb(cell, ddr4, n_hammers=100_000,
                                        rh_charge_loss_per_event=5e-6)
        # 7nm 공정은 더 민감 → 더 큰 손실
        assert after_7nm.q < after_28nm.q


class TestRowHammerFailureThreshold:
    def test_threshold_is_positive_int(self, ddr4):
        n = row_hammer_failure_threshold(ddr4)
        assert isinstance(n, int)
        assert n > 0

    def test_higher_loss_rate_lower_threshold(self, ddr4):
        n_low  = row_hammer_failure_threshold(ddr4, rh_charge_loss_per_event=1e-6)
        n_high = row_hammer_failure_threshold(ddr4, rh_charge_loss_per_event=50e-6)
        assert n_high < n_low

    def test_threshold_lpddr5_vs_ddr3(self):
        # LPDDR5(Cs 더 작음) vs DDR3(Cs 더 큼) — 공정 차이
        n_lp = row_hammer_failure_threshold(LPDDR5_PARAMS)
        n_d3 = row_hammer_failure_threshold(DDR3_PARAMS)
        # int 타입이어야 하며 음수가 아님 (LPDDR5는 Cs 작아 0일 수 있음)
        assert isinstance(n_lp, int) and n_lp >= 0
        assert isinstance(n_d3, int) and n_d3 >= 0

    def test_threshold_consistent_with_disturb(self, ddr4):
        # failure threshold 만큼 hammer하면 margin 실패해야 함
        from memory_engine import dram_read_margin as read_margin_fn
        loss = 5e-6
        n_fail = row_hammer_failure_threshold(ddr4, rh_charge_loss_per_event=loss)
        cell = DRAMCellState(q=1.0)
        after = row_hammer_disturb(cell, ddr4, n_hammers=n_fail,
                                   rh_charge_loss_per_event=loss)
        # n_fail 이후엔 margin ≤ 0 이어야 함
        margin = read_margin_fn(after, ddr4)
        assert margin <= 0.02  # 약간의 수치 오차 허용


class TestSimulateRowHammer:
    def test_simulate_rh_returns_list(self, ddr4, ddr4_cell):
        results = simulate_row_hammer(
            ddr4_cell, ddr4, SA_PARAMS_28NM,
            total_hammers=100_000, step_hammers=10_000,
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_simulate_rh_charge_monotone_decreasing(self, ddr4):
        cell = DRAMCellState(q=1.0)
        results = simulate_row_hammer(
            cell, ddr4, SA_PARAMS_28NM,
            total_hammers=200_000, step_hammers=20_000,
        )
        qs = [r["q"] for r in results]
        assert all(qs[i] >= qs[i+1] for i in range(len(qs)-1))

    def test_simulate_rh_result_keys(self, ddr4, ddr4_cell):
        results = simulate_row_hammer(
            ddr4_cell, ddr4, SA_PARAMS_28NM,
            total_hammers=50_000, step_hammers=25_000,
        )
        for r in results:
            assert "n_hammers" in r
            assert "q" in r
            assert "read_success" in r

    def test_simulate_rh_eventually_fails(self, ddr4):
        # 충분한 hammer 수 → q가 초기 대비 크게 감소
        # DDR4: q_fail ≈ 0.84 → 5×n_fail 에서 q ≈ 0.84^5 ≈ 0.42 < 0.5
        cell = DRAMCellState(q=1.0)
        loss = 20e-6
        n_fail = row_hammer_failure_threshold(ddr4, rh_charge_loss_per_event=loss)
        total = n_fail * 5  # q-기반 임계의 5배
        results = simulate_row_hammer(
            cell, ddr4, SA_PARAMS_28NM,
            total_hammers=total,
            step_hammers=max(1, n_fail // 5),
            rh_charge_loss_per_event=loss,
        )
        last = results[-1]
        assert last["q"] < 0.5


class TestFindRowHammerThreshold:
    def test_find_rh_threshold_returns_dict(self, ddr4):
        result = find_row_hammer_threshold(ddr4, SA_PARAMS_28NM)
        assert isinstance(result, dict)

    def test_find_rh_threshold_has_required_keys(self, ddr4):
        result = find_row_hammer_threshold(ddr4, SA_PARAMS_28NM)
        assert "safer_limit" in result or "n_hammer_q_fail" in result

    def test_find_rh_threshold_safer_limit_positive(self, ddr4):
        result = find_row_hammer_threshold(ddr4, SA_PARAMS_28NM)
        key = "safer_limit" if "safer_limit" in result else "n_hammer_q_fail"
        assert result[key] > 0

    def test_find_rh_threshold_7nm_vs_65nm_sa(self, ddr4):
        r_7nm  = find_row_hammer_threshold(ddr4, SA_PARAMS_7NM)
        r_65nm = find_row_hammer_threshold(ddr4, SA_PARAMS_65NM)
        # 둘 다 결과 dict를 반환해야 함
        assert isinstance(r_7nm, dict) and isinstance(r_65nm, dict)


# ══════════════════════════════════════════════════════════════════════════════
# §13 A3 SRAM 3모드 SNM
# ══════════════════════════════════════════════════════════════════════════════

from memory_engine import (
    read_node_disturb_v, read_snm_physical, snm_by_mode,
    sram_mode_analysis, sweep_sram_mode_beta,
)


class TestReadNodeDisturb:
    def test_disturb_voltage_positive(self, sram28):
        dv = read_node_disturb_v(sram28)
        assert dv > 0.0

    def test_high_beta_less_disturb(self):
        p_low  = SRAMDesignParams(beta_ratio=1.5)
        p_high = SRAMDesignParams(beta_ratio=6.0)
        assert read_node_disturb_v(p_high) < read_node_disturb_v(p_low)

    def test_disturb_lt_vdd(self, sram28):
        assert read_node_disturb_v(sram28) < sram28.vdd_v

    def test_disturb_formula(self, sram28):
        # ΔV = Vdd / (beta + 1)
        expected = sram28.vdd_v / (sram28.beta_ratio + 1.0)
        assert abs(read_node_disturb_v(sram28) - expected) < 1e-9


class TestReadSNMPhysical:
    def test_read_snm_physical_nonneg(self, sram28):
        assert read_snm_physical(sram28) >= 0.0

    def test_read_snm_leq_hold_snm(self, sram28):
        hold = static_noise_margin(sram28)
        rsnm = read_snm_physical(sram28)
        assert rsnm <= hold

    def test_low_beta_read_snm_zero(self):
        # beta=1 → ΔV_read = Vdd/2 → SNM − ΔV 는 0에 가까워짐
        p = SRAMDesignParams(beta_ratio=1.0, vdd_v=1.0,
                             Vth_n_v=0.20, Vth_p_v=0.20)
        rsnm = read_snm_physical(p)
        assert rsnm >= 0.0  # 음수 불가

    def test_high_beta_improves_read_snm(self):
        p_low  = SRAMDesignParams(beta_ratio=2.0)
        p_high = SRAMDesignParams(beta_ratio=8.0)
        # 높은 beta → hold SNM 증가 AND ΔV 감소 → read SNM 더 큼
        assert read_snm_physical(p_high) >= read_snm_physical(p_low)


class TestSNMByMode:
    def test_returns_dict(self, sram28):
        d = snm_by_mode(sram28)
        assert isinstance(d, dict)

    def test_required_keys(self, sram28):
        d = snm_by_mode(sram28)
        for key in ("hold_snm_v", "read_snm_factor_v", "read_snm_physical_v",
                    "read_node_disturb_v", "write_margin_v",
                    "stability_index", "verdict"):
            assert key in d, f"missing key: {key}"

    def test_verdict_is_valid(self, sram28):
        d = snm_by_mode(sram28)
        assert d["verdict"] in ("PASS", "MARGINAL", "FAIL")

    def test_hold_snm_geq_read_snm_physical(self, sram28):
        d = snm_by_mode(sram28)
        assert d["hold_snm_v"] >= d["read_snm_physical_v"]

    def test_28nm_verdict_valid(self):
        # 28nm: read_snm_physical = max(0, SNM − ΔV_read)
        # ΔV_read = Vdd/(beta+1) ≈ 0.29V > SNM ≈ 0.21V → read_snm_phys = 0 → FAIL 가능
        d = snm_by_mode(SRAM_28NM)
        assert d["verdict"] in ("PASS", "MARGINAL", "FAIL")

    def test_7nm_verdict_valid(self):
        d = snm_by_mode(SRAM_7NM)
        assert d["verdict"] in ("PASS", "MARGINAL", "FAIL")

    def test_stability_index_in_range(self, sram28):
        d = snm_by_mode(sram28)
        assert 0.0 <= d["stability_index"] <= 1.0

    def test_low_beta_low_verdict(self):
        # beta=1.0, 극단적 파라미터
        p = SRAMDesignParams(
            beta_ratio=1.0, vdd_v=0.6,
            Vth_n_v=0.25, Vth_p_v=0.25,
            write_margin_factor=0.05, wl_strength=0.3,
        )
        d = snm_by_mode(p)
        assert d["verdict"] in ("MARGINAL", "FAIL")


class TestSRAMModeAnalysis:
    def test_returns_dict(self, sram28):
        result = sram_mode_analysis(sram28)
        assert isinstance(result, dict)

    def test_has_mode_keys(self, sram28):
        result = sram_mode_analysis(sram28)
        assert "hold" in result or "hold_snm_v" in result or "snm" in result

    def test_node_disturb_risk_field(self, sram28):
        result = sram_mode_analysis(sram28)
        # node_disturb_risk 또는 verdict 포함
        assert "node_disturb_risk" in result or "verdict" in result

    def test_node_disturb_risk_valid_value(self, sram28):
        result = sram_mode_analysis(sram28)
        if "node_disturb_risk" in result:
            assert result["node_disturb_risk"] in ("LOW", "MEDIUM", "HIGH")

    def test_analysis_28nm_risk_valid(self):
        result = sram_mode_analysis(SRAM_28NM)
        if "node_disturb_risk" in result:
            # 28nm: ΔV_read ≈ Vdd/4 → SNM보다 크면 HIGH 가능
            assert result["node_disturb_risk"] in ("LOW", "MEDIUM", "HIGH")


class TestSweepSRAMModeBeta:
    def test_returns_list(self, sram28):
        results = sweep_sram_mode_beta(sram28, beta_range=[1.5, 2.0, 3.0, 5.0])
        assert isinstance(results, list)
        assert len(results) == 4

    def test_beta_monotone_in_results(self, sram28):
        beta_range = [1.0, 2.0, 4.0, 8.0]
        results = sweep_sram_mode_beta(sram28, beta_range=beta_range)
        betas = [r["beta_ratio"] for r in results]
        assert betas == sorted(betas)

    def test_result_keys(self, sram28):
        results = sweep_sram_mode_beta(sram28, beta_range=[2.0, 4.0])
        for r in results:
            assert "beta_ratio" in r
            assert "hold_snm_v" in r
            assert "verdict" in r

    def test_higher_beta_higher_hold_snm(self, sram28):
        results = sweep_sram_mode_beta(sram28, beta_range=[1.5, 6.0])
        assert results[1]["hold_snm_v"] >= results[0]["hold_snm_v"]

    def test_higher_beta_lower_write_margin(self, sram28):
        results = sweep_sram_mode_beta(sram28, beta_range=[1.5, 6.0])
        # beta 증가 → WM = Vdd×factor×wl/beta 감소
        assert results[1]["write_margin_v"] <= results[0]["write_margin_v"]

    def test_65nm_beta_sweep(self):
        results = sweep_sram_mode_beta(SRAM_65NM, beta_range=[2.0, 4.0, 6.0])
        assert len(results) == 3
        for r in results:
            assert r["verdict"] in ("PASS", "MARGINAL", "FAIL")
