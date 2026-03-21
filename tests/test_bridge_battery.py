"""Memory Engine — D 단계 Battery-Memory 브리지 테스트.

§D1  BatteryECMParams, simulate_battery_discharge
§D2  vterm_to_vdd 변환
§D3  battery_memory_cascade
§D4  sweep_soc_memory_health
"""

from __future__ import annotations

import pytest

from memory_engine import (
    DRAMCellState, DRAMDesignParams,
    DDR4_PARAMS, DDR5_PARAMS, LPDDR5_PARAMS,
    SRAM_7NM, SRAM_28NM, SRAM_65NM,
    SA_PARAMS_7NM, SA_PARAMS_28NM, SA_PARAMS_65NM,
    BatteryECMParams, BatteryStep,
    simulate_battery_discharge,
    vterm_to_vdd,
    battery_memory_cascade,
    sweep_soc_memory_health,
)


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def batt_params():
    return BatteryECMParams(q_ah=3.0, soc_v0=2.8, soc_ocv_v_per_unit=1.0,
                            r0_ohm=0.08, r1_ohm=0.04, c1_farad=2000.0)


@pytest.fixture
def batt_steps(batt_params):
    return simulate_battery_discharge(
        current_a=1.5, dt_s=60.0, n_steps=15,
        params=batt_params, soc_init=1.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §D1  simulate_battery_discharge
# ══════════════════════════════════════════════════════════════════════════════

class TestSimulateBatteryDischarge:
    def test_returns_list(self, batt_params):
        steps = simulate_battery_discharge(1.0, 60.0, 10, batt_params)
        assert isinstance(steps, list)
        assert len(steps) > 0

    def test_all_elements_are_batterystep(self, batt_params):
        steps = simulate_battery_discharge(1.0, 60.0, 5, batt_params)
        for s in steps:
            assert isinstance(s, BatteryStep)

    def test_soc_starts_at_init(self, batt_params):
        steps = simulate_battery_discharge(1.0, 60.0, 10, batt_params, soc_init=0.8)
        assert abs(steps[0].soc - 0.8) < 1e-6

    def test_soc_decreases_monotonically(self, batt_params):
        steps = simulate_battery_discharge(2.0, 30.0, 20, batt_params)
        socs = [s.soc for s in steps]
        assert all(socs[i] >= socs[i+1] - 1e-6 for i in range(len(socs)-1))

    def test_soc_never_negative(self, batt_params):
        steps = simulate_battery_discharge(10.0, 300.0, 50, batt_params)
        assert all(s.soc >= 0.0 for s in steps)

    def test_vterm_never_negative_above_cutoff(self, batt_params):
        steps = simulate_battery_discharge(1.0, 10.0, 10, batt_params)
        for s in steps:
            if not s.discharged:
                assert s.v_term >= 0.0

    def test_vterm_decreases_with_discharge(self, batt_params):
        steps = simulate_battery_discharge(2.0, 60.0, 15, batt_params)
        # 첫 스텝 V_term ≥ 마지막 스텝 V_term
        assert steps[0].v_term >= steps[-1].v_term - 1e-4

    def test_time_increases_monotonically(self, batt_params):
        steps = simulate_battery_discharge(1.0, 60.0, 10, batt_params)
        ts = [s.t_s for s in steps]
        assert all(ts[i] <= ts[i+1] for i in range(len(ts)-1))

    def test_discharged_flag_set_when_vterm_low(self, batt_params):
        # 높은 전류 → 빠른 방전 → discharged 플래그 발생
        params = BatteryECMParams(q_ah=0.1, soc_v0=2.8, t_cutoff_v=3.5)
        steps = simulate_battery_discharge(5.0, 1.0, 100, params, soc_init=0.5)
        assert any(s.discharged for s in steps)

    def test_zero_current_soc_unchanged(self, batt_params):
        steps = simulate_battery_discharge(0.0, 60.0, 5, batt_params, soc_init=0.7)
        # 전류 0 → SOC 변화 없음
        assert all(abs(s.soc - 0.7) < 1e-6 for s in steps)

    def test_step_count_respects_n_steps(self, batt_params):
        steps = simulate_battery_discharge(0.5, 60.0, 10, batt_params)
        # 방전 종료 없을 때 n_steps+1 개
        assert len(steps) <= 11

    def test_high_soh_longer_discharge(self, batt_params):
        steps_good = simulate_battery_discharge(1.0, 60.0, 30,
            BatteryECMParams(q_ah=3.0, soh=1.0), soc_init=1.0)
        steps_aged = simulate_battery_discharge(1.0, 60.0, 30,
            BatteryECMParams(q_ah=3.0, soh=0.5), soc_init=1.0)
        # SOH 높을수록 마지막 SOC가 높음 (더 느린 방전)
        assert steps_good[-1].soc >= steps_aged[-1].soc - 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# §D2  vterm_to_vdd
# ══════════════════════════════════════════════════════════════════════════════

class TestVtermToVdd:
    def test_basic_conversion(self):
        vdd = vterm_to_vdd(3.7, n_cells=1, pmic_efficiency=1.0)
        assert abs(vdd - 3.7) < 1e-9

    def test_efficiency_reduces_vdd(self):
        vdd_full = vterm_to_vdd(3.7, n_cells=1, pmic_efficiency=1.0)
        vdd_85   = vterm_to_vdd(3.7, n_cells=1, pmic_efficiency=0.85)
        assert vdd_85 < vdd_full

    def test_n_cells_scales_linearly(self):
        vdd_1 = vterm_to_vdd(3.7, n_cells=1, pmic_efficiency=1.0)
        vdd_2 = vterm_to_vdd(3.7, n_cells=2, pmic_efficiency=1.0)
        assert abs(vdd_2 - 2 * vdd_1) < 1e-9

    def test_zero_vterm_gives_zero_vdd(self):
        vdd = vterm_to_vdd(0.0, n_cells=1, pmic_efficiency=0.85)
        assert vdd == 0.0

    def test_result_nonnegative(self):
        for v in [0.0, 1.0, 2.5, 4.2]:
            assert vterm_to_vdd(v) >= 0.0

    def test_higher_vterm_higher_vdd(self):
        assert vterm_to_vdd(4.2) > vterm_to_vdd(3.0)

    def test_default_efficiency_applied(self):
        # 기본 pmic_efficiency=0.85
        vdd = vterm_to_vdd(3.7)
        assert vdd < 3.7  # 효율 손실로 감소


# ══════════════════════════════════════════════════════════════════════════════
# §D3  battery_memory_cascade
# ══════════════════════════════════════════════════════════════════════════════

class TestBatteryMemoryCascade:
    def test_returns_dict(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        assert isinstance(result, dict)

    def test_required_keys(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        for key in ("trajectory", "first_dram_fail_t_s", "first_sram_fail_t_s",
                    "first_cascade_t_s", "min_vdd", "summary"):
            assert key in result, f"missing key: {key}"

    def test_trajectory_length_matches_steps(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        assert len(result["trajectory"]) == len(batt_steps)

    def test_trajectory_row_keys(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        required = {"t_s", "soc", "v_term", "vdd",
                    "dram_read_ok", "sram_verdict", "combined_verdict"}
        for row in result["trajectory"]:
            missing = required - set(row.keys())
            assert not missing, f"missing keys: {missing}"

    def test_vdd_decreases_with_time(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        vdds = [r["vdd"] for r in result["trajectory"]]
        # 방전 → Vdd 단조 감소 (또는 유지)
        assert all(vdds[i] >= vdds[i+1] - 1e-4 for i in range(len(vdds)-1))

    def test_min_vdd_is_minimum(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        computed_min = min(r["vdd"] for r in result["trajectory"])
        assert abs(result["min_vdd"] - computed_min) < 1e-3

    def test_summary_is_nonempty_string(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 5

    def test_empty_steps_returns_empty_trajectory(self):
        result = battery_memory_cascade([], DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM)
        assert result["trajectory"] == []
        assert result["first_dram_fail_t_s"] is None

    def test_combined_verdict_valid_values(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        valid = {"OK", "DRAM_FAIL", "SRAM_FAIL", "BOTH_FAIL"}
        for row in result["trajectory"]:
            assert row["combined_verdict"] in valid

    def test_dram_fail_triggers_cascade(self, batt_params):
        # 낮은 PMIC 효율(5%) → Vdd < SA 임계 → DRAM 실패
        steps = simulate_battery_discharge(
            1.0, 60.0, 5, batt_params, soc_init=0.5,
        )
        result = battery_memory_cascade(
            steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            n_cells=1, pmic_efficiency=0.03,  # Vdd ≈ 0.109V → DRAM 실패
        )
        # 첫 스텝에서 dram_read_ok=False여야 함
        assert result["trajectory"][0]["dram_read_ok"] is False

    def test_cascade_start_consistent_with_fails(self, batt_steps):
        result = battery_memory_cascade(
            batt_steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        d = result["first_dram_fail_t_s"]
        s = result["first_sram_fail_t_s"]
        cascade = result["first_cascade_t_s"]
        fails = [v for v in [d, s] if v is not None]
        if fails:
            assert cascade == min(fails)
        else:
            assert cascade is None

    def test_different_sa_params(self, batt_steps):
        for sa in (SA_PARAMS_7NM, SA_PARAMS_28NM, SA_PARAMS_65NM):
            result = battery_memory_cascade(
                batt_steps, DDR4_PARAMS, SRAM_28NM, sa,
            )
            assert isinstance(result["trajectory"], list)

    def test_lpddr5_ddr5_presets(self, batt_steps):
        for dp in (DDR5_PARAMS, LPDDR5_PARAMS):
            result = battery_memory_cascade(
                batt_steps, dp, SRAM_7NM, SA_PARAMS_7NM,
            )
            assert len(result["trajectory"]) == len(batt_steps)


# ══════════════════════════════════════════════════════════════════════════════
# §D4  sweep_soc_memory_health
# ══════════════════════════════════════════════════════════════════════════════

class TestSweepSOCMemoryHealth:
    def test_returns_list(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[1.0, 0.5, 0.1],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_result_row_keys(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[0.8],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        required = {"soc", "v_term", "vdd", "dram_read_ok",
                    "sram_verdict", "dram_sa_margin_v", "sram_snm_v",
                    "combined_verdict"}
        for row in results:
            missing = required - set(row.keys())
            assert not missing

    def test_vterm_decreases_with_soc(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[1.0, 0.5, 0.1],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        vterms = [r["v_term"] for r in results]
        assert vterms[0] >= vterms[-1]

    def test_vdd_decreases_with_soc(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[1.0, 0.5, 0.1],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        vdds = [r["vdd"] for r in results]
        assert vdds[0] >= vdds[-1]

    def test_high_soc_dram_ok(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[1.0],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        # SOC=1.0 → V_term 높음 → Vdd 높음 → DRAM 안전
        assert results[0]["dram_read_ok"] is True

    def test_combined_verdict_valid(self, batt_params):
        results = sweep_soc_memory_health(
            soc_range=[1.0, 0.5, 0.1],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
        )
        valid = {"OK", "DRAM_FAIL", "SRAM_FAIL", "BOTH_FAIL"}
        for r in results:
            assert r["combined_verdict"] in valid

    def test_low_pmic_efficiency_causes_dram_fail(self, batt_params):
        # pmic_efficiency=0.03 → Vdd ≈ 0.109V → delta_v < SA_28NM offset(20mV) → 실패
        results = sweep_soc_memory_health(
            soc_range=[1.0],
            batt_params=batt_params,
            dram_params=DDR4_PARAMS,
            sram_params=SRAM_28NM,
            sa_params=SA_PARAMS_28NM,
            pmic_efficiency=0.03,
        )
        assert results[0]["dram_read_ok"] is False

    def test_n_cells_affects_vdd(self, batt_params):
        r1 = sweep_soc_memory_health(
            [0.8], batt_params, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM, n_cells=1
        )
        r2 = sweep_soc_memory_health(
            [0.8], batt_params, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM, n_cells=2
        )
        assert r2[0]["vdd"] > r1[0]["vdd"]

    def test_all_soc_range_covered(self, batt_params):
        soc_range = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
        results = sweep_soc_memory_health(
            soc_range, batt_params, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        assert len(results) == len(soc_range)

    def test_soc_values_preserved(self, batt_params):
        soc_range = [0.9, 0.5, 0.1]
        results = sweep_soc_memory_health(
            soc_range, batt_params, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
        )
        for r, soc in zip(results, soc_range):
            assert abs(r["soc"] - soc) < 1e-4

    def test_multiple_presets(self, batt_params):
        for dp, sp, sa in [
            (DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM),
            (DDR5_PARAMS, SRAM_7NM, SA_PARAMS_7NM),
            (LPDDR5_PARAMS, SRAM_65NM, SA_PARAMS_65NM),
        ]:
            results = sweep_soc_memory_health(
                [1.0, 0.5], batt_params, dp, sp, sa,
            )
            assert len(results) == 2
