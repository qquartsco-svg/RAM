"""Memory Engine — B 단계 시나리오 시뮬레이션 테스트.

§B1  고온 DRAM 붕괴 (scenario_high_temp_dram_collapse)
§B2  SRAM beta sweep 리포트 (scenario_sram_beta_sweep_report)
§B3  Vdd 강하 → 연쇄 실패 (scenario_vdd_drop_cascade)
"""

from __future__ import annotations

import math
import pytest

from memory_engine import (
    DRAMCellState, DRAMDesignParams, SRAMCellState, SRAMDesignParams,
    DDR4_PARAMS, DDR5_PARAMS, LPDDR5_PARAMS, DDR3_PARAMS,
    SRAM_7NM, SRAM_14NM, SRAM_28NM, SRAM_65NM,
    SA_PARAMS_7NM, SA_PARAMS_28NM, SA_PARAMS_65NM,
    retention_tau,
    scenario_high_temp_dram_collapse,
    scenario_sram_beta_sweep_report,
    scenario_vdd_drop_cascade,
)


# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_range():
    return [300.0, 325.0, 350.0, 375.0, 400.0]


@pytest.fixture
def beta_range_7nm():
    return [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


@pytest.fixture
def vdd_range_cascade():
    # 0.50V~0.08V — DRAM SA 실패 (0.14V 아래)를 포함하는 범위
    return [0.50, 0.30, 0.15, 0.12, 0.10, 0.08]


# ══════════════════════════════════════════════════════════════════════════════
# §B1  고온 DRAM 붕괴
# ══════════════════════════════════════════════════════════════════════════════

class TestHighTempDRAMCollapse:
    def test_returns_dict(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        assert isinstance(result, dict)

    def test_required_keys(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        for key in ("per_temperature", "reference_T_k", "reference_tau_s",
                    "collapse_T_k", "summary"):
            assert key in result, f"missing key: {key}"

    def test_per_temperature_length(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        assert len(result["per_temperature"]) == len(temp_range)

    def test_per_temperature_row_keys(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        required = {"T_k", "tau_s", "time_to_fail_s", "tau_speedup_x",
                    "omega_at_tau", "omega_at_2tau", "verdict_at_tau", "refresh_ok"}
        for row in result["per_temperature"]:
            missing = required - set(row.keys())
            assert not missing, f"missing row keys: {missing}"

    def test_tau_decreases_with_temperature(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        rows = result["per_temperature"]
        taus = [r["tau_s"] for r in rows]
        # Arrhenius: 고온 → τ 감소 (단조 비증가)
        assert all(taus[i] >= taus[i+1] for i in range(len(taus)-1))

    def test_speedup_increases_with_temperature(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        rows = result["per_temperature"]
        speedups = [r["tau_speedup_x"] for r in rows]
        # 첫 T = T_ref → speedup ≈ 1.0; 이후 증가
        assert speedups[0] <= speedups[-1]

    def test_reference_tau_matches_t_ref(self):
        ref_tau = retention_tau(DDR4_PARAMS)
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=[300.0])
        assert abs(result["reference_tau_s"] - ref_tau) < 1e-9

    def test_summary_is_nonempty_string(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_collapse_T_none_or_in_range(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        ct = result["collapse_T_k"]
        if ct is not None:
            assert ct in temp_range

    def test_omega_at_tau_in_range(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        for row in result["per_temperature"]:
            assert 0.0 <= row["omega_at_tau"] <= 1.0
            assert 0.0 <= row["omega_at_2tau"] <= 1.0

    def test_omega_at_2tau_leq_omega_at_tau(self, temp_range):
        # 2τ에서의 전하가 τ보다 낮으니 Ω도 같거나 낮아야 함
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        for row in result["per_temperature"]:
            assert row["omega_at_2tau"] <= row["omega_at_tau"] + 0.01  # 수치 오차 허용

    def test_high_temp_400k_shorter_ttf_than_300k(self):
        result = scenario_high_temp_dram_collapse(
            DDR4_PARAMS, T_range=[300.0, 400.0]
        )
        ttf_300 = result["per_temperature"][0]["time_to_fail_s"]
        ttf_400 = result["per_temperature"][1]["time_to_fail_s"]
        assert ttf_400 < ttf_300

    def test_refresh_ok_field_is_bool(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        for row in result["per_temperature"]:
            assert isinstance(row["refresh_ok"], bool)

    def test_ddr3_vs_lpddr5_collapse(self):
        # DDR3: 큰 Cs → 넉넉한 bitline swing → TTF 더 길 가능성
        # 두 프리셋 모두 결과 구조 반환 확인
        r3 = scenario_high_temp_dram_collapse(DDR3_PARAMS, T_range=[300.0, 350.0, 400.0])
        rl = scenario_high_temp_dram_collapse(LPDDR5_PARAMS, T_range=[300.0, 350.0, 400.0])
        assert len(r3["per_temperature"]) == 3
        assert len(rl["per_temperature"]) == 3

    def test_single_temperature_input(self):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=[350.0])
        assert len(result["per_temperature"]) == 1

    def test_verdict_at_tau_valid(self, temp_range):
        result = scenario_high_temp_dram_collapse(DDR4_PARAMS, T_range=temp_range)
        valid = {"HEALTHY", "STABLE", "FRAGILE", "CRITICAL"}
        for row in result["per_temperature"]:
            assert row["verdict_at_tau"] in valid


# ══════════════════════════════════════════════════════════════════════════════
# §B2  SRAM beta sweep 리포트
# ══════════════════════════════════════════════════════════════════════════════

class TestSRAMBetaSweepReport:
    def test_returns_dict(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        assert isinstance(result, dict)

    def test_required_keys(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        for key in ("per_beta", "beta_opt", "tradeoff_summary",
                    "pass_count", "fail_count", "node_nm"):
            assert key in result, f"missing key: {key}"

    def test_per_beta_length(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        assert len(result["per_beta"]) == len(beta_range_7nm)

    def test_per_beta_row_keys(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        required = {"beta_ratio", "hold_snm_v", "read_snm_v", "write_margin_v",
                    "node_disturb_risk", "verdict", "omega_global"}
        for row in result["per_beta"]:
            missing = required - set(row.keys())
            assert not missing, f"missing keys: {missing}"

    def test_hold_snm_increases_with_beta(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        snms = [r["hold_snm_v"] for r in result["per_beta"]]
        # beta 증가 → SNM 증가 (단조 비감소)
        assert all(snms[i] <= snms[i+1] + 1e-6 for i in range(len(snms)-1))

    def test_write_margin_decreases_with_beta(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        wms = [r["write_margin_v"] for r in result["per_beta"]]
        # beta 증가 → WM 감소 (단조 비증가)
        assert all(wms[i] >= wms[i+1] - 1e-6 for i in range(len(wms)-1))

    def test_tradeoff_summary_is_string(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        assert isinstance(result["tradeoff_summary"], str)
        assert len(result["tradeoff_summary"]) > 10

    def test_pass_fail_count_sum_leq_total(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        total = len(beta_range_7nm)
        assert result["pass_count"] + result["fail_count"] <= total

    def test_node_nm_matches_preset(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        assert result["node_nm"] == SRAM_7NM.node_nm

    def test_beta_opt_none_or_in_range(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        bo = result["beta_opt"]
        if bo is not None:
            assert any(abs(bo - b) < 1e-3 for b in beta_range_7nm)

    def test_verdict_values_valid(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        valid = {"PASS", "MARGINAL", "FAIL"}
        for row in result["per_beta"]:
            assert row["verdict"] in valid

    def test_omega_global_in_range(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        for row in result["per_beta"]:
            assert 0.0 <= row["omega_global"] <= 1.0

    def test_node_disturb_risk_valid(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        valid = {"LOW", "MEDIUM", "HIGH"}
        for row in result["per_beta"]:
            assert row["node_disturb_risk"] in valid

    def test_28nm_sweep(self):
        result = scenario_sram_beta_sweep_report(
            SRAM_28NM, beta_range=[2.0, 3.0, 5.0]
        )
        assert len(result["per_beta"]) == 3
        assert result["node_nm"] == SRAM_28NM.node_nm

    def test_65nm_sweep_high_beta_has_higher_snm(self):
        result = scenario_sram_beta_sweep_report(
            SRAM_65NM, beta_range=[1.5, 8.0]
        )
        snm_low  = result["per_beta"][0]["hold_snm_v"]
        snm_high = result["per_beta"][1]["hold_snm_v"]
        assert snm_high >= snm_low

    def test_single_beta_input(self):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=[3.0])
        assert len(result["per_beta"]) == 1

    def test_snm_wm_tradeoff_reflected_in_summary(self, beta_range_7nm):
        result = scenario_sram_beta_sweep_report(SRAM_7NM, beta_range=beta_range_7nm)
        # 요약에 SNM 또는 WM 관련 문자열 포함 확인
        summary = result["tradeoff_summary"]
        assert "SNM" in summary or "WM" in summary or "beta" in summary.lower()


# ══════════════════════════════════════════════════════════════════════════════
# §B3  Vdd 강하 → 연쇄 실패
# ══════════════════════════════════════════════════════════════════════════════

class TestVddDropCascade:
    def test_returns_dict(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        assert isinstance(result, dict)

    def test_required_keys(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        for key in ("per_vdd", "dram_fail_vdd", "sram_fail_vdd",
                    "cascade_start_vdd", "summary"):
            assert key in result, f"missing key: {key}"

    def test_per_vdd_length(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        assert len(result["per_vdd"]) == len(vdd_range_cascade)

    def test_per_vdd_row_keys(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        required = {"vdd_v", "dram_bitline_swing", "dram_read_margin",
                    "dram_sa_margin_v", "dram_read_ok",
                    "sram_snm_v", "sram_rnm_v", "sram_wm_v",
                    "sram_verdict", "combined_verdict"}
        for row in result["per_vdd"]:
            missing = required - set(row.keys())
            assert not missing, f"missing row keys: {missing}"

    def test_per_vdd_sorted_ascending(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        vdds = [r["vdd_v"] for r in result["per_vdd"]]
        assert vdds == sorted(vdds)

    def test_dram_bitline_swing_decreases_with_vdd(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        swings = [r["dram_bitline_swing"] for r in result["per_vdd"]]
        # 낮은 Vdd → 낮은 swing (단조 비증가)
        assert all(swings[i] <= swings[i+1] + 1e-9 for i in range(len(swings)-1))

    def test_dram_fails_at_very_low_vdd(self):
        # Vdd=0.08V: delta_v = 0.08 * 20/140 ≈ 0.0114V < SA_28NM offset 0.020V
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=[0.5, 0.12, 0.08],
        )
        assert result["dram_fail_vdd"] is not None
        # 실패 Vdd는 범위 내에 있어야 함
        assert result["dram_fail_vdd"] in [r["vdd_v"] for r in result["per_vdd"]]

    def test_dram_ok_at_normal_vdd(self):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=[1.2],
        )
        row = result["per_vdd"][0]
        assert row["dram_read_ok"] is True

    def test_combined_verdict_values_valid(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        valid = {"OK", "DRAM_FAIL", "SRAM_FAIL", "BOTH_FAIL"}
        for row in result["per_vdd"]:
            assert row["combined_verdict"] in valid

    def test_high_vdd_combined_ok(self):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=[1.2, 1.0],
        )
        for row in result["per_vdd"]:
            # 정상 전압에서는 DRAM 읽기 성공
            assert row["dram_read_ok"] is True

    def test_summary_is_nonempty_string(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 5

    def test_cascade_start_is_max_of_fails(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        d_fail = result["dram_fail_vdd"]
        s_fail = result["sram_fail_vdd"]
        cascade = result["cascade_start_vdd"]
        fails = [v for v in [d_fail, s_fail] if v is not None]
        if fails:
            assert cascade == max(fails)
        else:
            assert cascade is None

    def test_sram_snm_increases_with_vdd(self):
        # per_vdd는 Vdd 오름차순 → snms[0]=낮은Vdd(작은SNM), snms[-1]=높은Vdd(큰SNM)
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=[0.3, 0.5, 1.0],
        )
        snms = [r["sram_snm_v"] for r in result["per_vdd"]]
        # Vdd 증가 → SRAM SNM 증가 (또는 유지)
        assert snms[-1] >= snms[0] - 1e-6

    def test_7nm_sa_vs_65nm_sa_dram_tolerance(self):
        # SA_7NM(오프셋 낮음) → DRAM 실패 임계 더 낮음
        result_7nm = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_7NM, SA_PARAMS_7NM,
            vdd_range=[0.20, 0.10, 0.06],
        )
        result_65nm = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_65NM, SA_PARAMS_65NM,
            vdd_range=[0.20, 0.10, 0.06],
        )
        # 둘 다 구조 유효
        assert len(result_7nm["per_vdd"]) == 3
        assert len(result_65nm["per_vdd"]) == 3

    def test_ddr5_lpddr5_scenarios_valid(self):
        for dram_p in (DDR5_PARAMS, LPDDR5_PARAMS):
            result = scenario_vdd_drop_cascade(
                dram_p, SRAM_14NM, SA_PARAMS_7NM,
                vdd_range=[1.1, 0.5, 0.1],
            )
            assert len(result["per_vdd"]) == 3
            for row in result["per_vdd"]:
                assert row["combined_verdict"] in {"OK", "DRAM_FAIL", "SRAM_FAIL", "BOTH_FAIL"}

    def test_single_vdd_point(self):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=[0.8],
        )
        assert len(result["per_vdd"]) == 1

    def test_sram_omega_in_range(self, vdd_range_cascade):
        result = scenario_vdd_drop_cascade(
            DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
            vdd_range=vdd_range_cascade,
        )
        for row in result["per_vdd"]:
            assert 0.0 <= row["sram_omega"] <= 1.0
