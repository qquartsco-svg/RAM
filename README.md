# Memory Engine — DRAM / SRAM 반도체 메모리 설계·시뮬레이션·검증 엔진

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-267%20passed-brightgreen)](#테스트)
[![Version](https://img.shields.io/badge/version-0.4.0-orange)](#버전-히스토리)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

순수 Python으로 구현된 **DRAM / SRAM 반도체 메모리 물리 시뮬레이션 엔진**.
RC 방전·Arrhenius 가속·SNM butterfly curve·Row Hammer·센스앰프 BER 등
실제 팹리스 설계에서 사용하는 핵심 물리 모델을 stdlib만으로 구현합니다.

> **⚠️ 범위 안내** — 본 엔진은 Python 기반의 **설계 탐색·마진 추정·시나리오 분석** 도구입니다.
> SPICE 회로 시뮬레이션·TCAD 공정 시뮬레이션·레이아웃 signoff를 대체하지 않습니다.
> 모든 수치는 단순화된 해석 모델 기반 추정치이며, 실리콘 측정값과 차이가 있을 수 있습니다.

---

## 목차

1. [특징](#특징)
2. [아키텍처](#아키텍처)
3. [빠른 시작](#빠른-시작)
4. [DRAM 물리 모델](#dram-물리-모델)
5. [SRAM 물리 모델](#sram-물리-모델)
6. [센스앰프 (A1)](#센스앰프-a1)
7. [Row Hammer (A2)](#row-hammer-a2)
8. [SRAM 3모드 SNM (A3)](#sram-3모드-snm-a3)
9. [Observer Ω 5레이어](#observer-ω-5레이어)
10. [시나리오 시뮬레이션 (B)](#시나리오-시뮬레이션-b)
11. [Battery-Memory 브리지 (D)](#battery-memory-브리지-d)
12. [공정 프리셋](#공정-프리셋)
13. [API 레퍼런스](#api-레퍼런스)
14. [테스트](#테스트)
15. [버전 히스토리](#버전-히스토리)

---

## 특징

| 기능 | 설명 |
|------|------|
| **DRAM 물리** | RC 지수 방전, Arrhenius 온도 가속, 읽기 교란, Refresh, Row Hammer |
| **SRAM 물리** | VTC 3구간 모델, Seevinck SNM, Hold/Read/Write 3모드 마진 |
| **센스앰프** | 차동 래치, BER = Q(ΔV/σ), Q⁻¹ 역함수(이분법), 행 BER |
| **Observer Ω** | 5레이어 건강 지수 (0~1), HEALTHY/STABLE/FRAGILE/CRITICAL 판정 |
| **시나리오** | 고온 붕괴, beta sweep 리포트, Vdd 강하 연쇄 실패 |
| **배터리 브리지** | ECM 내장, V_term → Vdd → 메모리 마진 연동 |
| **의존성 없음** | 순수 stdlib (math, dataclasses) — NumPy/SciPy 불필요 |

---

## 아키텍처

```
memory_engine/
├── schema.py           — 데이터 모델 (DRAMCellState, SRAMCellState, SAParams, ...)
├── dram_physics.py     — DRAM 물리 (RC 방전, Arrhenius, Read Disturb, Refresh, Row Hammer)
├── sram_physics.py     — SRAM 물리 (VTC, SNM, RNM, WM, 3모드 분석)
├── sense_amplifier.py  — 센스앰프 (BER, Q-function, Q-역함수, SA 프리셋)
├── observer.py         — Ω 5레이어 관측 + 진단
├── design_engine.py    — 시뮬레이션, 파라미터 스윕, 검증 보고서, 시나리오
├── bridge_battery.py   — Battery ECM 내장 + V_term → Vdd → 메모리 연동
└── presets.py          — 공정 프리셋 (LPDDR5/DDR5/DDR4/DDR3, SRAM 7~65nm)
```

---

## 빠른 시작

```python
from memory_engine import (
    DRAMCellState, DDR4_PARAMS,
    retention_decay, observe_dram, verify_dram,
    sweep_dram_temperature,
)

# DRAM 전하 방전
cell = DRAMCellState(q=1.0)
cell = retention_decay(cell, DDR4_PARAMS, dt_s=32e-3)   # 32ms 경과

# Observer Ω 관측
obs = observe_dram(cell, DDR4_PARAMS)
print(obs.verdict, obs.omega_global)   # STABLE  0.74

# 검증 보고서
report = verify_dram(cell, DDR4_PARAMS)
print(report.verdict, report.notes)    # PASS  []

# 온도 스윕 (300K~400K)
sweep = sweep_dram_temperature(DDR4_PARAMS, T_range=[300, 325, 350, 375, 400])
for row in sweep:
    print(f"T={row['T_k']}K  τ={row['tau_ms']:.2f}ms  Ω={row['omega_global']:.3f}")
```

```python
from memory_engine import (
    SRAMCellState, SRAM_28NM,
    static_noise_margin, snm_by_mode, observe_sram,
)

# SRAM SNM 계산
snm = static_noise_margin(SRAM_28NM)
print(f"SNM = {snm*1000:.1f} mV")     # SNM = 207.1 mV

# 3모드 통합 분석
modes = snm_by_mode(SRAM_28NM)
print(modes["hold_snm_v"], modes["write_margin_v"], modes["verdict"])
```

---

## DRAM 물리 모델

### 1. 전하 보존 시정수 — Arrhenius 가속

```
τ(T) = τ_ref × exp( Ea/kB × (1/T − 1/T_ref) )
```

- **Ea**: 활성화 에너지 [eV] (기본 0.65 eV)
- **kB**: Boltzmann 상수 = 8.617×10⁻⁵ eV/K
- 고온 → τ 감소 → 빠른 전하 소실

```python
from memory_engine import retention_tau, DDR4_PARAMS
import dataclasses

tau_300 = retention_tau(DDR4_PARAMS)                                   # 64.0 ms
tau_350 = retention_tau(dataclasses.replace(DDR4_PARAMS, T_k=350.0))  # ~3.1 ms
print(f"300K: {tau_300*1000:.1f}ms → 350K: {tau_350*1000:.2f}ms  ({tau_300/tau_350:.0f}× 가속)")
```

### 2. RC 지수 방전

```
Q(t) = Q₀ × exp(−t / τ)
```

```python
from memory_engine import retention_decay, DRAMCellState, DDR4_PARAMS

cell = DRAMCellState(q=1.0)
after = retention_decay(cell, DDR4_PARAMS, dt_s=64e-3)   # 1τ 경과
print(f"q: 1.000 → {after.q:.4f}")    # q: 1.000 → 0.3679
```

### 3. 비트라인 스윙 & 읽기 마진

```
ΔVbl / Vdd = q × Cs / (Cs + Cbl)
Read Margin = ΔVbl/Vdd − sense_threshold_fraction
```

### 4. Refresh

```
q_new = min(1, q + refresh_recovery × (1 − q))
```

### 5. 읽기 교란

```
q_new = q − read_disturb_factor × q
```

---

## SRAM 물리 모델

### CMOS 인버터 VTC (3구간 piece-wise)

| 구간 | 조건 | Vout |
|------|------|------|
| Off (NMOS) | Vin ≤ Vth_n | Vdd |
| 전이 구간 | Vth_n < Vin < Vdd−Vth_p | Seevinck 스위칭 포인트 |
| Off (PMOS) | Vin ≥ Vdd−Vth_p | 0 |

스위칭 포인트:
```
Vm = (Vdd + Vth_n·√β − Vth_p) / (1 + √β)
```

### Static Noise Margin (SNM)

두 단계 계산으로 SNM을 추정합니다:

**① 해석적 추정 (Seevinck 공식)** — 스위칭 포인트 Vm 기반 닫힌 형태 근사:
```
SNM_analytic ≈ 0.5 × (Vdd − Vth_n − Vth_p) × (β−1)/(β+1)
```
빠른 트렌드 분석에 적합하지만, 실제 butterfly 최대 내접 정사각형과 차이가 있을 수 있습니다.

**② 수치 VTC 그리드 평균** — 201포인트 piece-wise VTC를 두 인버터에 대해 샘플링하고
교차 영역의 평균 마진을 계산하여 해석 공식을 보정합니다:
```
SNM = 0.5 × (SNM_analytic + SNM_numeric_avg)
```

> 두 방법 모두 **단순화 모델 기반 추정치**입니다. SPICE-level Monte Carlo 시뮬레이션이나
> 실리콘 측정 butterfly curve와 정확히 일치하지 않을 수 있습니다.

```python
from memory_engine import static_noise_margin, SRAM_7NM, SRAM_65NM

print(f"7nm  SNM = {static_noise_margin(SRAM_7NM)*1000:.1f} mV")
print(f"65nm SNM = {static_noise_margin(SRAM_65NM)*1000:.1f} mV")
```

---

## 센스앰프 (A1)

차동 래치 모델 — 비트라인 전압차 ΔV를 감지:

```
ΔV = q × Vdd × Cs / (Cs + Cbl)
SA margin = ΔV − sa_offset_v
BER = Q(ΔV / σ) = 0.5 × erfc(ΔV / (σ√2))
```

```python
from memory_engine import (
    sense_op, sa_bit_error_rate, sa_row_ber,
    sa_min_delta_v_for_ber,
    SA_PARAMS_7NM, SA_PARAMS_28NM, SA_PARAMS_65NM,
    DRAMCellState, DDR4_PARAMS,
)

cell = DRAMCellState(q=1.0)
obs = sense_op(cell, DDR4_PARAMS, SA_PARAMS_28NM)
print(f"ΔV={obs.delta_v:.4f}V  margin={obs.sa_margin_v:.4f}V  ok={obs.read_success}")
print(f"BER={sa_bit_error_rate(cell, DDR4_PARAMS, SA_PARAMS_28NM):.2e}")

# BER=1e-9 달성에 필요한 최소 ΔV
min_dv = sa_min_delta_v_for_ber(1e-9, SA_PARAMS_28NM)
print(f"BER≤1e-9 달성 최소 ΔV: {min_dv*1000:.2f} mV")
```

| 프리셋 | 공정 | SA 오프셋 | σ |
|--------|------|-----------|---|
| `SA_PARAMS_7NM` | 7nm | 8 mV | ~2.7 mV |
| `SA_PARAMS_28NM` | 28nm | 20 mV | ~6.7 mV |
| `SA_PARAMS_65NM` | 65nm | 35 mV | ~11.7 mV |

---

## Row Hammer (A2)

인접 행 반복 활성화 → 피해 셀 전하 지수 감쇠:

```
Q_victim(n) = Q₀ × (1 − loss_per_event)ⁿ
```

```python
from memory_engine import (
    row_hammer_disturb, row_hammer_failure_threshold,
    simulate_row_hammer, find_row_hammer_threshold,
    DRAMCellState, DDR4_PARAMS, SA_PARAMS_28NM,
)

cell = DRAMCellState(q=1.0)

# 100K 번 해머 후 전하 잔량
after = row_hammer_disturb(cell, DDR4_PARAMS, n_hammers=100_000, rh_charge_loss_per_event=5e-6)
print(f"q: 1.0000 → {after.q:.4f}")

# 읽기 마진 실패까지 필요한 hammer 수
n_fail = row_hammer_failure_threshold(DDR4_PARAMS, rh_charge_loss_per_event=5e-6)
print(f"실패 임계: {n_fail:,} hammers")

# SA 기준 + 전하 기준 이중 임계 분석
thresholds = find_row_hammer_threshold(DDR4_PARAMS, SA_PARAMS_28NM)
print(f"SA 실패: {thresholds['n_hammer_sa_fail']:,}  |  q 기준: {thresholds['n_hammer_q_fail']:,}")
print(f"안전 한계: {thresholds['safer_limit']:,} hammers")

# 전체 궤적 시뮬레이션
results = simulate_row_hammer(cell, DDR4_PARAMS, SA_PARAMS_28NM,
    total_hammers=50_000, step_hammers=5_000)
for r in results[::2]:
    print(f"  n={r['n_hammers']:6d}  q={r['q']:.4f}  ok={r['read_success']}")
```

---

## SRAM 3모드 SNM (A3)

6T SRAM은 동작 모드에 따라 SNM이 다릅니다:

| 모드 | 조건 | SNM |
|------|------|-----|
| **Hold** | BL=BLB=float (대기) | SNM_hold (최대) |
| **Read** | BL=BLB=Vdd, WL enable | SNM_hold − ΔV_read |
| **Write** | BL/BLB 차동, WL enable | Write Margin (WM) |

```
ΔV_read = Vdd / (β + 1)      # 읽기 시 '0' 노드 전압 상승
WM = Vdd × wm_factor × wl_strength / β
```

```python
from memory_engine import snm_by_mode, sram_mode_analysis, sweep_sram_mode_beta, SRAM_7NM

# 3모드 분석
modes = snm_by_mode(SRAM_7NM)
print(f"Hold SNM:  {modes['hold_snm_v']*1000:.1f} mV")
print(f"Read SNM:  {modes['read_snm_physical_v']*1000:.1f} mV")
print(f"Write Margin: {modes['write_margin_v']*1000:.1f} mV")
print(f"Verdict: {modes['verdict']}")

# beta 스윕 — Hold SNM ↑ / Write Margin ↓ 트레이드오프
sweep = sweep_sram_mode_beta(SRAM_7NM, beta_range=[1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
for r in sweep:
    print(f"β={r['beta_ratio']:.1f}  SNM={r['hold_snm_v']*1000:.1f}mV  "
          f"WM={r['write_margin_v']*1000:.1f}mV  risk={r['node_disturb_risk']}")
```

**beta 트레이드오프:**
```
β ↑  →  Hold SNM ↑  (읽기 안정성 향상)
β ↑  →  Write Margin ↓  (쓰기 어려워짐)
β ↑  →  ΔV_read = Vdd/(β+1) ↓  (읽기 교란 감소)
```

---

## Observer Ω 5레이어

모든 물리 지표를 0~1 범위의 건강 지수로 통합:

### DRAM Observer

| 레이어 | 가중치 | 기반 지표 |
|--------|--------|-----------|
| Ω_retention | 0.30 | 전하 잔량 q, τ 대비 시간 |
| Ω_margin | 0.25 | 비트라인 스윙, 읽기 마진 |
| Ω_endurance | 0.20 | 사이클 소모율 |
| Ω_speed | 0.15 | 접근 시간 t_access |
| Ω_power | 0.10 | Vdd, 누설 전류 |

### SRAM Observer

| 레이어 | 가중치 | 기반 지표 |
|--------|--------|-----------|
| Ω_snm | 0.30 | SNM / (Vdd/2) — 정규화 노이즈 마진 |
| Ω_margin | 0.25 | RNM (읽기 노이즈 마진) / (Vdd×0.3) |
| Ω_write | 0.20 | Write Margin / (Vdd×0.25) |
| Ω_leakage | 0.15 | 누설 열화율 (leakage_factor) |
| Ω_power | 0.10 | Vdd / Vdd_nom |

```
Ω_global = Σ (weight_i × Ω_i)
```

| Ω_global | 판정 |
|----------|------|
| ≥ 0.80 | HEALTHY |
| ≥ 0.60 | STABLE |
| ≥ 0.40 | FRAGILE |
| < 0.40 | CRITICAL |

```python
from memory_engine import observe_dram, observe_sram, diagnose
from memory_engine import DRAMCellState, DDR4_PARAMS, SRAM_28NM, SRAMCellState

# DRAM 관측
obs = observe_dram(DRAMCellState(q=0.8), DDR4_PARAMS)
print(f"Ω={obs.omega_global:.3f}  verdict={obs.verdict}")
print(f"flags={obs.flags}")

# 진단 권고
advice = diagnose(obs)
for line in advice:
    print(f"  ▸ {line}")
```

---

## 시나리오 시뮬레이션 (B)

### B1. 고온 DRAM 붕괴 분석

```python
from memory_engine import scenario_high_temp_dram_collapse, DDR4_PARAMS

result = scenario_high_temp_dram_collapse(
    DDR4_PARAMS,
    T_range=[300, 325, 350, 375, 400, 425, 450],
)
print(result["summary"])

for row in result["per_temperature"]:
    status = "✓" if row["refresh_ok"] else "✗ COLLAPSE"
    print(f"  T={row['T_k']}K  τ={row['tau_s']*1000:.2f}ms  "
          f"{row['tau_speedup_x']:.0f}×가속  {status}")
```

출력 예시:
```
붕괴 임계 온도: 300.0K — retention이 표준 refresh 주기(64ms) 이하로 단축.
  T=300K  τ=64.00ms  1×가속  ✗ COLLAPSE
  T=350K  τ=3.06ms   21×가속  ✗ COLLAPSE
  T=400K  τ=0.31ms   208×가속  ✗ COLLAPSE
```

> **📌 모델 의존 수치** — τ 및 가속 배율은 프리셋의 `Ea`(활성화 에너지), `tau_ref_s`, `T_ref_k`
> 파라미터에 강하게 의존합니다. JEDEC JESD79 규격 수치와 직접 대응하지 않으며,
> 실제 다이 특성은 셀 구조·도핑 프로파일·측정 조건에 따라 크게 달라집니다.

### B2. SRAM beta sweep 리포트

```python
from memory_engine import scenario_sram_beta_sweep_report, SRAM_7NM

report = scenario_sram_beta_sweep_report(
    SRAM_7NM,
    beta_range=[1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
)
print(f"최적 beta: {report['beta_opt']}")
print(f"PASS: {report['pass_count']}  FAIL: {report['fail_count']}")
print(report["tradeoff_summary"])
```

### B3. Vdd 강하 → 연쇄 실패

```python
from memory_engine import (
    scenario_vdd_drop_cascade,
    DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
)

result = scenario_vdd_drop_cascade(
    DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
    vdd_range=[1.2, 0.8, 0.5, 0.3, 0.15, 0.12, 0.10, 0.08],
)
print(result["summary"])

for row in result["per_vdd"]:
    print(f"  Vdd={row['vdd_v']:.2f}V  "
          f"DRAM={'OK' if row['dram_read_ok'] else 'FAIL'}  "
          f"SRAM={row['sram_verdict']}  "
          f"→ {row['combined_verdict']}")
```

---

## Battery-Memory 브리지 (D)

배터리 방전 시 단자 전압 강하 → PMIC 변환 → 메모리 Vdd 감소 → 마진 열화를 연동 분석.

> **📌 간략 ECM 기반** — 내장 배터리 모델은 단일 RC 등가회로(ECM: Equivalent Circuit Model)
> 기반의 간략 모델입니다. 실제 전기화학 셀의 농도 분극·SEI 성장·온도 의존성 등
> 상세 거동은 반영되지 않으며, 상세 전기화학 셀 모델은 후속 확장 범위입니다.

```
V_term(SOC) = V₀ + k_ocv × SOC − I × R₀ − V_RC
Vdd = V_term × n_cells × η_pmic
```

```python
from memory_engine import (
    BatteryECMParams, simulate_battery_discharge,
    vterm_to_vdd, battery_memory_cascade, sweep_soc_memory_health,
    DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
)

# 1. 배터리 방전 궤적 생성
batt = BatteryECMParams(q_ah=3.0, soc_v0=2.8, soc_ocv_v_per_unit=1.0)
steps = simulate_battery_discharge(current_a=2.0, dt_s=60.0, n_steps=30, params=batt)

# 2. 방전 궤적 → 메모리 연쇄 실패 분석
result = battery_memory_cascade(
    steps, DDR4_PARAMS, SRAM_28NM, SA_PARAMS_28NM,
    n_cells=1, pmic_efficiency=0.85,
)
print(result["summary"])
print(f"Vdd 최소: {result['min_vdd']:.4f}V")

# 3. SOC 스냅샷 스윕
health = sweep_soc_memory_health(
    soc_range=[1.0, 0.8, 0.6, 0.4, 0.2, 0.05],
    batt_params=batt,
    dram_params=DDR4_PARAMS,
    sram_params=SRAM_28NM,
    sa_params=SA_PARAMS_28NM,
    current_a=2.0, n_cells=1, pmic_efficiency=0.85,
)
for r in health:
    print(f"  SOC={r['soc']:.2f}  Vdd={r['vdd']:.3f}V  "
          f"DRAM={'OK' if r['dram_read_ok'] else 'FAIL'}  "
          f"→ {r['combined_verdict']}")
```

---

## 공정 프리셋

### DRAM

| 프리셋 | 공정 | Vdd | Cs | t_ret | 용도 |
|--------|------|-----|----|-------|------|
| `LPDDR5_PARAMS` | 12nm | 1.05V | 4.5fF | 32ms | 모바일 |
| `DDR5_PARAMS` | 15nm | 1.10V | 6fF | 32ms | 서버/PC |
| `DDR4_PARAMS` | 20nm | 1.20V | 20fF | 64ms | PC/서버 |
| `DDR3_PARAMS` | 30nm | 1.50V | 30fF | 64ms | 레거시 |

### SRAM

| 프리셋 | 공정 | Vdd | β | Vth_n/p |
|--------|------|-----|---|---------|
| `SRAM_7NM` | 7nm | 0.70V | 3.0 | 0.18/0.18V |
| `SRAM_14NM` | 14nm | 0.80V | 3.0 | 0.20/0.20V |
| `SRAM_28NM` | 28nm | 1.00V | 3.0 | 0.25/0.25V |
| `SRAM_65NM` | 65nm | 1.20V | 2.5 | 0.30/0.30V |

```python
from memory_engine import get_dram_preset, get_sram_preset, list_presets

# 프리셋 + 파라미터 오버라이드
hot_ddr4 = get_dram_preset("ddr4", T_k=370.0)   # 고온 시나리오
fast_sram = get_sram_preset("sram_7nm", beta_ratio=5.0)

# 사용 가능한 프리셋 목록
print(list_presets())
```

---

## API 레퍼런스

### DRAM 물리 (`dram_physics.py`)

```python
retention_tau(params)                                    # Arrhenius τ [s]
retention_decay(cell, params, dt_s)                      # RC 방전
read_disturb(cell, params)                               # 읽기 교란
refresh(cell, params)                                    # 전하 회복
dram_write(cell, params, value=1.0)                      # 쓰기
bitline_swing_fraction(cell, params)                     # ΔVbl/Vdd
dram_read_margin(cell, params)                           # 읽기 마진
refresh_needed(cell, params)                             # refresh 필요 판정
time_to_fail(params)                                     # retention 한계 시간 [s]
row_hammer_disturb(cell, params, n_hammers, loss)        # Row Hammer 적용
row_hammer_failure_threshold(params, loss)               # RH 실패 임계 hammer 수
```

### SRAM 물리 (`sram_physics.py`)

```python
vtc_curve(params, n_points=201)                          # VTC 커브 샘플링
static_noise_margin(params)                              # SNM [V]
read_noise_margin(params)                                # RNM = SNM × factor
write_margin(params)                                     # WM [V]
stability_index(params)                                  # SNM / (Vdd/2)
hold_snm(params)                                         # Hold 모드 SNM
read_node_disturb_v(params)                              # ΔV_read = Vdd/(β+1)
read_snm_physical(params)                                # SNM_hold − ΔV_read
snm_by_mode(params)                                      # 3모드 통합 dict
sram_initial_state(params, stored_high=True)             # 초기 셀 상태
sram_write(cell, params, value)                          # 쓰기
sram_read(cell, params)                                  # 읽기
sram_leakage_decay(cell, params, dt_s)                   # 누설 열화
```

### 센스앰프 (`sense_amplifier.py`)

```python
sense_op(cell, dram_params, sa_params)                  # SAObservation
sa_bit_error_rate(cell, dram_params, sa_params)          # BER (bit)
sa_row_ber(cell, dram_params, sa_params)                 # BER (row)
sa_min_delta_v_for_ber(target_ber, sa_params)            # Q⁻¹ 역함수
```

### Observer (`observer.py`)

```python
observe_dram(cell, params)     # → MemoryObservation
observe_sram(cell, params)     # → MemoryObservation
diagnose(obs)                  # → List[str]  권고 메시지
```

### 설계 엔진 (`design_engine.py`)

```python
# DRAM 시뮬레이션
simulate_dram_retention(cell, params, total_s, dt_s)
simulate_dram_read_disturb(cell, params, n_reads)
simulate_dram_refresh_cycle(cell, params, n_cycles, period_s)
sweep_dram_temperature(params, T_range)
sweep_dram_vdd(params, vdd_range)
sweep_dram_cs(params, cs_range)
verify_dram(cell, params)                                # → VerificationReport

# SRAM 시뮬레이션
simulate_sram_leakage(cell, params, total_s, dt_s)
sweep_sram_beta(params, beta_range)
sweep_sram_vdd(params, vdd_range)
sweep_sram_temperature(params, T_range)
verify_sram(cell, params)                                # → VerificationReport

# A1 센스앰프
simulate_sa_vs_retention(cell, dram_params, sa_params, total_s, dt_s)
sweep_sa_offset(cell, dram_params, sa_params_base, offset_range)

# A2 Row Hammer
simulate_row_hammer(victim_cell, dram_params, sa_params, total_hammers, step_hammers, loss)
find_row_hammer_threshold(dram_params, sa_params, loss)   # → Dict

# A3 SRAM 3모드
sram_mode_analysis(params)                               # → Dict
sweep_sram_mode_beta(params, beta_range)                 # → List[Dict]

# B 시나리오
scenario_high_temp_dram_collapse(params, T_range)
scenario_sram_beta_sweep_report(params, beta_range)
scenario_vdd_drop_cascade(dram_params, sram_params, sa_params, vdd_range)
```

### Battery-Memory 브리지 (`bridge_battery.py`)

```python
simulate_battery_discharge(current_a, dt_s, n_steps, params, soc_init)  # → List[BatteryStep]
vterm_to_vdd(v_term_cell, n_cells=1, pmic_efficiency=0.85)              # → float
battery_memory_cascade(battery_steps, dram_params, sram_params, sa_params, n_cells, η)
sweep_soc_memory_health(soc_range, batt_params, dram_params, sram_params, sa_params, ...)
```

---

## 테스트

```bash
# 전체 테스트 실행
cd memory_engine_repo
python -m pytest tests/ -v

# 섹션별 실행
python -m pytest tests/test_memory_engine.py -v    # §1~§13 (224 tests)
python -m pytest tests/test_scenarios.py -v        # §B1~§B3 (50 tests)
python -m pytest tests/test_bridge_battery.py -v   # §D1~§D4 (43 tests)
```

**테스트 구성 (총 267 tests):**

| 파일 | 섹션 | 테스트 수 |
|------|------|-----------|
| `test_memory_engine.py` | §1 DRAM 물리 / §2 파생지표 / §3 SRAM 물리 / §4~5 Observer / §6~7 설계엔진 / §8 검증 / §9 프리셋 / §10 진단 / §11 SA / §12 RH / §13 3모드 | 224 |
| `test_scenarios.py` | §B1 고온붕괴 / §B2 beta리포트 / §B3 Vdd연쇄 | 50 |
| `test_bridge_battery.py` | §D1 방전 / §D2 Vdd변환 / §D3 cascade / §D4 SOC스윕 | 43 |

---

## 버전 히스토리

| 버전 | 내용 |
|------|------|
| **v0.4.0** | Battery-Memory 브리지 (bridge_battery.py), ECM 내장, V_term→Vdd→cascade 연동 |
| **v0.3.0** | 시나리오 시뮬레이션 3종: 고온 DRAM 붕괴, SRAM beta 리포트, Vdd 연쇄 실패 |
| **v0.2.0** | A1 센스앰프 BER, A2 Row Hammer, A3 SRAM 3모드 SNM |
| **v0.1.0** | 초기 릴리스: DRAM/SRAM 물리, Observer Ω 5레이어, 공정 프리셋, 검증 엔진 |

---

## 관련 저장소

- **[SNN_Backends](https://github.com/qquartsco-svg/SNN_Backends)** — 스파이킹 신경망 백엔드 (LIF, Izhikevich, STDP)
- **[ENGINE_HUB](https://github.com/qquartsco-svg/ENGINE_HUB)** — 동역학 시뮬레이션 엔진 허브
- **[Connectome](https://github.com/qquartsco-svg/Connectome)** — 커넥톰 그래프 분석 엔진
