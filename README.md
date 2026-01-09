# MRR-FSO 링크 버짓 시뮬레이터 이론서

## 개요

본 문서는 **MRR (Modulating RetroReflector) 기반 FSO (Free Space Optical) 통신** 링크 버짓 시뮬레이터의 이론적 기반을 설명한다. MQW (Multiple Quantum Well) 기반의 UAV-지상국 간 광통신 링크 마진을 계산하기 위한 두 가지 모델 — **Antenna Model**과 **Optical Model** — 의 물리적 원리와 수학적 수식을 상세히 기술한다.

### 참조 논문
> Dabiri et al., "Modulating Retroreflector Based Free Space Optical Link for UAV-to-Ground Communications", IEEE Transactions on Wireless Communications, 2022

---

# 1. Antenna Model (dB 기반 링크 버짓)

Antenna Model은 **로그 스케일 (dB)** 에서 각 손실 요소를 개별적으로 계산하고 합산하는 전통적인 RF 링크 버짓 접근 방식을 광통신에 적용한 것이다. 이 모델은 직관적이고 각 손실 요소의 기여도를 명확히 파악할 수 있다는 장점이 있다.

## 1.1 시스템 구성

```
┌─────────────┐     Uplink      ┌─────────────┐     Downlink     ┌─────────────┐
│  지상국(GS) │ ───────────────▶│     MRR     │ ───────────────▶ │  지상국(GS) │
│  Transmitter│    λ = 1550nm   │  (on UAV)   │   Modulated      │  Receiver   │
└─────────────┘                 └─────────────┘                  └─────────────┘
```

## 1.2 Uplink 수신 전력 [dBm]

MRR에 도달하는 광 전력:

```
P_MRR_in = P_tx - L_tx_optics + G_tx + L_FSL - L_atm - L_scint - L_tracking - L_orientation - L_AR + G_rx_MRR
```

| 기호 | 설명 | 단위 |
|------|------|------|
| P_tx | 송신 전력 | dBm |
| L_tx_optics | 송신 광학계 손실 | dB |
| G_tx | 송신기 이득 | dB |
| L_FSL | 자유 공간 손실 | dB |
| L_atm | 대기 감쇠 손실 | dB |
| L_scint | 신틸레이션 손실 | dB |
| L_tracking | 추적 오차 손실 | dB |
| L_orientation | UAV 자세 오차 손실 | dB |
| L_AR | AR 코팅 손실 | dB |
| G_rx_MRR | MRR 수신 이득 | dB |

## 1.3 Downlink 수신 전력 [dBm]

지상국 수신기에 도달하는 광 전력:

```
P_rx = P_MRR_in + G_MRR_rx - L_mod - L_passive + L_FSL - L_atm - L_scint + G_rx_GS - L_rx_config - L_rx_optics
```

| 기호 | 설명 | 단위 |
|------|------|------|
| G_MRR_rx | MRR 반사 이득 (M² 보정 포함) | dB |
| L_mod | 변조 손실 | dB |
| L_passive | MRR passive 손실 | dB |
| L_rx_config | 수신부 구조 손실 (Bistatic/Concentric) | dB |
| L_rx_optics | 수신 광학계 손실 | dB |
| G_rx_GS | 지상국 수신 이득 | dB |

## 1.4 링크 마진

```
Link_Margin [dB] = P_rx - Receiver_Sensitivity
```

---

# 2. Antenna Model 구성 요소별 물리적 원리

## 2.1 송신기 이득 (Transmitter Gain)

### 물리적 원리
송신기 이득은 **등방성 안테나 대비** 특정 방향으로 집중된 에너지의 비율이다. Gaussian 빔에서 발산각이 작을수록 에너지가 더 집중되어 이득이 증가한다.

### 수식 유도
Gaussian 빔의 1/e² 전각(full-angle) 발산각 θ_div에 대해:

```
G_tx = 32 / θ_div²  [선형]

G_tx_dB = 10·log₁₀(32 / θ_div²)
```

**유도 과정:**
1. Gaussian 빔의 원거리장에서 입체각: Ω ≈ π·θ_div²/4
2. 등방성 복사의 입체각: 4π
3. 방향성 이득: G = 4π/Ω = 16/θ_div²
4. e⁻² 정의와 빔 전력 분포 보정 계수 2 적용: G = 32/θ_div²

### 코드 구현 (`src/models/common.py:146-165`)
```python
def transmitter_gain(divergence_rad: float) -> tuple[float, float]:
    full_angle = 2.0 * divergence_rad  # half-angle → full-angle
    gain_linear = 32.0 / (full_angle ** 2)
    gain_dB = 10.0 * np.log10(gain_linear)
    return gain_linear, gain_dB
```

---

## 2.2 자유 공간 손실 (Free Space Loss)

### 물리적 원리
자유 공간 손실은 **구면파 확산**으로 인한 전력 밀도 감소를 나타낸다. 전자기파가 거리 R만큼 전파될 때, 전력이 4πR² 면적에 균등하게 분포한다.

### Friis 전송 방정식
```
P_rx/P_tx = G_tx · G_rx · (λ / 4πR)²
```

여기서 자유 공간 손실 항:
```
L_FSL = (λ / 4πR)²  [선형]

L_FSL_dB = 20·log₁₀(λ / 4πR)  [음수, 손실]
```

### 물리적 해석
- λ: 파장 [m] — 파장이 길수록 손실 감소 (더 넓은 유효 수신 면적)
- R: 거리 [m] — 거리의 제곱에 비례하여 손실 증가

### 코드 구현 (`src/models/common.py:106-122`)
```python
def free_space_loss(wavelength_m: float, distance_m: float) -> tuple[float, float]:
    fsl_linear = (wavelength_m / (4.0 * np.pi * distance_m)) ** 2
    fsl_dB = 10.0 * np.log10(fsl_linear)  # 음수 값
    return fsl_linear, fsl_dB
```

---

## 2.3 대기 감쇠 (Atmospheric Attenuation)

### 물리적 원리
대기 중 입자(에어로졸, 수증기, 먼지)에 의한 **Mie 산란**과 분자에 의한 **Rayleigh 산란**이 광 신호를 감쇠시킨다.

### 2.3.1 Kim Model (가시거리 V ≥ 1 km)

맑은 날씨에서의 감쇠:

```
β [dB/km] = (3.91 / V) · (λ / 550)^(-q)
```

여기서 q는 가시거리에 따른 계수:

| 가시거리 V [km] | q 값 |
|----------------|------|
| V > 50 | 1.6 |
| 6 < V ≤ 50 | 1.3 |
| 1 < V ≤ 6 | 0.16V + 0.34 |
| 0.5 < V ≤ 1 | V - 0.5 |

총 감쇠:
```
L_atm = 10^(-β·Z/10)  [선형]
L_atm_dB = β · Z  [dB]
```

### 2.3.2 Ijaz Model (안개/연기 V < 1 km)

저시정 조건:
```
q(λ) = 0.1428·λ - 0.0947  (안개)
     = 0.8467·λ - 0.5212  (연기)

β [dB/km] = (17 / V) · (λ / λ_ref)^(-q)
```

### 2.3.3 Beer-Lambert 법칙

기본적인 지수 감쇠:
```
h_l = exp(-ζ · Z)
```
- ζ: 산란 계수 [1/m]
- Z: 전파 거리 [m]

### 코드 구현 (`src/atmosphere/attenuation.py`)
```python
def kim_attenuation(wavelength_nm, visibility_km, distance_km):
    q = _calculate_kim_q(visibility_km)
    beta_dB_per_km = (3.91 / visibility_km) * (wavelength_nm / 550.0) ** (-q)
    attenuation_dB = beta_dB_per_km * distance_km
    attenuation_linear = 10.0 ** (-attenuation_dB / 10.0)
    return attenuation_linear, attenuation_dB
```

---

## 2.4 대기 난류 및 신틸레이션 (Atmospheric Turbulence & Scintillation)

### 물리적 원리
대기의 온도 불균일로 인한 **굴절률 요동**이 광파면을 왜곡시켜 수신 전력의 시간적 변동(신틸레이션)을 야기한다.

### 2.4.1 굴절률 구조 상수 C_n²

**Hufnagel-Valley 5/7 프로파일** (고도 의존):
```
C_n²(h) = 0.00594·(V/27)² · (10⁻⁵·h)^10 · exp(-h/1000)
        + 2.7×10⁻¹⁶ · exp(-h/1500)
        + C_n²(0) · exp(-h/100)
```

| 항 | 물리적 의미 |
|----|------------|
| 1항 | 성층권 기여 (고고도 풍속 영향) |
| 2항 | 대류권 상부 기여 |
| 3항 | 지표면 경계층 기여 |

파라미터:
- V: 고고도 풍속 [m/s] (일반적으로 21 m/s)
- C_n²(0): 지표면 난류 강도 (일반적으로 1.7×10⁻¹⁴ m^(-2/3))

### 2.4.2 Rytov 분산 (Rytov Variance)

난류 강도를 정량화하는 무차원 파라미터:

**구면파 (Uplink, GS→MRR):**
```
σ_R² = 0.5 · C_n² · k^(7/6) · Z^(11/6)
```

**평면파 (Downlink, MRR→GS):**
```
σ_R² = 1.23 · C_n² · k^(7/6) · Z^(11/6)
```

여기서 k = 2π/λ (파수)

**고도 프로파일 적분 (정밀 계산):**
```
σ_R² = 9·(2π/λ)^(7/6) · (Z/Z_hd)^(11/6) · ∫[C_n²(h)·(1-(h-h_gs)/Z_hd)^(5/6)·(h-h_gs)^(5/6)] dh
```

### 2.4.3 페이딩 분포 선택

| σ_R² 범위 | 난류 강도 | 분포 모델 |
|----------|----------|----------|
| σ_R² < 1 | 약-중 | Log-Normal |
| σ_R² ≥ 1 | 중-강 | Gamma-Gamma |

**Log-Normal 분포:**
```
f_LN(h_a) = 1/(2·h_a·√(2π·σ_L²)) · exp(-(ln(h_a) + 2·σ_L²)² / (8·σ_L²))

σ_L² ≈ σ_R² / 4
```

**Gamma-Gamma 분포:**
```
f_GG(h_a) = 2·(αβ)^((α+β)/2) / (Γ(α)·Γ(β)) · h_a^((α+β)/2-1) · K_{α-β}(2·√(αβ·h_a))
```

α, β 파라미터 (평면파):
```
α = [exp(0.49·σ_R² / (1 + 1.11·σ_R^(12/5))^(7/6)) - 1]^(-1)
β = [exp(0.51·σ_R² / (1 + 0.69·σ_R^(12/5))^(5/6)) - 1]^(-1)
```

### 코드 구현 (`src/atmosphere/turbulence.py:68-99`)
```python
def calculate_rytov_variance_constant(wavelength_m, distance_m, cn2, wave_type):
    k = 2.0 * np.pi / wavelength_m
    if wave_type == "spherical":
        coefficient = 0.5   # Uplink
    else:
        coefficient = 1.23  # Downlink (plane wave)
    sigma_r2 = coefficient * cn2 * (k ** (7.0/6.0)) * (distance_m ** (11.0/6.0))
    return sigma_r2
```

---

## 2.5 추적 오차 손실 (Tracking/Pointing Loss)

### 물리적 원리
송신 빔이 수신기 중심에서 벗어나면 Gaussian 강도 프로파일에 따라 수신 전력이 감소한다.

### 수식
```
h_pointing = exp(-2·d_p² / w_z²)

L_tracking_dB = -10·log₁₀(h_pointing) = 8.686·d_p² / w_z²
```

여기서:
- d_p: 지향 오차에 의한 빔 중심 이탈 거리 [m]
- w_z: 수신기 위치에서 빔 반경 (1/e²) [m]

**각도 오차로부터 이탈 거리 계산:**
```
d_p = θ_e · Z  (소각 근사)
```

### 코드 구현 (`src/geometry/pointing.py:58-76`)
```python
def gaussian_pointing_loss(d_p_m: float, w_z_m: float) -> tuple[float, float]:
    exponent = -2.0 * (d_p_m ** 2) / (w_z_m ** 2)
    loss_linear = np.exp(exponent)
    loss_dB = -10.0 * np.log10(loss_linear)
    return loss_linear, loss_dB
```

---

## 2.6 Aperture 수신 이득 (Aperture Receiver Gain)

### 물리적 원리
수신 aperture가 자유 공간의 전력 밀도를 수집하는 유효 면적에 해당하는 이득이다. 이는 안테나 이론의 **유효 개구면(Effective Aperture)** 개념을 광학에 적용한 것이다.

### 수식
```
G_rx = η · (π·D / λ)²  [선형]

G_rx_dB = 10·log₁₀(η) + 20·log₁₀(π·D / λ)
```

여기서:
- D: aperture 직경 [m]
- λ: 파장 [m]
- η: aperture 효율 (0~1)

### 물리적 해석
- 유효 수신 면적: A_eff = G·λ²/(4π)
- 원형 aperture의 기하학적 면적: A_geo = π·(D/2)²
- η = 1일 때 A_eff = A_geo

### 코드 구현 (`src/models/common.py:126-143`)
```python
def aperture_gain(diameter_m: float, wavelength_m: float, efficiency: float = 1.0):
    gain_linear = efficiency * (np.pi * diameter_m / wavelength_m) ** 2
    gain_dB = 10.0 * np.log10(gain_linear)
    return gain_linear, gain_dB
```

---

## 2.7 MRR 반사 이득 (MRR Reflection Gain)

### 물리적 원리
Corner-cube retroreflector는 입사광을 **정확히 입사 방향으로 반사**한다. 그러나 실제 MRR은 광학적 수차로 인해 반사 빔의 품질이 저하되며, 이를 **M² (빔 품질 인자)** 로 정량화한다.

### 수식
```
G_MRR_rx = (π·D_eff / λ)² / (M²)²  [선형]

G_MRR_rx_dB = G_rx_dB - 20·log₁₀(M²)
```

### M² 빔 품질 인자
- M² = 1: 이상적인 회절 한계 빔
- M² > 1: 실제 빔 (수차로 인한 품질 저하)

**M²과 Strehl Ratio의 관계:**
```
M² ≈ 1 / √S
```

### 코드 구현 (`src/models/antenna_model.py:210-214`)
```python
_, G_mrr_rx_base_dB = aperture_gain(mrr_diameter_m, wavelength_m)
G_mrr_rx_dB = G_mrr_rx_base_dB - 20.0 * np.log10(mrr_m2_value)
```

---

## 2.8 다운링크 발산각 (Downlink Divergence)

### 물리적 원리
MRR에서 반사된 빔의 발산각은 **회절 한계**와 **빔 품질 저하(M²)** 에 의해 결정된다.

### Gaussian 빔 이론
```
θ_half = M² · 2λ / (π·D_eff)  [반각]

θ_full = M² · 4λ / (π·D_eff)  [전각]
```

### 물리적 해석
- 이상적 회절 한계 (M²=1): θ = 2λ/(π·D)
- MRR의 수차: M² 배 만큼 발산각 증가
- 결과: 지상국에서 더 큰 빔 footprint → 수신 전력 밀도 감소

### 코드 구현 (`src/geometry/pointing.py:204-228`)
```python
def downlink_divergence(wavelength_m, mrr_effective_diameter_m, m2=1.0):
    return m2 * 2.0 * wavelength_m / (np.pi * mrr_effective_diameter_m)
```

---

## 2.9 수신부 구조 손실 (Receiver Configuration Loss)

### 2.9.1 Bistatic 구조

송신기와 수신기가 **offset** 만큼 분리되어 있어 반사 빔 중심에서 벗어남.

```
L_bistatic = P_centered / P_offset
```

### 2.9.2 Concentric 구조

송신기가 중앙에 위치하고 수신기가 **환형(annular)** 형태로 둘러싸는 구조.

```
P_annulus = P(r < r_outer) - P(r < r_inner)
         = [1 - exp(-2·r_outer²/w²)] - [1 - exp(-2·r_inner²/w²)]
```

### 코드 구현 (`src/geometry/receiver.py:219-260`)
```python
def bistatic_loss(beam_radius_m, offset_m, rx_radius_m, profile):
    power_centered = beam_power_in_circle(beam_radius_m, rx_radius_m, 0.0, profile)
    power_offset = beam_power_in_circle(beam_radius_m, rx_radius_m, offset_m, profile)
    efficiency_ratio = power_offset / power_centered
    loss_dB = -10.0 * np.log10(efficiency_ratio)
    return efficiency_ratio, loss_dB
```

---

# 3. Optical Model (선형 채널 계수)

Optical Model은 **선형 스케일**에서 각 채널 계수를 곱하여 전체 채널 응답을 계산한다. 이 모델은 Gaussian 빔 전파 이론에 기반하며, 확률적 시뮬레이션(Monte Carlo)에 적합하다.

## 3.1 전체 채널 계수 (Total Channel Coefficient)

```
h_total = h_uplink · h_MRR · h_downlink · η_optics
```

각 구성 요소:
```
h_uplink = h_pu · h_lgu · h_aug · h_orientation
h_downlink = h_pg · h_lgu_down · h_agu
h_MRR = h_modulation
η_optics = η_tx · η_rx
```

## 3.2 수신 전력 계산

```
P_rx [W] = P_tx [W] · h_total
```

---

# 4. Optical Model 구성 요소별 물리적 원리

## 4.1 Uplink 기하 손실 계수 (h_pu)

### 물리적 원리
Gaussian 빔이 거리 Z를 전파한 후 MRR aperture가 수집하는 전력 비율과 지향 오차의 영향을 결합.

### 수식
```
h_pu = h_geometric · h_tracking

h_geometric = 2·A_r / (π·w_z²)
h_tracking = exp(-2·(d_px² + d_py²) / w_z²)
```

여기서:
- A_r: MRR 유효 수신 면적 [m²]
- w_z: MRR 위치에서 빔 반경 [m]
- d_px, d_py: X, Y 방향 지향 이탈 [m]

### Gaussian 빔 강도 분포
```
I(r, Z) = (2 / π·w_z²) · exp(-2·r² / w_z²)
```

빔 반경의 거리 의존성:
```
w_z = w_0 · √(1 + (Z/z_R)²) ≈ θ_div · Z  (원거리장)
```

### 코드 구현 (`src/geometry/pointing.py:145-176`)
```python
def uplink_geometric_coefficient_separate(aperture_area_m2, beam_radius_m, d_px_m, d_py_m):
    w_z_sq = beam_radius_m ** 2
    h_geometric = (2.0 * aperture_area_m2) / (np.pi * w_z_sq)
    d_p_sq = d_px_m ** 2 + d_py_m ** 2
    h_tracking = np.exp(-2.0 * d_p_sq / w_z_sq)
    h_pu = h_geometric * h_tracking
    return h_geometric, h_tracking, h_pu
```

---

## 4.2 대기 감쇠 계수 (h_lgu)

### 물리적 원리
Beer-Lambert 법칙에 따른 지수 감쇠.

### 수식
```
h_lgu = h_lgu_down = exp(-ζ · Z) = 10^(-L_atm_dB / 10)
```

---

## 4.3 신틸레이션 페이딩 계수 (h_aug, h_agu)

### 물리적 원리
대기 난류로 인한 수신 전력의 **확률적 변동**을 확정적 손실로 변환.

### 수식
특정 확률(예: 99%)에서의 페이딩 마진:
```
h_aug = 10^(-L_scint_up_dB / 10)
h_agu = 10^(-L_scint_down_dB / 10)
```

신틸레이션 손실은 페이딩 분포의 하위 백분위수에서 결정:
- σ_R² < 1: Log-Normal CDF의 역함수
- σ_R² ≥ 1: Gamma-Gamma CDF의 수치적 역함수

---

## 4.4 Downlink 수광 계수 (h_pg)

### 물리적 원리
Gaussian 빔에서 **원형 aperture가 수집하는 전력 비율**을 정확하게 계산.

### 수식 (정확한 적분)
```
h_pg = 1 - exp(-2·r_g² / w_zg²)
```

여기서:
- r_g: 지상국 수신 aperture 반경 [m]
- w_zg: 지상국 위치에서 빔 반경 [m]

### 선형 근사와의 비교
기존 근사식: h_pg ≈ 2·r_g²/w_zg²

이 근사는 r_g << w_zg 일 때만 유효하며, r_g/w_zg > 0.3이면 10% 이상의 오차 발생.

### 코드 구현 (`src/geometry/pointing.py:179-201`)
```python
def downlink_geometric_coefficient(receiver_radius_m, beam_radius_m):
    exponent = -2.0 * (receiver_radius_m ** 2) / (beam_radius_m ** 2)
    return 1.0 - np.exp(exponent)
```

---

## 4.5 MRR 자세 오차 손실 (h_orientation)

### 물리적 원리
UAV의 자세 변동으로 MRR의 입사각이 변하면 **효율 저하**와 **반사 비율 감소**가 발생.

### 각도 의존적 MRR 효율
```
η_mrr(θ) = 1.0                      [θ ≤ θ_knee]
         = smoothstep roll-off      [θ_knee < θ < θ_max]
         = 0.0                      [θ ≥ θ_max]
```

Smoothstep 함수:
```
t = (θ - θ_knee) / (θ_max - θ_knee)
smooth = 3t² - 2t³
η = 1 - smooth
```

### 코드 구현 (`src/mrr/efficiency.py:12-43`)
```python
def eta_mrr(theta_deg, knee_deg=2.12, max_deg=3.2):
    a = abs(theta_deg)
    if a <= knee_deg:
        return 1.0
    if a >= max_deg:
        return 0.0
    t = (a - knee_deg) / (max_deg - knee_deg)
    smooth = 3 * t ** 2 - 2 * t ** 3
    return 1.0 - smooth
```

---

## 4.6 각도 의존적 M² 인자

### 물리적 원리
MRR의 입사각이 증가하면 광학적 수차가 증가하여 **반사 빔의 품질이 저하**된다.

### 수식
```
M²(θ) = M²_min                                    [θ ≤ θ_knee]
      = M²_min + (M²_max - M²_min)·smoothstep    [θ_knee < θ < θ_max]
      = M²_max                                    [θ ≥ θ_max]
```

일반적인 값:
- M²_min = 1.3 (정상 입사 근처)
- M²_max = 3.4 (FOV 가장자리)

### 다운링크에 미치는 영향
```
θ_downlink = M² · 4λ / (π·D_eff)
```
M²가 증가하면 발산각 증가 → 지상국에서 더 넓은 빔 → 수신 전력 밀도 감소

---

## 4.7 Corner-Cube 반사 비율 (3축)

### 물리적 원리
Corner-cube retroreflector는 **3개의 직교 평면**으로 구성되며, 각 평면에서의 입사각에 따라 반사 효율이 달라진다.

### 수식
```
h_MRR = h_MRR_xy · h_MRR_xz · h_MRR_yz

h_MRR_n = 1 - tan(θ_n)  [θ_n < 45°]
        = 0              [θ_n ≥ 45°]
```

### 코드 구현 (`src/mrr/efficiency.py:149-193`)
```python
def mrr_reflection_ratio_single(theta_n_deg):
    theta_rad = np.deg2rad(abs(theta_n_deg))
    if theta_rad >= np.pi / 4:  # >= 45°
        return 0.0
    ratio = 1.0 - np.tan(theta_rad)
    return max(0.0, ratio)

def mrr_reflection_ratio(theta_xy_deg, theta_xz_deg, theta_yz_deg):
    h_xy = mrr_reflection_ratio_single(theta_xy_deg)
    h_xz = mrr_reflection_ratio_single(theta_xz_deg)
    h_yz = mrr_reflection_ratio_single(theta_yz_deg)
    return h_xy * h_xz * h_yz
```

---

# 5. Antenna Model vs Optical Model 비교

## 5.1 특성 비교표

| 특성 | Antenna Model | Optical Model |
|------|---------------|---------------|
| **계산 도메인** | 로그 (dB) | 선형 |
| **손실 결합 방식** | 덧셈/뺄셈 | 곱셈 |
| **기본 접근** | RF 링크 버짓 | Gaussian 빔 전파 |
| **직관성** | 높음 (각 항목 분리) | 중간 |
| **Monte Carlo 적합성** | 낮음 | 높음 |
| **확률적 분석** | 어려움 | 용이 |
| **정밀도** | 중간 | 높음 |

## 5.2 수식 대응 관계

| 물리적 현상 | Antenna Model [dB] | Optical Model [선형] |
|------------|-------------------|---------------------|
| 자유 공간 손실 | L_FSL = 20·log₁₀(λ/4πR) | FSL = (λ/4πR)² |
| 대기 감쇠 | L_atm [dB] | h_l = 10^(-L_atm/10) |
| 신틸레이션 | L_scint [dB] | h_a = 10^(-L_scint/10) |
| 추적 오차 | L_tracking [dB] | h_tracking = exp(-2d_p²/w_z²) |
| 수신기 이득 | G_rx = 10·log₁₀(πD/λ)² | (암묵적으로 h_pu, h_pg에 포함) |
| 변조 손실 | L_mod = -10·log₁₀(M) | h_mod = M |

## 5.3 계산 흐름 비교

### Antenna Model
```
P_rx [dBm] = P_tx [dBm] + Σ(Gains) - Σ(Losses)
```

### Optical Model
```
P_rx [W] = P_tx [W] × Π(Channel Coefficients)
```

## 5.4 사용 권장 시나리오

| 시나리오 | 권장 모델 | 이유 |
|---------|----------|------|
| 초기 시스템 설계 | Antenna | 직관적 분석, 각 요소 기여도 파악 |
| 상세 성능 분석 | Optical | 물리적 정확성 |
| Monte Carlo 시뮬레이션 | Optical | 확률적 처리 용이 |
| 링크 마진 빠른 계산 | Antenna | 단순한 덧셈/뺄셈 |
| 빔 프로파일 최적화 | Optical | Gaussian 빔 이론 적용 |

---

# 6. MQW (Multiple Quantum Well) 변조기

## 6.1 구조 및 동작 원리

### 6.1.1 양자우물 구조
MQW 변조기는 **얇은 반도체층(우물)**과 **넓은 밴드갭 장벽층**이 교대로 적층된 구조이다.

```
┌─────────────────────────────────────────────────────────┐
│                     상부 전극                           │
├─────────────────────────────────────────────────────────┤
│ ▓▓▓▓▓▓ 장벽 (AlGaAs) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ░░░░░░ 양자우물 (GaAs) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ▓▓▓▓▓▓ 장벽 (AlGaAs) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ░░░░░░ 양자우물 (GaAs) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ▓▓▓▓▓▓ 장벽 (AlGaAs) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│         ... (수십~수백 층 반복) ...                     │
├─────────────────────────────────────────────────────────┤
│                     하부 전극                           │
└─────────────────────────────────────────────────────────┘
```

### 6.1.2 QCSE (Quantum-Confined Stark Effect)

**물리적 원리:**
외부 전기장이 인가되면 양자우물 내 전자와 정공의 파동함수가 반대 방향으로 이동하여:
1. **엑시톤 에너지 준위 감소** → 흡수 스펙트럼 적색 이동 (Red-shift)
2. **파동함수 중첩 감소** → 흡수 계수 감소

```
전기장 OFF (V = 0):              전기장 ON (V > 0):
     ┌────────┐                      ╱────────╲
  e⁻ │████████│                   e⁻╱          ╲
     │   ○    │                    ╱     ○      ╲
     │   ○    │  ─────────▶       ╱              ╲
     │   ●    │                  ╱        ●       ╲
  h⁺ │████████│               h⁺╲                ╱
     └────────┘                  ╲──────────────╱

  강한 흡수                        약한 흡수 (적색 이동)
```

## 6.2 변조 효율 수식

### 6.2.1 기본 정의

**투과율 (Transmittance):**
```
T = exp(-α · d)
```
- α: 흡수 계수 [1/m]
- d: MQW 총 두께 [m]

**ON/OFF 상태:**
- T_on: 전기장 인가 시 투과율 (높음)
- T_off: 전기장 미인가 시 투과율 (낮음)

### 6.2.2 Contrast Ratio
```
C_MQW = T_on / T_off = exp(α_off - α_on)
```

### 6.2.3 변조 효율 (Modulation Efficiency)

**직접 계산:**
```
M = exp(-α_on) - exp(-α_off)
```

**Contrast Ratio 기반:**
```
M = exp(-α_off) · (C_MQW - 1)
```

### 수식 유도
```
M = T_on - T_off
  = exp(-α_on) - exp(-α_off)
  = exp(-α_off) · [exp(α_off - α_on) - 1]
  = exp(-α_off) · (C_MQW - 1)
```

### 6.2.4 변조 손실 [dB]
```
L_mod = -10 · log₁₀(M)
```

### 코드 구현 (`src/mrr/modulation.py`)
```python
def mqw_modulation_efficiency(alpha_on: float, alpha_off: float) -> float:
    return np.exp(-alpha_on) - np.exp(-alpha_off)

def mqw_modulation_efficiency_from_contrast(contrast_ratio: float, alpha_off: float = 0.1) -> float:
    return np.exp(-alpha_off) * (contrast_ratio - 1)

def modulation_loss_dB(modulation_efficiency: float) -> float:
    return -10.0 * np.log10(modulation_efficiency)
```

## 6.3 전형적인 MQW 파라미터

| 파라미터 | 전형적 값 | 범위 |
|---------|----------|------|
| 동작 파장 | 1550 nm | 1520-1570 nm |
| Contrast Ratio (C_MQW) | 2-5 | 1.5-10 |
| OFF 상태 흡수 (α_off) | 0.1 | 0.05-0.3 |
| 변조 효율 (M) | 0.3-0.6 | 0.1-0.8 |
| 변조 속도 | 10-100 MHz | 1 MHz - 1 GHz |
| 구동 전압 | 5-15 V | 3-30 V |

## 6.4 추가 고려 손실

### 6.4.1 MRR Passive 손실

**AR 코팅 손실:**
- 원인: 입사면의 반사
- 전형적 값: 0.3-1.0 dB

**반사면 손실:**
- 원인: 불완전한 반사 (Al 코팅: ~96%, Au 코팅: ~98%)
- 전형적 값: 0.2-0.5 dB

```python
def mrr_passive_loss_dB(ar_coating_loss_dB=0.5, reflectivity_loss_dB=0.3):
    return ar_coating_loss_dB + reflectivity_loss_dB
```

### 6.4.2 온도 의존성

**원인:**
- 밴드갭 에너지의 온도 의존성
- 흡수 스펙트럼 이동

**영향:**
- 동작 파장 이동: ~0.5 nm/K
- Contrast ratio 변화

**대책:**
- TEC (열전 냉각기) 탑재
- 넓은 온도 범위 설계

### 6.4.3 삽입 손실 (Insertion Loss)

**구성 요소:**
1. 결합 손실: 광섬유-MQW 인터페이스 (0.5-2 dB)
2. 전파 손실: MQW 내부 산란 (0.1-0.5 dB)
3. 기판 흡수: GaAs 기판 (0.1-0.3 dB)

**총 삽입 손실:** 1-4 dB (설계에 따라 다름)

### 6.4.4 편광 의존성

**원인:**
- 양자우물의 TE/TM 모드 차등 흡수
- Exciton 전이 선택 규칙

**PDL (Polarization Dependent Loss):**
- 전형적 값: 0.5-2 dB
- 고급 설계: < 0.5 dB

### 6.4.5 비선형 효과

**Two-Photon Absorption (TPA):**
- 고출력에서 추가 흡수
- 임계 강도: ~10 MW/cm²

**Self-Phase Modulation:**
- 굴절률 변화로 인한 위상 왜곡
- 고속 변조 시 신호 품질 저하

### 6.4.6 파장 의존성

**대역폭 제한:**
- 동작 대역폭: 30-50 nm (전형적)
- 중심 파장에서 멀어지면 Contrast ratio 감소

**Spectral Flatness:**
- 대역 내 Contrast ratio 변동: ±10-20%

---

# 7. 고급 주제

## 7.1 Monte Carlo 시뮬레이션

### 7.1.1 확률적 소스 모델링

**신틸레이션 페이딩:**
```python
if sigma_r2 < 1:
    h_scint = sample_lognormal_fading(sigma_r2, n_samples)
else:
    h_scint = sample_gamma_gamma_fading(sigma_r2, n_samples, wave_type)
```

**지향 오차:**
```python
d_p = sample_pointing_displacement(sigma_theta_rad, distance_m, n_samples)
# d_p는 Rayleigh 분포를 따름 (2D Gaussian의 반경)
```

**MRR 자세 변동:**
```python
theta_samples = rng.normal(0, sigma_theta_deg, n_samples)
h_orientation = eta_mrr_array(theta_samples)
```

### 7.1.2 Outage Probability

```
P_outage = P(P_rx < P_sensitivity) = N(P_rx < P_sensitivity) / N_total
```

### 7.1.3 BER 계산

**OOK (On-Off Keying):**
```
BER_OOK = 0.5 · erfc(√(SNR/2))
```

**BPSK:**
```
BER_BPSK = 0.5 · erfc(√SNR)
```

## 7.2 Parameter Sweep 분석

지원되는 sweep 파라미터:
- 거리 (Distance)
- 가시거리 (Visibility)
- 자세 오차 (Orientation error)
- 추적 오차 (Tracking error)
- MRR 직경
- 송신 전력
- 발산각
- C_n²

---

# 8. 참고 문헌

1. Dabiri et al., "Modulating Retroreflector Based Free Space Optical Link for UAV-to-Ground Communications", IEEE TWC, 2022

2. L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random Media", SPIE Press, 2005

3. H. Kaushal and G. Kaddoum, "Optical Communication in Space: Challenges and Mitigation Techniques", IEEE Communications Surveys & Tutorials, 2017

4. D. A. B. Miller et al., "Band-Edge Electroabsorption in Quantum Well Structures: The Quantum-Confined Stark Effect", Physical Review Letters, 1984

5. G. D. Boyd and D. A. Kleinman, "Parametric Interaction of Focused Gaussian Light Beams", Journal of Applied Physics, 1968

---

# 부록 A: 단위 변환

```python
# 파장
nm_to_m(nm) = nm × 10⁻⁹
um_to_m(um) = um × 10⁻⁶

# 각도
deg_to_rad(deg) = deg × π/180
mrad_to_rad(mrad) = mrad × 10⁻³
urad_to_rad(urad) = urad × 10⁻⁶

# 전력
dBm_to_W(dBm) = 10^((dBm-30)/10)
W_to_dBm(W) = 10·log₁₀(W) + 30

# dB 변환
linear_to_dB(x) = 10·log₁₀(x)
dB_to_linear(dB) = 10^(dB/10)
```

---

# 부록 B: 물리 상수

```python
SPEED_OF_LIGHT = 299,792,458 m/s
PLANCK_CONSTANT = 6.626 × 10⁻³⁴ J·s
```

---

*본 문서는 MRR-FSO 링크 버짓 시뮬레이터의 이론적 기반을 설명하며, 실제 구현은 `src/` 디렉토리의 Python 모듈을 참조하시기 바랍니다.*
