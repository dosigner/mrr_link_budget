# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MRR (Modulating RetroReflector) 기반 FSO (Free Space Optical) 통신 링크 버짓 시뮬레이터. MQW(Multiple Quantum Well) 기반의 UAV-지상국 간 광통신 링크 마진을 계산한다.

## Reference Paper
Dabiri et al., "Modulating Retroreflector Based Free Space Optical Link for UAV-to-Ground Communications", IEEE TWC 2022

## Link Budget Models
Two calculation approaches (`링크 마진 분석.md`):
1. **Antenna Model** - dB-based gain/loss calculations
2. **Optical Model** - Gaussian beam propagation (linear scale)

---

# Antenna Model Equations

## Uplink (GS → MRR) 수신 전력 [dB]
```
P_MRR_in = P_tx
         - L_tx_optics
         + G_tx
         - L_FSL
         - L_atm
         - L_scint
         - L_tracking
         - L_orientation
         - L_AR
         + G_rx_MRR
```

### Transmitter Gain (Full angle e^-2 divergence)
```
G_tx = 32 / theta_div^2  [linear]
G_tx_dB = 10*log10(32 / theta_div^2)
```

### Free Space Loss
```
L_FSL = (lambda / (4*pi*R))^2  [linear]
L_FSL_dB = 20*log10(lambda / (4*pi*R))
```

### Aperture Receiver Gain
```
G_rx_MRR = (pi * D_eff / lambda)^2  [linear]
```

## Downlink (MRR → GS) 수신 전력 [dB]
```
P_rx = P_MRR_in
     + G_MRR_rx
     - L_mod
     - L_FSL
     - L_atm
     - L_scint
     + G_GS_rx
     - L_rx_config
     - L_rx_optics
```

### MRR reflection Gain (with Strehl Ratio)
```
G_MRR = (pi * D_eff / lambda)^2 * S  [with Strehl ratio]
G_MRR = (pi * D_eff / lambda)^2 / (M^2)^2  [with M^2 beam quality]
```

### MRR Modulation Efficiency
```
M = exp(-alpha_On) - exp(-alpha_Off) = exp(-alpha_Off) * (C_MQW - 1)
```

### Receiver Configuration Loss
- Bi-static: TX/RX offset partial beam capture
- Concentric: Annular aperture loss

## Link Margin
```
Link_Margin = P_rx - Receiver_Sensitivity
```

---

# Optical Model Equations

## Channel Coefficient (Eq. 11)
```
h = h_lug * h_lgu * h_aug * h_agu * h_pu * h_pg * h_MRR
```

## Gaussian Beam Intensity (Eq. 3)
```
I_r(d, Z) = (2 / (pi * w_z^2)) * exp(-2*(x^2 + y^2) / w_z^2)
w_z = theta_div * Z
```

## Atmospheric Attenuation - Beer-Lambert (Eq. 7)
```
h_lgu = h_lug = exp(-Z * zeta)
```
- `zeta`: Scattering coefficient (Kim model for visibility)

## Uplink Geometric Loss with Pointing Error (Eq. 5)
```
h_pu = (2*A_r / (pi*w_z^2)) * exp(-2*(d_px^2 + d_py^2) / w_z^2)
d_px = Z * theta_ex,  d_py = Z * theta_ey
```

## Downlink Geometric Loss at GS (Eq. 6)
```
h_pg = 2*r_g^2 / w_zg^2 ≈ 2*r_g^2 / (Z^2 * theta_div^2)
```
## SNR (Eq. 12)
```
SNR = 2*R^2*P_t^2*h^2 / sigma_n^2
```

## Key Parameter K
```
K = w_z^2 / (Z^2 * sigma_theta_e^2)
```

---

# Atmospheric Turbulence (`turbulence_fading.md`)

## Cn^2 Profile Selection Logic
- **Case A (고도 정보 있음)**: Hufnagel-Valley 5/7 Model
- **Case B (고도 정보 없음)**: Constant Cn^2 (default: 5e-15 m^(-2/3))

## Hufnagel-Valley Model
```
C_n^2(h) = 0.00594*(V/27)^2 * (1e-5*h)^10 * exp(-h/1000)
         + 2.7e-16 * exp(-h/1500)
         + C_n^2(0) * exp(-h/100)
```
- `V`: Wind speed (m/s)
- `C_n^2(0)`: Ground turbulence (typical: 1.7e-14 m^(-2/3))

## Rytov Variance

### Uplink (Spherical Wave, Eq. 9)
```
sigma_R_up^2 = 9*(2*pi/lambda)^(7/6) * (Z/Z_hd)^(11/6)
             * integral[C_n^2(h) * (1 - (h-Z_hg)/Z_hd)^(5/6) * (h-Z_hg)^(5/6) dh]
```
Constant Cn^2 fallback:
```
sigma_R_up^2 = 0.5 * C_n^2 * k^(7/6) * Z^(11/6)
```

### Downlink (Plane Wave)
Constant Cn^2:
```
sigma_R_down^2 = 1.23 * C_n^2 * k^(7/6) * Z^(11/6)
```

## Fading Distribution Selection
- **sigma_R^2 < 1**: Log-Normal Distribution (Eq. 8)
- **sigma_R^2 >= 1**: Gamma-Gamma Distribution (Eq. 10)

### Log-Normal PDF
```
f_LN(h_a) = 1/(2*h_a*sqrt(2*pi*sigma_L^2)) * exp(-(ln(h_a) + 2*sigma_L^2)^2 / (8*sigma_L^2))
sigma_L^2 ≈ sigma_R^2 / 4
```

### Gamma-Gamma PDF
```
f_GG(h_a) = 2*(alpha*beta)^((alpha+beta)/2) / (Gamma(alpha)*Gamma(beta))
          * h_a^((alpha+beta)/2 - 1) * K_{alpha-beta}(2*sqrt(alpha*beta*h_a))
```

### Alpha/Beta Parameters (Plane Wave)
```
alpha = [exp(0.49*sigma_R^2 / (1 + 1.11*sigma_R^(12/5))^(7/6)) - 1]^(-1)
beta  = [exp(0.51*sigma_R^2 / (1 + 0.69*sigma_R^(12/5))^(5/6)) - 1]^(-1)
```
(Spherical Wave: 계수 0.49/0.51 대신 0.42/0.56)

---

# MRR-Specific Functions

## Angle-Dependent Efficiency (eta_mrr)
```python
def eta_mrr(theta_deg, knee_deg=2.12, max_deg=3.2):
    # theta <= knee: 1.0
    # knee < theta < max: smoothstep roll-off
    # theta >= max: 0.0
```

## Angle-Dependent M^2 Factor (mrr_m2)
```python
def mrr_m2(theta_deg, m2_min=1.3, m2_max=3.4, knee_deg=2.12, max_deg=3.2):
    # Affects downlink divergence
    # theta_downlink ≈ M^2 * 4*lambda / (pi*D_eff)
```