"""
대기 감쇠 모델 (Atmospheric Attenuation Models)

- Kim Model: 가시거리 기반 감쇠 (V >= 1km)
- Ijaz Model: 안개/연기 조건 감쇠 (V < 1km)
"""
import numpy as np
from typing import Literal


def _calculate_kim_q(visibility_km: float) -> float:
    """
    Kim model의 q(V) 계수 계산

    Parameters:
        visibility_km: 가시거리 [km]

    Returns:
        q 계수
    """
    if visibility_km > 50:
        return 1.6
    elif visibility_km > 6:
        return 1.3
    elif visibility_km > 1:
        return 0.16 * visibility_km + 0.34
    elif visibility_km > 0.5:
        return visibility_km - 0.5
    else:
        return 0.0


def kim_attenuation(wavelength_nm: float,
                    visibility_km: float,
                    distance_km: float) -> tuple[float, float]:
    """
    Kim model 대기 감쇠 계산 (V >= 1km 조건)

    수식:
        β [dB/km] = (3.91 / V) * (λ / 550)^(-q)
        L_atm = 10^(-β * Z / 10)

    Parameters:
        wavelength_nm: 파장 [nm]
        visibility_km: 가시거리 [km]
        distance_km: 전파 거리 [km]

    Returns:
        (attenuation_linear, attenuation_dB):
            - attenuation_linear: 투과율 (0~1)
            - attenuation_dB: 감쇠량 [dB] (양수 값, 손실 표현)
    """
    if visibility_km <= 0:
        raise ValueError("Visibility must be positive")

    q = _calculate_kim_q(visibility_km)

    # 감쇠 계수 [dB/km]
    beta_dB_per_km = (3.91 / visibility_km) * (wavelength_nm / 550.0) ** (-q)

    # 총 감쇠량 [dB]
    attenuation_dB = beta_dB_per_km * distance_km

    # 선형 투과율
    attenuation_linear = 10.0 ** (-attenuation_dB / 10.0)

    return attenuation_linear, attenuation_dB


def _calculate_ijaz_q(wavelength_um: float,
                      condition: Literal["fog", "smoke"]) -> float:
    """
    Ijaz model의 q(λ) 계수 계산

    Parameters:
        wavelength_um: 파장 [μm]
        condition: "fog" 또는 "smoke"

    Returns:
        q(λ) 계수
    """
    if condition == "fog":
        return 0.1428 * wavelength_um - 0.0947
    elif condition == "smoke":
        return 0.8467 * wavelength_um - 0.5212
    else:
        raise ValueError(f"Unknown condition: {condition}. Use 'fog' or 'smoke'.")


def ijaz_attenuation(wavelength_um: float,
                     visibility_km: float,
                     distance_km: float,
                     condition: Literal["fog", "smoke"] = "fog",
                     reference_wavelength_um: float = 1.55) -> tuple[float, float]:
    """
    Ijaz model 대기 감쇠 계산 (V < 1km, 안개/연기 조건)

    수식:
        q(λ) = 0.1428*λ - 0.0947  (Fog)
             = 0.8467*λ - 0.5212  (Smoke)
        β [dB/km] = (17 / V) * (λ / λ_o)^(-q(λ))
        L_fog = 10^(-β * Z / 10)

    Parameters:
        wavelength_um: 파장 [μm]
        visibility_km: 가시거리 [km]
        distance_km: 전파 거리 [km]
        condition: "fog" 또는 "smoke"
        reference_wavelength_um: 기준 파장 [μm] (default: 1.55)

    Returns:
        (attenuation_linear, attenuation_dB):
            - attenuation_linear: 투과율 (0~1)
            - attenuation_dB: 감쇠량 [dB] (양수 값, 손실 표현)
    """
    if visibility_km <= 0:
        raise ValueError("Visibility must be positive")

    q = _calculate_ijaz_q(wavelength_um, condition)

    # 감쇠 계수 [dB/km]
    beta_dB_per_km = (17.0 / visibility_km) * (wavelength_um / reference_wavelength_um) ** (-q)

    # 총 감쇠량 [dB]
    attenuation_dB = beta_dB_per_km * distance_km

    # 선형 투과율
    attenuation_linear = 10.0 ** (-attenuation_dB / 10.0)

    return attenuation_linear, attenuation_dB


def atmospheric_attenuation(wavelength_nm: float,
                            visibility_km: float,
                            distance_km: float,
                            fog_condition: Literal["fog", "smoke", None] = None) -> tuple[float, float]:
    """
    통합 대기 감쇠 계산 (Kim + Ijaz 모델 자동 선택)

    - V >= 1km: Kim model 사용
    - V < 1km: Ijaz model 사용 (fog_condition 필요)

    Parameters:
        wavelength_nm: 파장 [nm]
        visibility_km: 가시거리 [km]
        distance_km: 전파 거리 [km]
        fog_condition: V < 1km일 때 "fog" 또는 "smoke" (None이면 "fog" 기본값)

    Returns:
        (attenuation_linear, attenuation_dB)
    """
    if visibility_km >= 1.0:
        return kim_attenuation(wavelength_nm, visibility_km, distance_km)
    else:
        # V < 1km: Ijaz model
        wavelength_um = wavelength_nm / 1000.0
        condition = fog_condition if fog_condition is not None else "fog"
        return ijaz_attenuation(wavelength_um, visibility_km, distance_km, condition)


def beer_lambert_attenuation(scattering_coeff: float, distance_m: float) -> tuple[float, float]:
    """
    Beer-Lambert 법칙에 따른 대기 감쇠

    수식:
        h_l = exp(-ζ * Z)

    Parameters:
        scattering_coeff: 산란 계수 ζ [1/m]
        distance_m: 전파 거리 [m]

    Returns:
        (attenuation_linear, attenuation_dB)
    """
    attenuation_linear = np.exp(-scattering_coeff * distance_m)
    attenuation_dB = -10.0 * np.log10(attenuation_linear)

    return attenuation_linear, attenuation_dB
