"""
MRR (Modulating Retro-Reflector) 효율 함수

- eta_mrr: 각도 의존적 MRR 효율 (FOV/clipping)
- mrr_m2: 각도 의존적 빔 품질 인자 M²
- mrr_reflection_ratio: MRR 반사 비율 (orientation fluctuation)
"""
import numpy as np
from typing import Union


def eta_mrr(theta_deg: float,
            knee_deg: float = 2.12,
            max_deg: float = 3.2) -> float:
    """
    각도 의존적 MRR 효율 (FOV / clipping gate)

    - theta <= knee: 1.0 (flat region)
    - knee < theta < max: smoothstep roll-off
    - theta >= max: 0.0 (outside FOV)

    Parameters:
        theta_deg: 입사각 오차 [deg]
        knee_deg: Flat-performance knee angle [deg] (default: 2.12)
        max_deg: Maximum usable FOV angle [deg] (default: 3.2)

    Returns:
        MRR 효율 [0, 1]
    """
    a = abs(theta_deg)

    # Flat region
    if a <= knee_deg:
        return 1.0

    # Outside FOV
    if a >= max_deg:
        return 0.0

    # Smooth roll-off region (C1 continuous - smoothstep)
    t = (a - knee_deg) / (max_deg - knee_deg)  # 0 → 1
    smooth = 3 * t ** 2 - 2 * t ** 3  # smoothstep
    return 1.0 - smooth


def eta_mrr_array(theta_deg: np.ndarray,
                  knee_deg: float = 2.12,
                  max_deg: float = 3.2) -> np.ndarray:
    """
    eta_mrr의 배열 버전

    Parameters:
        theta_deg: 입사각 오차 배열 [deg]
        knee_deg: Flat-performance knee angle [deg]
        max_deg: Maximum usable FOV angle [deg]

    Returns:
        MRR 효율 배열 [0, 1]
    """
    a = np.abs(theta_deg)
    result = np.ones_like(a)

    # Outside FOV
    result[a >= max_deg] = 0.0

    # Roll-off region
    mask = (a > knee_deg) & (a < max_deg)
    t = (a[mask] - knee_deg) / (max_deg - knee_deg)
    smooth = 3 * t ** 2 - 2 * t ** 3
    result[mask] = 1.0 - smooth

    return result


def mrr_m2(theta_deg: float,
           m2_min: float = 1.3,
           m2_max: float = 3.4,
           knee_deg: float = 2.12,
           max_deg: float = 3.2) -> float:
    """
    각도 의존적 유효 M² (MRR 광학 수차로 인한 빔 품질 저하)

    - theta <= knee: m2_min (diffraction-limited-like region)
    - theta >= max: m2_max (saturation at FOV edge)
    - knee < theta < max: smooth degradation

    다운링크 발산각에 영향:
        θ_downlink ≈ M² * 4λ / (π * D_eff)

    Parameters:
        theta_deg: 입사각 오차 [deg]
        m2_min: 최소 M² 값 (default: 1.3)
        m2_max: 최대 M² 값 (default: 3.4)
        knee_deg: Knee angle [deg] (default: 2.12)
        max_deg: Max FOV angle [deg] (default: 3.2)

    Returns:
        유효 M² 값
    """
    a = abs(theta_deg)

    # Flat (diffraction-limited-like) region
    if a <= knee_deg:
        return m2_min

    # Saturation at FOV edge
    if a >= max_deg:
        return m2_max

    # Smooth degradation (same smoothstep for consistency)
    t = (a - knee_deg) / (max_deg - knee_deg)
    smooth = 3 * t ** 2 - 2 * t ** 3
    return m2_min + (m2_max - m2_min) * smooth


def mrr_m2_array(theta_deg: np.ndarray,
                 m2_min: float = 1.3,
                 m2_max: float = 3.4,
                 knee_deg: float = 2.12,
                 max_deg: float = 3.2) -> np.ndarray:
    """
    mrr_m2의 배열 버전

    Parameters:
        theta_deg: 입사각 오차 배열 [deg]
        m2_min: 최소 M² 값
        m2_max: 최대 M² 값
        knee_deg: Knee angle [deg]
        max_deg: Max FOV angle [deg]

    Returns:
        유효 M² 배열
    """
    a = np.abs(theta_deg)
    result = np.full_like(a, m2_min)

    # Saturation region
    result[a >= max_deg] = m2_max

    # Degradation region
    mask = (a > knee_deg) & (a < max_deg)
    t = (a[mask] - knee_deg) / (max_deg - knee_deg)
    smooth = 3 * t ** 2 - 2 * t ** 3
    result[mask] = m2_min + (m2_max - m2_min) * smooth

    return result


def mrr_reflection_ratio_single(theta_n_deg: float) -> float:
    """
    단일 평면에서의 MRR 반사 비율

    h_MRRn = 1 - tan(θ_n)  for θ_n < 45°
           = 0             for θ_n >= 45°

    Parameters:
        theta_n_deg: 평면 내 각도 오차 [deg]

    Returns:
        반사 비율 [0, 1]
    """
    theta_rad = np.deg2rad(abs(theta_n_deg))

    if theta_rad >= np.pi / 4:  # >= 45°
        return 0.0

    ratio = 1.0 - np.tan(theta_rad)
    return max(0.0, ratio)


def mrr_reflection_ratio(theta_xy_deg: float,
                         theta_xz_deg: float,
                         theta_yz_deg: float) -> float:
    """
    MRR 반사 비율 (3축 orientation 고려)

    h_MRR = h_MRRxy * h_MRRxz * h_MRRyz

    Corner cube retro-reflector의 3개 직교 평면에서의 반사 비율

    Parameters:
        theta_xy_deg: x-y 평면 각도 오차 [deg]
        theta_xz_deg: x-z 평면 각도 오차 [deg]
        theta_yz_deg: y-z 평면 각도 오차 [deg]

    Returns:
        총 반사 비율 [0, 1]
    """
    h_xy = mrr_reflection_ratio_single(theta_xy_deg)
    h_xz = mrr_reflection_ratio_single(theta_xz_deg)
    h_yz = mrr_reflection_ratio_single(theta_yz_deg)

    return h_xy * h_xz * h_yz


def sample_mrr_orientation_fluctuation(sigma_theta_deg: float,
                                       n_samples: int,
                                       rng: np.random.Generator = None) -> np.ndarray:
    """
    UAV orientation fluctuation으로 인한 h_MRR 샘플링

    θ_xy, θ_xz, θ_yz ~ N(0, σ²)

    Parameters:
        sigma_theta_deg: Orientation fluctuation σ [deg]
        n_samples: 샘플 수
        rng: numpy random generator

    Returns:
        h_MRR 샘플 배열
    """
    if rng is None:
        rng = np.random.default_rng()

    # 3축 각도 샘플링
    theta_xy = rng.normal(0, sigma_theta_deg, n_samples)
    theta_xz = rng.normal(0, sigma_theta_deg, n_samples)
    theta_yz = rng.normal(0, sigma_theta_deg, n_samples)

    # h_MRR 계산
    h_mrr = np.zeros(n_samples)
    for i in range(n_samples):
        h_mrr[i] = mrr_reflection_ratio(theta_xy[i], theta_xz[i], theta_yz[i])

    return h_mrr


def mrr_orientation_loss_dB(sigma_theta_deg: float,
                            knee_deg: float = 2.12,
                            max_deg: float = 3.2) -> float:
    """
    UAV orientation fluctuation에 의한 평균 MRR 손실 [dB]

    eta_mrr 기반 손실 (간략화된 계산)

    Parameters:
        sigma_theta_deg: Orientation fluctuation σ [deg]
        knee_deg: eta_mrr knee angle
        max_deg: eta_mrr max angle

    Returns:
        평균 손실 [dB] (양수)
    """
    # 평균 eta 추정 (Monte Carlo)
    n_samples = 10000
    rng = np.random.default_rng(42)  # 재현성을 위한 시드

    theta_samples = np.abs(rng.normal(0, sigma_theta_deg, n_samples))
    eta_samples = eta_mrr_array(theta_samples, knee_deg, max_deg)

    mean_eta = np.mean(eta_samples)

    if mean_eta <= 0:
        return float('inf')

    loss_dB = -10.0 * np.log10(mean_eta)
    return loss_dB
