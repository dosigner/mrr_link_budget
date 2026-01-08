"""
빔 전파 및 Pointing Error 계산
"""
import numpy as np
from typing import Optional, Union


def calculate_beam_diameter_at_distance(divergence_rad: float,
                                         distance_m: float,
                                         tx_diameter_m: Optional[float] = None) -> float:
    """
    거리 Z에서 빔 직경 계산

    Parameters:
        divergence_rad: 발산각 (half-angle) [rad]
        distance_m: 전파 거리 [m]
        tx_diameter_m: 송신부 초기 빔 직경 [m] (optional)

    Returns:
        D_beam(Z) = D_tx + 2*θ_div*Z  (tx_diameter 있을 때)
                  = 2*θ_div*Z         (tx_diameter 없을 때, far-field 근사)
    """
    beam_spread = 2.0 * divergence_rad * distance_m

    if tx_diameter_m is not None:
        return tx_diameter_m + beam_spread
    else:
        return beam_spread


def calculate_beam_radius_at_distance(divergence_rad: float,
                                       distance_m: float,
                                       tx_diameter_m: Optional[float] = None) -> float:
    """
    거리 Z에서 빔 반경 (직경의 절반)

    Returns:
        w_z = D_beam / 2
    """
    diameter = calculate_beam_diameter_at_distance(divergence_rad, distance_m, tx_diameter_m)
    return diameter / 2.0


def calculate_beam_footprint_area(beam_diameter_m: float) -> float:
    """
    빔 footprint 면적

    Parameters:
        beam_diameter_m: 빔 직경 [m]

    Returns:
        A = π * (D/2)²
    """
    radius = beam_diameter_m / 2.0
    return np.pi * radius ** 2


def gaussian_pointing_loss(d_p_m: float, w_z_m: float) -> tuple[float, float]:
    """
    Gaussian 빔 pointing error 손실

    h_pointing = exp(-2 * d_p² / w_z²)
    L_tracking_dB = 8.686 * d_p² / w_z²

    Parameters:
        d_p_m: Pointing displacement (radial) [m]
        w_z_m: Beam radius at receiver [m]

    Returns:
        (loss_linear, loss_dB): 선형 및 dB 값
    """
    exponent = -2.0 * (d_p_m ** 2) / (w_z_m ** 2)
    loss_linear = np.exp(exponent)
    loss_dB = -10.0 * np.log10(loss_linear) if loss_linear > 0 else float('inf')

    return loss_linear, loss_dB


def pointing_displacement_from_angle(angle_error_rad: float, distance_m: float) -> float:
    """
    각도 오차로부터 pointing displacement 계산

    d_p ≈ θ_e * Z (small angle approximation)

    Parameters:
        angle_error_rad: 추적 각도 오차 [rad]
        distance_m: 거리 [m]

    Returns:
        d_p [m]
    """
    return angle_error_rad * distance_m


def tracking_error_loss(sigma_theta_rad: float,
                        distance_m: float,
                        beam_radius_m: float,
                        n_sigma: float = 1.0) -> tuple[float, float]:
    """
    추적 오차에 의한 손실 (1σ 기준)

    Parameters:
        sigma_theta_rad: 추적 오차 σ [rad]
        distance_m: 거리 [m]
        beam_radius_m: 빔 반경 [m]
        n_sigma: σ 배수 (default: 1.0)

    Returns:
        (loss_linear, loss_dB)
    """
    d_p = n_sigma * sigma_theta_rad * distance_m
    return gaussian_pointing_loss(d_p, beam_radius_m)


def uplink_geometric_coefficient(aperture_area_m2: float,
                                  beam_radius_m: float,
                                  d_px_m: float = 0.0,
                                  d_py_m: float = 0.0) -> float:
    """
    Uplink 기하 손실 계수 (pointing error 포함)

    h_pu = (2*A_r / (π*w_z²)) * exp(-2*(d_px² + d_py²) / w_z²)

    Parameters:
        aperture_area_m2: MRR 유효 수신 면적 [m²]
        beam_radius_m: MRR 위치에서 빔 반경 [m]
        d_px_m: X축 pointing displacement [m]
        d_py_m: Y축 pointing displacement [m]

    Returns:
        h_pu (선형)
    """
    w_z_sq = beam_radius_m ** 2

    # 기하 손실 (aperture/footprint ratio)
    geometric_ratio = (2.0 * aperture_area_m2) / (np.pi * w_z_sq)

    # Pointing loss
    d_p_sq = d_px_m ** 2 + d_py_m ** 2
    pointing_factor = np.exp(-2.0 * d_p_sq / w_z_sq)

    return geometric_ratio * pointing_factor


def uplink_geometric_coefficient_separate(aperture_area_m2: float,
                                           beam_radius_m: float,
                                           d_px_m: float = 0.0,
                                           d_py_m: float = 0.0) -> tuple[float, float, float]:
    """
    Uplink 기하 손실 계수를 분리하여 반환

    h_pu = h_geometric * h_tracking
    h_geometric = 2*A_r / (π*w_z²)
    h_tracking = exp(-2*(d_px² + d_py²) / w_z²)

    Parameters:
        aperture_area_m2: MRR 유효 수신 면적 [m²]
        beam_radius_m: MRR 위치에서 빔 반경 [m]
        d_px_m: X축 pointing displacement [m]
        d_py_m: Y축 pointing displacement [m]

    Returns:
        (h_geometric, h_tracking, h_pu) 튜플
    """
    w_z_sq = beam_radius_m ** 2

    # 기하 손실 (aperture/footprint ratio)
    h_geometric = (2.0 * aperture_area_m2) / (np.pi * w_z_sq)

    # Pointing loss (tracking error)
    d_p_sq = d_px_m ** 2 + d_py_m ** 2
    h_tracking = np.exp(-2.0 * d_p_sq / w_z_sq)

    h_pu = h_geometric * h_tracking

    return h_geometric, h_tracking, h_pu


def downlink_geometric_coefficient(receiver_radius_m: float,
                                    beam_radius_m: float) -> float:
    """
    Downlink 기하 손실 계수 (Gaussian 빔 적분)

    h_pg = 1 - exp(-2*r_g² / w_zg²)

    Gaussian 빔의 원형 aperture 수광 효율:
    P_captured/P_total = ∫∫ I(r) dA / P_total
                       = 1 - exp(-2*r²/w²)

    Note: 기존 근사식 2*r²/w²는 r << w 일 때만 유효.
          r/w > 0.3 이면 오차가 10% 이상 발생.

    Parameters:
        receiver_radius_m: GS 수신기 반경 [m]
        beam_radius_m: GS 위치에서 빔 반경 [m]

    Returns:
        h_pg (선형)
    """
    exponent = -2.0 * (receiver_radius_m ** 2) / (beam_radius_m ** 2)
    return 1.0 - np.exp(exponent)


def downlink_divergence(wavelength_m: float,
                        mrr_effective_diameter_m: float,
                        m2: float = 1.0) -> float:
    """
    다운링크 발산각 계산 (half-angle)

    θ_half = M² × 2λ / (π × D_eff)

    Full-angle divergence: θ_full = M² × 4λ / (π × D_eff)
    Half-angle: θ_half = θ_full / 2

    Gaussian 빔 이론에서:
    - Far-field divergence (1/e²): θ = λ / (π × w₀)
    - M² 적용 시: θ = M² × λ / (π × w₀)
    - w₀ = D/2 이면: θ_half = M² × 2λ / (π × D)

    Parameters:
        wavelength_m: 파장 [m]
        mrr_effective_diameter_m: MRR 유효 직경 [m]
        m2: 빔 품질 인자 M²

    Returns:
        다운링크 발산각 (half-angle) [rad]
    """
    return m2 * 2.0 * wavelength_m / (np.pi * mrr_effective_diameter_m)


def sample_pointing_displacement(sigma_theta_rad: float,
                                  distance_m: float,
                                  n_samples: int,
                                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Pointing displacement 샘플링 (2D Gaussian)

    d_p = sqrt(d_px² + d_py²) ~ Rayleigh distribution

    Parameters:
        sigma_theta_rad: 추적 오차 σ [rad]
        distance_m: 거리 [m]
        n_samples: 샘플 수
        rng: numpy random generator

    Returns:
        d_p 샘플 배열 [m]
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_d = sigma_theta_rad * distance_m

    # 2D Gaussian → Rayleigh distribution for radial distance
    d_px = rng.normal(0, sigma_d, n_samples)
    d_py = rng.normal(0, sigma_d, n_samples)

    d_p = np.sqrt(d_px ** 2 + d_py ** 2)

    return d_p
