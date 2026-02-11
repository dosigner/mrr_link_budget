"""
Optical Model (Gaussian 빔 전파 기반 채널 계수 계산)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

from ..models.common import (
    nm_to_m, cm_to_m, mm_to_m, mrad_to_rad, deg_to_rad, urad_to_rad,
    linear_to_dB, dB_to_linear, dBm_to_W, W_to_dBm
)
from ..models.results import (
    BeamParameters, OpticalModelResult
)
from ..atmosphere.attenuation import atmospheric_attenuation
from ..atmosphere.turbulence import (
    calculate_rytov_variance_constant, hufnagel_valley_cn2,
    calculate_rytov_variance_profile, scintillation_loss_dB
)
from ..mrr.efficiency import mrr_m2, mrr_orientation_loss_dB
from ..mrr.modulation import mqw_modulation_efficiency_from_contrast
from ..geometry.pointing import (
    calculate_beam_diameter_at_distance, calculate_beam_radius_at_distance,
    uplink_geometric_coefficient_separate,
    downlink_divergence, calculate_beam_footprint_area
)
from ..geometry.receiver import downlink_collection_coefficient


@dataclass
class OpticalModelParams:
    """Optical Model 입력 파라미터"""
    # 송신부
    P_tx_W: float = 1.0                 # 송신 전력 [W]
    eta_tx: float = 0.8                 # 송신 광학계 효율 (0~1)
    wavelength_nm: float = 1550.0       # 파장 [nm]
    theta_div_full_mrad: float = 1.0    # 발산각 (full-angle) [mrad]
    tx_beam_diameter_mm: Optional[float] = None  # 송신부 빔 직경 [mm] (optional)

    # 링크
    distance_m: float = 1000.0          # 링크 거리 [m]
    visibility_km: float = 10.0         # 가시거리 [km]
    fog_condition: Optional[Literal["fog", "smoke"]] = None  # 안개 조건

    # MRR
    mrr_diameter_cm: float = 2.0        # MRR 유효 직경 [cm]
    sigma_orientation_deg: float = 3.0  # UAV 자세 오차 σ [deg]
    mrr_knee_deg: float = 2.12          # eta_mrr knee angle
    mrr_max_deg: float = 3.2            # eta_mrr max angle
    mrr_m2_min: float = 1.3             # M² 최소값 (use_strehl=False일 때)
    mrr_m2_max: float = 3.4             # M² 최대값

    # M² 입력 방식 선택
    use_strehl: bool = False            # True: Strehl ratio로 M² 계산 / False: M² 직접 입력
    strehl_ratio: float = 0.8           # Strehl ratio → M² = 1/√Strehl

    # MQW 변조 파라미터
    use_mqw_params: bool = False        # True: α_off, C_MQW 사용 / False: modulation_efficiency 직접 사용
    alpha_off: float = 0.1              # OFF 상태 흡수 계수 α_off
    c_mqw: float = 3.0                  # Contrast ratio C_MQW (T_on/T_off)
    modulation_efficiency: float = 0.5  # 변조 효율 M (0~1) - use_mqw_params=False일 때 사용

    # 수신부
    rx_diameter_cm: float = 16.0        # GS 수신 직경 [cm]
    eta_rx: float = 0.8                 # 수신 광학계 효율 (0~1)
    rx_config: Literal["bistatic", "concentric"] = "bistatic"
    rx_offset_cm: float = 10.0           # Bistatic offset [cm]
    tx_inner_diameter_cm: float = 5.0   # Concentric TX 내경 [cm]
    beam_profile: Literal["uniform", "gaussian", "airy"] = "uniform"  # 다운링크 빔 프로파일

    # 추적 (지향 오차)
    tracking_offset_urad: float = 100.0  # 지향 오차 각도 [μrad]

    # 난류
    use_altitude_profile: bool = False  # 고도 프로파일 사용 여부
    cn2_constant: float = 5e-15         # 상수 Cn² [m^(-2/3)]
    h_gs_m: float = 0.0                 # GS 고도 [m]
    h_uav_m: float = 100.0              # UAV 고도 [m]
    wind_speed_ms: float = 21.0         # 풍속 [m/s]

    # 난류 (Turbulence)
    use_turbulence: bool = True         # 난류 손실 적용 여부
    scint_probability: float = 0.99     # 마진 확률 (0.99 = 99% 가용성)


def calculate_optical_channel_coefficient(params: OpticalModelParams) -> OpticalModelResult:
    """
    Optical Model 채널 계수 계산

    h = h_l * h_a * h_p * h_MRR

    Where:
        h_l = h_lug * h_lgu : Uplink/Downlink geometric + atmospheric loss
        h_a = h_aug * h_agu : Uplink/Downlink scintillation fading
        h_p = h_pu * h_pg   : Uplink/Downlink pointing loss
        h_MRR               : MRR efficiency (eta, modulation, reflection)

    Parameters:
        params: OpticalModelParams 인스턴스

    Returns:
        OpticalModelResult 인스턴스
    """
    # === 단위 변환 ===
    wavelength_m = nm_to_m(params.wavelength_nm)
    # Full divergence → Half-angle (내부 계산용)
    divergence_half_rad = mrad_to_rad(params.theta_div_full_mrad) / 2.0
    divergence_full_rad = mrad_to_rad(params.theta_div_full_mrad)
    mrr_diameter_m = cm_to_m(params.mrr_diameter_cm)
    rx_diameter_m = cm_to_m(params.rx_diameter_cm)
    tx_inner_diameter_m = cm_to_m(params.tx_inner_diameter_cm)
    rx_offset_m = cm_to_m(params.rx_offset_cm)
    tracking_offset_rad = urad_to_rad(params.tracking_offset_urad)
    tx_beam_diameter_m = mm_to_m(params.tx_beam_diameter_mm) if params.tx_beam_diameter_mm else None

    distance_km = params.distance_m / 1000.0

    # === 빔 파라미터 계산 ===
    beam_diameter_at_mrr = calculate_beam_diameter_at_distance(
        divergence_half_rad, params.distance_m, tx_beam_diameter_m
    )
    beam_radius_at_mrr = beam_diameter_at_mrr / 2.0

    # MRR M² 계산 (입력 방식에 따라 m2_min 결정, 항상 angle-dependent)
    if params.use_strehl:
        # Strehl ratio에서 최소 M² 계산: M² = 1/√Strehl
        m2_min = 1.0 / np.sqrt(max(params.strehl_ratio, 0.01))
    else:
        m2_min = params.mrr_m2_min

    mrr_m2_value = mrr_m2(
        params.sigma_orientation_deg,
        m2_min, params.mrr_m2_max,
        params.mrr_knee_deg, params.mrr_max_deg
    )

    # 다운링크 발산각
    downlink_div_rad = downlink_divergence(wavelength_m, mrr_diameter_m, mrr_m2_value)

    # 다운링크 빔 크기 at GS
    beam_diameter_at_gs = calculate_beam_diameter_at_distance(
        downlink_div_rad, params.distance_m, mrr_diameter_m
    )
    beam_radius_at_gs = beam_diameter_at_gs / 2.0

    beam_params = BeamParameters(
        tx_beam_diameter_m=tx_beam_diameter_m,
        divergence_rad=divergence_full_rad,  # Full divergence for display
        beam_diameter_at_mrr_m=beam_diameter_at_mrr,
        beam_footprint_area_m2=calculate_beam_footprint_area(beam_diameter_at_mrr),
        mrr_m2_factor=mrr_m2_value,
        downlink_divergence_rad=downlink_div_rad,
        beam_diameter_at_gs_m=beam_diameter_at_gs
    )

    # === Uplink 채널 계수 계산 ===

    # 1. 기하 손실 h_geometric + 추적 오차 손실 h_tracking (분리 계산)
    mrr_area = np.pi * (mrr_diameter_m / 2.0) ** 2

    # Pointing displacement 계산 (지향 오차 각도 × 거리)
    d_p = tracking_offset_rad * params.distance_m
    d_px = d_p / np.sqrt(2)  # X/Y 균등 분배
    d_py = d_p / np.sqrt(2)

    # h_geometric: aperture/footprint ratio (pointing 미포함)
    # h_tracking: pointing error로 인한 손실
    # h_pu: h_geometric * h_tracking
    h_geometric, h_tracking, h_pu = uplink_geometric_coefficient_separate(
        mrr_area, beam_radius_at_mrr, d_px, d_py
    )

    # 2. 대기 감쇠 h_lgu
    h_lgu, _ = atmospheric_attenuation(
        params.wavelength_nm, params.visibility_km,
        distance_km, params.fog_condition
    )

    # 3. 신틸레이션 페이딩 h_aug
    if params.use_altitude_profile:
        cn2_func = lambda h: hufnagel_valley_cn2(h, params.wind_speed_ms)
        sigma_r2_up = calculate_rytov_variance_profile(
            wavelength_m, params.h_gs_m, params.h_uav_m,
            cn2_func, "uplink"
        )
    else:
        sigma_r2_up = calculate_rytov_variance_constant(
            wavelength_m, params.distance_m,
            params.cn2_constant, "spherical"
        )

    # 난류 손실 적용
    if params.use_turbulence:
        L_scint_up_dB = scintillation_loss_dB(sigma_r2_up, params.scint_probability, "spherical")
        h_aug = 10.0 ** (-L_scint_up_dB / 10.0)  # dB → linear
    else:
        L_scint_up_dB = 0.0
        h_aug = 1.0

    # 4. UAV 자세 오차 손실 h_orientation (Antenna Model L_orientation에 대응)
    # Uplink에서 MRR에 도달하는 전력에 영향
    L_orientation_dB = mrr_orientation_loss_dB(
        params.sigma_orientation_deg,
        params.mrr_knee_deg, params.mrr_max_deg
    )
    h_orientation = 10.0 ** (-L_orientation_dB / 10.0)  # dB → linear

    # === MRR 채널 계수 ===

    # 변조 효율 (MQW 파라미터 또는 직접 입력)
    if params.use_mqw_params:
        # M = exp(-α_off) * (C_MQW - 1)
        h_modulation = mqw_modulation_efficiency_from_contrast(params.c_mqw, params.alpha_off)
    else:
        h_modulation = params.modulation_efficiency

    # h_MRR = h_modulation (단순화: 변조 효율만)
    h_MRR = h_modulation

    # === Downlink 채널 계수 계산 ===

    # 1. 수광 계수 h_pg (beam profile + receiver config 통합)
    # Bistatic: offset 고려한 원형 aperture 수광 효율
    # Concentric: 환형 aperture 수광 효율
    rx_radius_m = rx_diameter_m / 2.0
    h_pg = downlink_collection_coefficient(
        beam_radius_m=beam_radius_at_gs,
        config=params.rx_config,
        rx_outer_radius_m=rx_radius_m,
        offset_m=rx_offset_m,
        tx_inner_radius_m=tx_inner_diameter_m / 2.0,
        profile=params.beam_profile
    )

    # 2. 대기 감쇠 h_lgu (downlink, same as uplink)
    h_lgu_down = h_lgu

    # 3. 신틸레이션 페이딩 h_agu
    if params.use_altitude_profile:
        sigma_r2_down = calculate_rytov_variance_profile(
            wavelength_m, params.h_gs_m, params.h_uav_m,
            cn2_func, "downlink"
        )
    else:
        sigma_r2_down = calculate_rytov_variance_constant(
            wavelength_m, params.distance_m,
            params.cn2_constant, "plane"
        )

    # 난류 손실 적용
    if params.use_turbulence:
        L_scint_down_dB = scintillation_loss_dB(sigma_r2_down, params.scint_probability, "plane")
        h_agu = 10.0 ** (-L_scint_down_dB / 10.0)  # dB → linear
    else:
        L_scint_down_dB = 0.0
        h_agu = 1.0

    # === 광학계 효율 ===
    eta_optics = params.eta_tx * params.eta_rx

    # === 전체 채널 계수 ===
    # Uplink: h_up = h_pu * h_lgu * h_aug * h_orientation (orientation을 Uplink에 포함)
    h_uplink = h_pu * h_lgu * h_aug * h_orientation

    # Downlink: h_down = h_pg * h_lgu_down * h_agu (h_pg에 rx_config 통합됨)
    h_downlink = h_pg * h_lgu_down * h_agu

    # Total: h = h_up * h_MRR * h_down * eta_optics
    # h_MRR = h_modulation (변조 효율만)
    h_total = h_uplink * h_MRR * h_downlink * eta_optics

    # === 수신 전력 ===
    P_rx_W = params.P_tx_W * h_total

    # === 결과 구성 ===
    result = OpticalModelResult(
        beam=beam_params,
        # 송신 광학계
        eta_tx=params.eta_tx,
        # Uplink 계수 (h_orientation 포함)
        h_geometric=h_geometric,
        h_tracking=h_tracking,
        h_pu=h_pu,
        h_lgu=h_lgu,
        h_aug=h_aug,
        h_orientation=h_orientation,  # Uplink에 포함
        sigma_r2_uplink=sigma_r2_up,
        L_scint_up_dB=L_scint_up_dB,
        h_uplink=h_uplink,
        # MRR 계수 (단순화: 변조만)
        h_modulation=h_modulation,
        h_MRR=h_MRR,
        # Downlink 계수 (h_pg에 rx_config 통합됨)
        h_pg=h_pg,
        h_lgu_down=h_lgu_down,
        h_agu=h_agu,
        sigma_r2_downlink=sigma_r2_down,
        L_scint_down_dB=L_scint_down_dB,
        h_downlink=h_downlink,
        # 수신 광학계 (분리)
        eta_rx=params.eta_rx,
        eta_optics=eta_optics,
        # 전체
        h_total=h_total,
        P_tx_W=params.P_tx_W,
        P_rx_W=P_rx_W
    )

    return result


def optical_to_antenna_comparison(
    optical_result: OpticalModelResult,
    receiver_sensitivity_dBm: float = -40.0
) -> dict:
    """
    Optical Model 결과를 Antenna Model (dB) 형식으로 변환하여 비교

    Parameters:
        optical_result: OpticalModelResult 인스턴스
        receiver_sensitivity_dBm: 수신 감도 [dBm]

    Returns:
        dict with dB-converted values and comparison metrics
    """
    # 채널 계수를 dB로 변환
    def to_dB(x):
        if x <= 0:
            return float('-inf')
        return 10.0 * np.log10(x)

    comparison = {
        # TX Optics (dB) - Antenna Model L_tx_optics에 대응
        "eta_tx_dB": to_dB(optical_result.eta_tx),

        # Uplink (dB) - h_orientation 포함
        "h_geometric_dB": to_dB(optical_result.h_geometric),
        "h_tracking_dB": to_dB(optical_result.h_tracking),  # Antenna Model L_tracking에 대응
        "h_pu_dB": to_dB(optical_result.h_pu),
        "h_lgu_dB": to_dB(optical_result.h_lgu),
        "h_aug_dB": to_dB(optical_result.h_aug),
        "h_orientation_dB": to_dB(optical_result.h_orientation),  # Antenna Model L_orientation에 대응
        "h_uplink_dB": to_dB(optical_result.h_uplink),

        # MRR (dB) - 변조만
        "h_modulation_dB": to_dB(optical_result.h_modulation),
        "h_MRR_dB": to_dB(optical_result.h_MRR),  # = h_modulation_dB

        # Downlink (dB) - h_pg에 rx_config 통합됨
        "h_pg_dB": to_dB(optical_result.h_pg),
        "h_lgu_down_dB": to_dB(optical_result.h_lgu_down),
        "h_agu_dB": to_dB(optical_result.h_agu),
        "h_downlink_dB": to_dB(optical_result.h_downlink),

        # RX Optics (dB) - Antenna Model L_rx_optics에 대응
        "eta_rx_dB": to_dB(optical_result.eta_rx),
        "eta_optics_dB": to_dB(optical_result.eta_optics),

        # Total
        "h_total_dB": to_dB(optical_result.h_total),

        # Power
        "P_tx_dBm": W_to_dBm(optical_result.P_tx_W),
        "P_rx_dBm": W_to_dBm(optical_result.P_rx_W),

        # Link margin
        "link_margin_dB": W_to_dBm(optical_result.P_rx_W) - receiver_sensitivity_dBm,

        # Rytov variances
        "sigma_r2_uplink": optical_result.sigma_r2_uplink,
        "sigma_r2_downlink": optical_result.sigma_r2_downlink
    }

    return comparison
