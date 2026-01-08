"""
Antenna Model (dB 기반 링크 버짓 계산)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

from ..models.common import (
    nm_to_m, cm_to_m, mm_to_m, mrad_to_rad, deg_to_rad, urad_to_rad,
    linear_to_dB, dB_to_linear, dBm_to_W, W_to_dBm,
    free_space_loss, aperture_gain, transmitter_gain, wave_number
)
from ..models.results import (
    BeamParameters, UplinkBudget, DownlinkBudget, LinkBudgetResult
)
from ..atmosphere.attenuation import atmospheric_attenuation
from ..atmosphere.turbulence import (
    calculate_rytov_variance_constant, hufnagel_valley_cn2,
    calculate_rytov_variance_profile, scintillation_loss_dB
)
from ..mrr.efficiency import eta_mrr, mrr_m2, mrr_orientation_loss_dB
from ..mrr.modulation import modulation_loss_dB, mrr_passive_loss_dB, mqw_modulation_efficiency_from_contrast
from ..geometry.pointing import (
    calculate_beam_diameter_at_distance, calculate_beam_radius_at_distance,
    gaussian_pointing_loss, downlink_divergence, calculate_beam_footprint_area
)
from ..geometry.receiver import receiver_config_loss


@dataclass
class AntennaModelParams:
    """Antenna Model 입력 파라미터"""
    # 송신부
    P_tx_dBm: float = 30.0              # 송신 전력 [dBm]
    L_tx_optics_dB: float = 2.0         # 송신 광학계 손실 [dB]
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
    mrr_m2_min: float = 1.3             # M² 최소값 (angle_dependent 모드)
    mrr_m2_max: float = 3.4             # M² 최대값 (angle_dependent 모드)

    # M² 입력 방식 선택
    use_strehl: bool = False            # True: Strehl ratio로 M² 계산 / False: M² 직접 입력
    strehl_ratio: float = 0.8           # Strehl ratio → M² = 1/√Strehl

    # MQW 변조 파라미터
    use_mqw_params: bool = False        # True: α_off, C_MQW 사용 / False: modulation_efficiency 직접 사용
    alpha_off: float = 0.1              # OFF 상태 흡수 계수 α_off
    c_mqw: float = 3.0                  # Contrast ratio C_MQW (T_on/T_off)
    modulation_efficiency: float = 0.5  # 변조 효율 M (0~1) - use_mqw_params=False일 때 사용

    mrr_ar_coating_loss_dB: float = 0.5 # AR 코팅 손실 [dB]
    mrr_passive_loss_dB: float = 0.3    # MRR passive 손실 [dB]

    # 수신부
    rx_diameter_cm: float = 16.0        # GS 수신 직경 [cm]
    L_rx_optics_dB: float = 3.0         # 수신 광학계 손실 [dB]
    rx_config: Literal["bistatic", "concentric"] = "bistatic"
    rx_offset_m: float = 0.1            # Bistatic offset [m]
    tx_inner_diameter_cm: float = 5.0   # Concentric TX 내경 [cm]
    beam_profile: Literal["uniform", "gaussian", "airy"] = "uniform"  # 다운링크 빔 프로파일
    receiver_sensitivity_dBm: float = -40.0  # 수신 감도 [dBm]

    # 추적 (지향 오차)
    tracking_offset_urad: float = 100.0  # 지향 오차 각도 [μrad]

    # 난류 (Turbulence)
    use_turbulence: bool = True          # 난류 손실 적용 여부
    use_altitude_profile: bool = False  # 고도 프로파일 사용 여부
    cn2_constant: float = 5e-15         # 상수 Cn² [m^(-2/3)]
    h_gs_m: float = 0.0                 # GS 고도 [m]
    h_uav_m: float = 100.0              # UAV 고도 [m]
    wind_speed_ms: float = 21.0         # 풍속 [m/s]
    scint_probability: float = 0.99     # 신틸레이션 확률 (마진 계산용)


def calculate_antenna_link_budget(params: AntennaModelParams) -> LinkBudgetResult:
    """
    Antenna Model 링크 버짓 계산

    Parameters:
        params: AntennaModelParams 인스턴스

    Returns:
        LinkBudgetResult 인스턴스
    """
    # === 단위 변환 ===
    wavelength_m = nm_to_m(params.wavelength_nm)
    # Full divergence → Half-angle (내부 계산용)
    divergence_half_rad = mrad_to_rad(params.theta_div_full_mrad) / 2.0
    divergence_full_rad = mrad_to_rad(params.theta_div_full_mrad)
    mrr_diameter_m = cm_to_m(params.mrr_diameter_cm)
    rx_diameter_m = cm_to_m(params.rx_diameter_cm)
    tx_inner_diameter_m = cm_to_m(params.tx_inner_diameter_cm)
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

    # === Uplink 계산 ===

    # 송신기 이득 (half-angle 사용)
    _, G_tx_dB = transmitter_gain(divergence_half_rad)

    # 자유 공간 손실
    _, fsl_dB = free_space_loss(wavelength_m, params.distance_m)

    # 대기 감쇠
    _, L_atm_dB = atmospheric_attenuation(
        params.wavelength_nm, params.visibility_km,
        distance_km, params.fog_condition
    )

    # 신틸레이션 손실 (Uplink - spherical wave)
    if params.use_altitude_profile:
        # 고도 프로파일 기반
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
    if params.use_turbulence:
        L_scint_up_dB = scintillation_loss_dB(sigma_r2_up, params.scint_probability, "spherical")
    else:
        L_scint_up_dB = 0.0

    # 추적 오차 손실 (지향 오차 각도 × 거리 = offset)
    d_p_tracking = tracking_offset_rad * params.distance_m
    _, L_tracking_dB = gaussian_pointing_loss(d_p_tracking, beam_radius_at_mrr)

    # UAV 자세 오차 손실 (eta_mrr 기반)
    L_orientation_dB = mrr_orientation_loss_dB(
        params.sigma_orientation_deg,
        params.mrr_knee_deg, params.mrr_max_deg
    )

    # MRR 수신 이득
    mrr_area = np.pi * (mrr_diameter_m / 2.0) ** 2
    _, G_rx_mrr_dB = aperture_gain(mrr_diameter_m, wavelength_m)

    uplink = UplinkBudget(
        P_tx_dBm=params.P_tx_dBm,
        L_tx_optics_dB=params.L_tx_optics_dB,
        G_tx_dB=G_tx_dB,
        L_FSL_dB=fsl_dB,  # 이미 음수
        L_atm_dB=L_atm_dB,
        L_scint_dB=L_scint_up_dB,
        L_tracking_dB=L_tracking_dB,
        L_orientation_dB=L_orientation_dB,
        L_AR_coating_dB=params.mrr_ar_coating_loss_dB,
        G_rx_mrr_dB=G_rx_mrr_dB
    )

    # === Downlink 계산 ===

    # MRR 반사 이득 (retro-reflection with M² degradation)
    # G_mrr_rx = (πD/λ)² / M²² = G_rx_mrr / M²²
    # G_mrr_rx_dB = G_rx_mrr_dB - 20*log10(M²)
    _, G_mrr_rx_base_dB = aperture_gain(mrr_diameter_m, wavelength_m)
    G_mrr_rx_dB = G_mrr_rx_base_dB - 20.0 * np.log10(mrr_m2_value)

    # 변조 손실 (MQW 파라미터 또는 직접 입력)
    if params.use_mqw_params:
        # M = exp(-α_off) * (C_MQW - 1)
        mod_eff = mqw_modulation_efficiency_from_contrast(params.c_mqw, params.alpha_off)
    else:
        mod_eff = params.modulation_efficiency
    L_modulation_dB = modulation_loss_dB(mod_eff)

    # MRR passive 손실
    L_mrr_passive_dB = params.mrr_passive_loss_dB

    # 다운링크 FSL (같은 거리)
    _, fsl_down_dB = free_space_loss(wavelength_m, params.distance_m)

    # 다운링크 대기 감쇠 (같음)
    L_atm_down_dB = L_atm_dB

    # 신틸레이션 손실 (Downlink - plane wave)
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
    if params.use_turbulence:
        L_scint_down_dB = scintillation_loss_dB(sigma_r2_down, params.scint_probability, "plane")
    else:
        L_scint_down_dB = 0.0

    # GS 수신 이득
    _, G_rx_gs_dB = aperture_gain(rx_diameter_m, wavelength_m)

    # 수신부 구조 손실
    _, L_rx_config_dB = receiver_config_loss(
        beam_radius_at_gs,
        params.rx_config,
        rx_diameter_m / 2.0,
        params.rx_offset_m,
        tx_inner_diameter_m / 2.0,
        params.beam_profile
    )

    downlink = DownlinkBudget(
        P_mrr_in_dBm=uplink.P_mrr_in_dBm,
        G_mrr_rx_dB=G_mrr_rx_dB,
        L_modulation_dB=L_modulation_dB,
        L_mrr_passive_dB=L_mrr_passive_dB,
        L_FSL_dB=fsl_down_dB,
        L_atm_dB=L_atm_down_dB,
        L_scint_dB=L_scint_down_dB,
        G_rx_gs_dB=G_rx_gs_dB,
        L_rx_config_dB=L_rx_config_dB,
        L_rx_optics_dB=params.L_rx_optics_dB
    )

    # === 최종 결과 ===
    result = LinkBudgetResult(
        beam=beam_params,
        uplink=uplink,
        downlink=downlink,
        receiver_sensitivity_dBm=params.receiver_sensitivity_dBm
    )

    return result
