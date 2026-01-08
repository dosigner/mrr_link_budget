"""
Monte Carlo 시뮬레이션 엔진
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Literal

from ..models.common import dB_to_linear, linear_to_dB, W_to_dBm, dBm_to_W
from ..models.antenna_model import AntennaModelParams, calculate_antenna_link_budget
from ..models.optical_model import OpticalModelParams, calculate_optical_channel_coefficient
from ..atmosphere.turbulence import (
    sample_lognormal_fading, sample_gamma_gamma_fading,
    calculate_rytov_variance_constant, calculate_rytov_variance_profile,
    hufnagel_valley_cn2
)
from ..mrr.efficiency import sample_mrr_orientation_fluctuation
from ..geometry.pointing import sample_pointing_displacement


@dataclass
class MonteCarloConfig:
    """Monte Carlo 시뮬레이션 설정"""
    n_samples: int = 10000
    seed: Optional[int] = 42
    include_pointing: bool = True
    include_scintillation: bool = True
    include_orientation: bool = True


@dataclass
class MonteCarloResult:
    """Monte Carlo 시뮬레이션 결과"""
    n_samples: int

    # 채널 계수 샘플
    h_samples: np.ndarray           # 전체 채널 계수 샘플
    h_uplink_samples: np.ndarray    # Uplink 채널 샘플
    h_downlink_samples: np.ndarray  # Downlink 채널 샘플

    # 수신 전력 샘플
    P_rx_dBm_samples: np.ndarray

    # 통계
    h_mean: float
    h_std: float
    h_median: float
    h_percentile_1: float           # 1% percentile
    h_percentile_5: float           # 5% percentile
    h_percentile_10: float          # 10% percentile
    h_percentile_99: float          # 99% percentile

    P_rx_mean_dBm: float
    P_rx_std_dB: float

    # Outage
    outage_probability: float       # P(P_rx < sensitivity)

    # Rytov variance (참고용)
    sigma_r2_uplink: float
    sigma_r2_downlink: float

    def get_pdf_histogram(self, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """채널 계수 히스토그램 반환"""
        hist, bin_edges = np.histogram(self.h_samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def get_cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """채널 계수 CDF 반환"""
        sorted_h = np.sort(self.h_samples)
        cdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
        return sorted_h, cdf

    def get_P_rx_histogram(self, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """수신 전력 히스토그램 반환"""
        valid_samples = self.P_rx_dBm_samples[np.isfinite(self.P_rx_dBm_samples)]
        hist, bin_edges = np.histogram(valid_samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist


def run_monte_carlo_antenna(
    params: AntennaModelParams,
    config: MonteCarloConfig = MonteCarloConfig()
) -> MonteCarloResult:
    """
    Antenna Model 기반 Monte Carlo 시뮬레이션

    Parameters:
        params: AntennaModelParams 인스턴스
        config: MonteCarloConfig 설정

    Returns:
        MonteCarloResult 인스턴스
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # 기본 링크 버짓 계산 (deterministic)
    base_result = calculate_antenna_link_budget(params)

    # 단위 변환
    wavelength_m = params.wavelength_nm * 1e-9
    divergence_half_rad = params.theta_div_full_mrad * 1e-3 / 2.0  # Full → Half
    sigma_tracking_rad = params.tracking_offset_urad * 1e-6

    # Rytov variance 계산
    if params.use_altitude_profile:
        cn2_func = lambda h: hufnagel_valley_cn2(h, params.wind_speed_ms)
        sigma_r2_up = calculate_rytov_variance_profile(
            wavelength_m, params.h_gs_m, params.h_uav_m,
            cn2_func, "uplink"
        )
        sigma_r2_down = calculate_rytov_variance_profile(
            wavelength_m, params.h_gs_m, params.h_uav_m,
            cn2_func, "downlink"
        )
    else:
        sigma_r2_up = calculate_rytov_variance_constant(
            wavelength_m, params.distance_m,
            params.cn2_constant, "spherical"
        )
        sigma_r2_down = calculate_rytov_variance_constant(
            wavelength_m, params.distance_m,
            params.cn2_constant, "plane"
        )

    # === 페이딩 샘플 생성 ===

    # 1. 신틸레이션 페이딩 (uplink - spherical wave)
    if config.include_scintillation and sigma_r2_up > 0:
        if sigma_r2_up < 1.0:
            h_scint_up = sample_lognormal_fading(sigma_r2_up, n, rng)
        else:
            h_scint_up = sample_gamma_gamma_fading(sigma_r2_up, n, "spherical", rng)
    else:
        h_scint_up = np.ones(n)

    # 2. 신틸레이션 페이딩 (downlink - plane wave)
    if config.include_scintillation and sigma_r2_down > 0:
        if sigma_r2_down < 1.0:
            h_scint_down = sample_lognormal_fading(sigma_r2_down, n, rng)
        else:
            h_scint_down = sample_gamma_gamma_fading(sigma_r2_down, n, "plane", rng)
    else:
        h_scint_down = np.ones(n)

    # 3. Pointing error 샘플링
    if config.include_pointing:
        d_p_samples = sample_pointing_displacement(
            sigma_tracking_rad, params.distance_m, n, rng
        )
        beam_radius_at_mrr = base_result.beam.beam_diameter_at_mrr_m / 2.0
        # Gaussian pointing loss
        h_pointing = np.exp(-2.0 * d_p_samples**2 / beam_radius_at_mrr**2)
    else:
        h_pointing = np.ones(n)

    # 4. MRR orientation fluctuation
    if config.include_orientation:
        h_mrr_samples = sample_mrr_orientation_fluctuation(
            params.sigma_orientation_deg, n, rng
        )
    else:
        h_mrr_samples = np.ones(n)

    # === 채널 계수 결합 ===

    # Uplink: 신틸레이션 + pointing + orientation
    h_uplink_samples = h_scint_up * h_pointing * h_mrr_samples

    # Downlink: 신틸레이션
    h_downlink_samples = h_scint_down

    # Total channel coefficient (상대적 변동)
    h_samples = h_uplink_samples * h_downlink_samples

    # === 수신 전력 계산 ===

    # 기본 수신 전력 (deterministic)
    P_rx_base_dBm = base_result.downlink.P_rx_dBm

    # Scintillation loss는 이미 deterministic 계산에서 평균값 사용
    # MC에서는 실제 페이딩 적용
    L_scint_avg_dB = base_result.uplink.L_scint_dB + base_result.downlink.L_scint_dB

    # MC 샘플에서의 수신 전력
    # P_rx = P_rx_base + L_scint_avg (보상) + 10*log10(h_samples)
    h_samples_dB = np.where(h_samples > 0, 10.0 * np.log10(h_samples), -np.inf)
    P_rx_dBm_samples = P_rx_base_dBm + L_scint_avg_dB + h_samples_dB

    # === 통계 계산 ===

    valid_h = h_samples[h_samples > 0]
    valid_P_rx = P_rx_dBm_samples[np.isfinite(P_rx_dBm_samples)]

    h_mean = np.mean(valid_h) if len(valid_h) > 0 else 0.0
    h_std = np.std(valid_h) if len(valid_h) > 0 else 0.0
    h_median = np.median(valid_h) if len(valid_h) > 0 else 0.0
    h_percentile_1 = np.percentile(valid_h, 1) if len(valid_h) > 0 else 0.0
    h_percentile_5 = np.percentile(valid_h, 5) if len(valid_h) > 0 else 0.0
    h_percentile_10 = np.percentile(valid_h, 10) if len(valid_h) > 0 else 0.0
    h_percentile_99 = np.percentile(valid_h, 99) if len(valid_h) > 0 else 0.0

    P_rx_mean_dBm = np.mean(valid_P_rx) if len(valid_P_rx) > 0 else -np.inf
    P_rx_std_dB = np.std(valid_P_rx) if len(valid_P_rx) > 0 else 0.0

    # Outage probability
    outage_count = np.sum(P_rx_dBm_samples < params.receiver_sensitivity_dBm)
    outage_probability = outage_count / n

    return MonteCarloResult(
        n_samples=n,
        h_samples=h_samples,
        h_uplink_samples=h_uplink_samples,
        h_downlink_samples=h_downlink_samples,
        P_rx_dBm_samples=P_rx_dBm_samples,
        h_mean=h_mean,
        h_std=h_std,
        h_median=h_median,
        h_percentile_1=h_percentile_1,
        h_percentile_5=h_percentile_5,
        h_percentile_10=h_percentile_10,
        h_percentile_99=h_percentile_99,
        P_rx_mean_dBm=P_rx_mean_dBm,
        P_rx_std_dB=P_rx_std_dB,
        outage_probability=outage_probability,
        sigma_r2_uplink=sigma_r2_up,
        sigma_r2_downlink=sigma_r2_down
    )


def run_monte_carlo_optical(
    params: OpticalModelParams,
    config: MonteCarloConfig = MonteCarloConfig(),
    receiver_sensitivity_dBm: float = -40.0
) -> MonteCarloResult:
    """
    Optical Model 기반 Monte Carlo 시뮬레이션

    Parameters:
        params: OpticalModelParams 인스턴스
        config: MonteCarloConfig 설정
        receiver_sensitivity_dBm: 수신 감도 [dBm]

    Returns:
        MonteCarloResult 인스턴스
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # 기본 채널 계수 계산 (deterministic)
    base_result = calculate_optical_channel_coefficient(params)

    # 단위 변환
    wavelength_m = params.wavelength_nm * 1e-9
    divergence_half_rad = params.theta_div_full_mrad * 1e-3 / 2.0  # Full → Half
    sigma_tracking_rad = params.tracking_offset_urad * 1e-6

    sigma_r2_up = base_result.sigma_r2_uplink
    sigma_r2_down = base_result.sigma_r2_downlink

    # === 페이딩 샘플 생성 ===

    # 1. 신틸레이션 페이딩 (uplink)
    if config.include_scintillation and sigma_r2_up > 0:
        if sigma_r2_up < 1.0:
            h_scint_up = sample_lognormal_fading(sigma_r2_up, n, rng)
        else:
            h_scint_up = sample_gamma_gamma_fading(sigma_r2_up, n, "spherical", rng)
    else:
        h_scint_up = np.ones(n)

    # 2. 신틸레이션 페이딩 (downlink)
    if config.include_scintillation and sigma_r2_down > 0:
        if sigma_r2_down < 1.0:
            h_scint_down = sample_lognormal_fading(sigma_r2_down, n, rng)
        else:
            h_scint_down = sample_gamma_gamma_fading(sigma_r2_down, n, "plane", rng)
    else:
        h_scint_down = np.ones(n)

    # 3. Pointing error 샘플링
    if config.include_pointing:
        d_p_samples = sample_pointing_displacement(
            sigma_tracking_rad, params.distance_m, n, rng
        )
        beam_radius_at_mrr = base_result.beam.beam_diameter_at_mrr_m / 2.0
        h_pointing = np.exp(-2.0 * d_p_samples**2 / beam_radius_at_mrr**2)
    else:
        h_pointing = np.ones(n)

    # 4. MRR orientation fluctuation
    if config.include_orientation:
        h_mrr_samples = sample_mrr_orientation_fluctuation(
            params.sigma_orientation_deg, n, rng
        )
    else:
        h_mrr_samples = np.ones(n)

    # === 채널 계수 결합 ===

    h_uplink_samples = h_scint_up * h_pointing * h_mrr_samples
    h_downlink_samples = h_scint_down

    # 기본 채널 계수에 페이딩 적용
    # base_result.h_total 에는 h_aug=1, h_agu=1 이므로 scintillation 제외됨
    h_samples = h_uplink_samples * h_downlink_samples

    # 전체 채널 계수 (deterministic * stochastic)
    h_total_samples = base_result.h_total * h_samples

    # === 수신 전력 계산 ===

    P_rx_W_samples = params.P_tx_W * h_total_samples
    P_rx_dBm_samples = np.where(
        P_rx_W_samples > 0,
        10.0 * np.log10(P_rx_W_samples * 1000),
        -np.inf
    )

    # === 통계 계산 ===

    valid_h = h_total_samples[h_total_samples > 0]
    valid_P_rx = P_rx_dBm_samples[np.isfinite(P_rx_dBm_samples)]

    h_mean = np.mean(valid_h) if len(valid_h) > 0 else 0.0
    h_std = np.std(valid_h) if len(valid_h) > 0 else 0.0
    h_median = np.median(valid_h) if len(valid_h) > 0 else 0.0
    h_percentile_1 = np.percentile(valid_h, 1) if len(valid_h) > 0 else 0.0
    h_percentile_5 = np.percentile(valid_h, 5) if len(valid_h) > 0 else 0.0
    h_percentile_10 = np.percentile(valid_h, 10) if len(valid_h) > 0 else 0.0
    h_percentile_99 = np.percentile(valid_h, 99) if len(valid_h) > 0 else 0.0

    P_rx_mean_dBm = np.mean(valid_P_rx) if len(valid_P_rx) > 0 else -np.inf
    P_rx_std_dB = np.std(valid_P_rx) if len(valid_P_rx) > 0 else 0.0

    # Outage probability
    outage_count = np.sum(P_rx_dBm_samples < receiver_sensitivity_dBm)
    outage_probability = outage_count / n

    return MonteCarloResult(
        n_samples=n,
        h_samples=h_total_samples,
        h_uplink_samples=h_uplink_samples,
        h_downlink_samples=h_downlink_samples,
        P_rx_dBm_samples=P_rx_dBm_samples,
        h_mean=h_mean,
        h_std=h_std,
        h_median=h_median,
        h_percentile_1=h_percentile_1,
        h_percentile_5=h_percentile_5,
        h_percentile_10=h_percentile_10,
        h_percentile_99=h_percentile_99,
        P_rx_mean_dBm=P_rx_mean_dBm,
        P_rx_std_dB=P_rx_std_dB,
        outage_probability=outage_probability,
        sigma_r2_uplink=sigma_r2_up,
        sigma_r2_downlink=sigma_r2_down
    )


def calculate_ber_ook(snr_linear: np.ndarray) -> np.ndarray:
    """
    OOK (On-Off Keying) BER 계산

    BER = 0.5 * erfc(sqrt(SNR/2))

    Parameters:
        snr_linear: SNR (선형)

    Returns:
        BER 값
    """
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(snr_linear / 2.0))


def calculate_ber_bpsk(snr_linear: np.ndarray) -> np.ndarray:
    """
    BPSK BER 계산

    BER = 0.5 * erfc(sqrt(SNR))

    Parameters:
        snr_linear: SNR (선형)

    Returns:
        BER 값
    """
    from scipy.special import erfc
    return 0.5 * erfc(np.sqrt(snr_linear))


def calculate_required_fade_margin(
    target_outage: float,
    sigma_r2_up: float,
    sigma_r2_down: float,
    n_samples: int = 100000
) -> float:
    """
    목표 outage probability 달성을 위한 페이드 마진 계산

    Parameters:
        target_outage: 목표 outage probability (예: 0.01 = 1%)
        sigma_r2_up: Uplink Rytov variance
        sigma_r2_down: Downlink Rytov variance
        n_samples: 샘플 수

    Returns:
        필요한 페이드 마진 [dB]
    """
    rng = np.random.default_rng(42)

    # 페이딩 샘플 생성
    if sigma_r2_up < 1.0:
        h_up = sample_lognormal_fading(sigma_r2_up, n_samples, rng)
    else:
        h_up = sample_gamma_gamma_fading(sigma_r2_up, n_samples, "spherical", rng)

    if sigma_r2_down < 1.0:
        h_down = sample_lognormal_fading(sigma_r2_down, n_samples, rng)
    else:
        h_down = sample_gamma_gamma_fading(sigma_r2_down, n_samples, "plane", rng)

    h_total = h_up * h_down

    # target_outage percentile
    h_threshold = np.percentile(h_total, target_outage * 100)

    # 필요한 마진 = -10*log10(h_threshold)
    if h_threshold > 0:
        margin_dB = -10.0 * np.log10(h_threshold)
    else:
        margin_dB = float('inf')

    return margin_dB
