"""
Parameter Sweep 분석
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable
from copy import deepcopy

from ..models.antenna_model import AntennaModelParams, calculate_antenna_link_budget
from ..models.optical_model import OpticalModelParams, calculate_optical_channel_coefficient
from .monte_carlo import MonteCarloConfig, run_monte_carlo_antenna, run_monte_carlo_optical


@dataclass
class SweepResult:
    """Parameter Sweep 결과"""
    parameter_name: str
    parameter_values: np.ndarray
    parameter_unit: str

    # 결과 배열
    link_margin_dB: np.ndarray
    P_rx_dBm: np.ndarray

    # MC 결과 (optional)
    outage_probability: Optional[np.ndarray] = None
    h_mean: Optional[np.ndarray] = None
    h_percentile_1: Optional[np.ndarray] = None
    h_percentile_5: Optional[np.ndarray] = None

    def get_crossing_value(self, metric: str, threshold: float) -> Optional[float]:
        """
        특정 메트릭이 threshold를 교차하는 파라미터 값 찾기

        Parameters:
            metric: "link_margin_dB", "outage_probability" 등
            threshold: 임계값

        Returns:
            교차하는 파라미터 값 (없으면 None)
        """
        values = getattr(self, metric, None)
        if values is None:
            return None

        # 교차점 찾기 (선형 보간)
        for i in range(len(values) - 1):
            if (values[i] - threshold) * (values[i + 1] - threshold) < 0:
                # 선형 보간
                x1, x2 = self.parameter_values[i], self.parameter_values[i + 1]
                y1, y2 = values[i], values[i + 1]
                x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
                return x_cross

        return None


def sweep_distance(
    base_params: AntennaModelParams,
    distances_m: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    거리에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        distances_m: 거리 배열 [m]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(distances_m)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, dist in enumerate(distances_m):
        params = deepcopy(base_params)
        params.distance_m = dist

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Distance",
        parameter_values=distances_m,
        parameter_unit="m",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_visibility(
    base_params: AntennaModelParams,
    visibilities_km: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    가시거리에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        visibilities_km: 가시거리 배열 [km]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(visibilities_km)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, vis in enumerate(visibilities_km):
        params = deepcopy(base_params)
        params.visibility_km = vis

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Visibility",
        parameter_values=visibilities_km,
        parameter_unit="km",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_orientation_error(
    base_params: AntennaModelParams,
    orientation_errors_deg: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    UAV 자세 오차에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        orientation_errors_deg: 자세 오차 배열 [deg]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(orientation_errors_deg)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, sigma_deg in enumerate(orientation_errors_deg):
        params = deepcopy(base_params)
        params.sigma_orientation_deg = sigma_deg

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Orientation Error σ",
        parameter_values=orientation_errors_deg,
        parameter_unit="deg",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_tracking_error(
    base_params: AntennaModelParams,
    tracking_errors_urad: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    추적 오차에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        tracking_errors_urad: 추적 오차 배열 [μrad]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(tracking_errors_urad)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, sigma_urad in enumerate(tracking_errors_urad):
        params = deepcopy(base_params)
        params.tracking_offset_urad = sigma_urad

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Tracking Error σ",
        parameter_values=tracking_errors_urad,
        parameter_unit="μrad",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_mrr_diameter(
    base_params: AntennaModelParams,
    mrr_diameters_cm: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    MRR 직경에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        mrr_diameters_cm: MRR 직경 배열 [cm]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(mrr_diameters_cm)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, d_cm in enumerate(mrr_diameters_cm):
        params = deepcopy(base_params)
        params.mrr_diameter_cm = d_cm

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="MRR Diameter",
        parameter_values=mrr_diameters_cm,
        parameter_unit="cm",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_tx_power(
    base_params: AntennaModelParams,
    tx_powers_dBm: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    송신 전력에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        tx_powers_dBm: 송신 전력 배열 [dBm]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(tx_powers_dBm)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, p_dBm in enumerate(tx_powers_dBm):
        params = deepcopy(base_params)
        params.P_tx_dBm = p_dBm

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="TX Power",
        parameter_values=tx_powers_dBm,
        parameter_unit="dBm",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_divergence(
    base_params: AntennaModelParams,
    divergences_mrad: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    발산각에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        divergences_mrad: 발산각 배열 [mrad]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(divergences_mrad)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, div_mrad in enumerate(divergences_mrad):
        params = deepcopy(base_params)
        params.theta_div_full_mrad = div_mrad

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Divergence θ",
        parameter_values=divergences_mrad,
        parameter_unit="mrad",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_cn2(
    base_params: AntennaModelParams,
    cn2_values: np.ndarray,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    Cn² 값에 따른 링크 버짓 스윕

    Parameters:
        base_params: 기본 파라미터
        cn2_values: Cn² 배열 [m^(-2/3)]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(cn2_values)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, cn2 in enumerate(cn2_values):
        params = deepcopy(base_params)
        params.cn2_constant = cn2
        params.use_altitude_profile = False

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name="Cn²",
        parameter_values=cn2_values,
        parameter_unit="m^(-2/3)",
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_generic(
    base_params: AntennaModelParams,
    parameter_name: str,
    parameter_values: np.ndarray,
    parameter_unit: str,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    일반 파라미터 스윕

    Parameters:
        base_params: 기본 파라미터
        parameter_name: 파라미터 속성 이름 (AntennaModelParams의 필드명)
        parameter_values: 파라미터 값 배열
        parameter_unit: 단위 문자열
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    if not hasattr(base_params, parameter_name):
        raise ValueError(f"Unknown parameter: {parameter_name}")

    n = len(parameter_values)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, val in enumerate(parameter_values):
        params = deepcopy(base_params)
        setattr(params, parameter_name, val)

        result = calculate_antenna_link_budget(params)
        link_margin[i] = result.link_margin_dB
        P_rx[i] = result.receiver_power_dBm

        if run_mc:
            mc_result = run_monte_carlo_antenna(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        parameter_unit=parameter_unit,
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_2d(
    base_params: AntennaModelParams,
    param1_name: str,
    param1_values: np.ndarray,
    param2_name: str,
    param2_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 파라미터 스윕

    Parameters:
        base_params: 기본 파라미터
        param1_name: 첫 번째 파라미터 이름
        param1_values: 첫 번째 파라미터 값 배열
        param2_name: 두 번째 파라미터 이름
        param2_values: 두 번째 파라미터 값 배열

    Returns:
        (param1_grid, param2_grid, link_margin_grid)
    """
    n1 = len(param1_values)
    n2 = len(param2_values)
    link_margin = np.zeros((n2, n1))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            params = deepcopy(base_params)
            setattr(params, param1_name, v1)
            setattr(params, param2_name, v2)

            result = calculate_antenna_link_budget(params)
            link_margin[j, i] = result.link_margin_dB

    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)

    return param1_grid, param2_grid, link_margin


# =============================================================================
# Optical Model Sweep Functions
# =============================================================================

from ..models.optical_model import optical_to_antenna_comparison


def _optical_sweep_core(
    base_params: OpticalModelParams,
    parameter_name: str,
    parameter_values: np.ndarray,
    parameter_unit: str,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """
    Optical Model sweep 공통 로직

    Parameters:
        base_params: 기본 파라미터
        parameter_name: 변경할 파라미터 이름
        parameter_values: 파라미터 값 배열
        parameter_unit: 단위 문자열
        receiver_sensitivity_dBm: 수신기 감도 [dBm]
        run_mc: Monte Carlo 실행 여부
        mc_config: MC 설정

    Returns:
        SweepResult
    """
    n = len(parameter_values)
    link_margin = np.zeros(n)
    P_rx = np.zeros(n)
    outage = np.zeros(n) if run_mc else None
    h_mean = np.zeros(n) if run_mc else None
    h_p1 = np.zeros(n) if run_mc else None
    h_p5 = np.zeros(n) if run_mc else None

    for i, val in enumerate(parameter_values):
        params = deepcopy(base_params)
        setattr(params, parameter_name, val)

        result = calculate_optical_channel_coefficient(params)
        comparison = optical_to_antenna_comparison(result, receiver_sensitivity_dBm)
        link_margin[i] = comparison['link_margin_dB']
        P_rx[i] = comparison['P_rx_dBm']

        if run_mc:
            mc_result = run_monte_carlo_optical(params, mc_config or MonteCarloConfig())
            outage[i] = mc_result.outage_probability
            h_mean[i] = mc_result.h_mean
            h_p1[i] = mc_result.h_percentile_1
            h_p5[i] = mc_result.h_percentile_5

    return SweepResult(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        parameter_unit=parameter_unit,
        link_margin_dB=link_margin,
        P_rx_dBm=P_rx,
        outage_probability=outage,
        h_mean=h_mean,
        h_percentile_1=h_p1,
        h_percentile_5=h_p5
    )


def sweep_distance_optical(
    base_params: OpticalModelParams,
    distances_m: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """거리에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "distance_m", distances_m, "m",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "Distance"
    return result


def sweep_visibility_optical(
    base_params: OpticalModelParams,
    visibilities_km: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """가시거리에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "visibility_km", visibilities_km, "km",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "Visibility"
    return result


def sweep_orientation_error_optical(
    base_params: OpticalModelParams,
    orientation_errors_deg: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """UAV 자세 오차에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "sigma_orientation_deg", orientation_errors_deg, "deg",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "Orientation Error σ"
    return result


def sweep_tracking_error_optical(
    base_params: OpticalModelParams,
    tracking_errors_urad: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """추적 오차에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "tracking_offset_urad", tracking_errors_urad, "μrad",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "Tracking Error σ"
    return result


def sweep_mrr_diameter_optical(
    base_params: OpticalModelParams,
    mrr_diameters_cm: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """MRR 직경에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "mrr_diameter_cm", mrr_diameters_cm, "cm",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "MRR Diameter"
    return result


def sweep_divergence_optical(
    base_params: OpticalModelParams,
    divergences_mrad: np.ndarray,
    receiver_sensitivity_dBm: float,
    run_mc: bool = False,
    mc_config: Optional[MonteCarloConfig] = None
) -> SweepResult:
    """발산각에 따른 Optical Model 스윕"""
    result = _optical_sweep_core(
        base_params, "theta_div_full_mrad", divergences_mrad, "mrad",
        receiver_sensitivity_dBm, run_mc, mc_config
    )
    result.parameter_name = "Divergence θ"
    return result
