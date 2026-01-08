"""
수신기 구조 손실 계산 (Bi-static / Concentric)

빔 프로파일:
- Uniform (Top-hat): 균일 강도
- Gaussian: 중심 집중, exp(-2r²/w²)
- Airy Disk: 회절 한계, [2*J₁(x)/x]²
"""
import numpy as np
from scipy import integrate
from scipy.special import j1  # Bessel function of the first kind, order 1
from typing import Literal


def gaussian_beam_power_in_circle(beam_radius_m: float,
                                   aperture_radius_m: float,
                                   offset_m: float = 0.0) -> float:
    """
    Gaussian 빔에서 원형 aperture가 수집하는 전력 비율

    빔 중심에서 offset 만큼 떨어진 원형 aperture

    Parameters:
        beam_radius_m: Gaussian 빔 반경 (1/e² 기준) [m]
        aperture_radius_m: 수신 aperture 반경 [m]
        offset_m: 빔 중심과 aperture 중심 간 거리 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    if offset_m == 0:
        # 중심 정렬: 해석적 해
        # P/P_total = 1 - exp(-2 * r_ap² / w²)
        exponent = -2.0 * (aperture_radius_m ** 2) / (beam_radius_m ** 2)
        return 1.0 - np.exp(exponent)
    else:
        # Offset 있는 경우: 수치 적분
        w = beam_radius_m
        r_ap = aperture_radius_m
        d = offset_m

        def integrand(r, theta):
            # 극좌표에서 빔 중심 기준 좌표
            x = r * np.cos(theta) + d
            y = r * np.sin(theta)
            r_from_beam = np.sqrt(x ** 2 + y ** 2)
            intensity = (2 / (np.pi * w ** 2)) * np.exp(-2 * r_from_beam ** 2 / w ** 2)
            return intensity * r  # r for polar integration

        result, _ = integrate.dblquad(
            integrand,
            0, 2 * np.pi,  # theta
            lambda theta: 0, lambda theta: r_ap  # r
        )

        return result


# =============================================================================
# Uniform (Top-hat) Beam Profile
# =============================================================================

def uniform_beam_power_in_circle(beam_radius_m: float,
                                  aperture_radius_m: float,
                                  offset_m: float = 0.0) -> float:
    """
    Uniform (Top-hat) 빔에서 원형 aperture가 수집하는 전력 비율

    Parameters:
        beam_radius_m: 빔 반경 [m]
        aperture_radius_m: 수신 aperture 반경 [m]
        offset_m: 빔 중심과 aperture 중심 간 거리 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    # 빔 전체 면적
    beam_area = np.pi * beam_radius_m ** 2

    # Overlap 면적 계산
    overlap = circle_overlap_area(beam_radius_m, aperture_radius_m, offset_m)

    # 수집 비율 (aperture가 빔보다 클 수 있으므로 min으로 제한)
    return min(overlap / beam_area, 1.0) if beam_area > 0 else 0.0


# =============================================================================
# Airy Disk Beam Profile
# =============================================================================

def airy_intensity(r: float, airy_radius_m: float) -> float:
    """
    Airy disk 강도 프로파일

    I(r) = [2*J₁(π*r/r_airy) / (π*r/r_airy)]²

    Parameters:
        r: 빔 중심으로부터 거리 [m]
        airy_radius_m: Airy disk 반경 (첫 번째 영점) [m]

    Returns:
        정규화된 강도 (중심에서 1.0)
    """
    if r == 0:
        return 1.0

    x = np.pi * r / airy_radius_m
    # Airy pattern: [2*J1(x)/x]²
    return (2.0 * j1(x) / x) ** 2


def airy_beam_power_in_circle(airy_radius_m: float,
                               aperture_radius_m: float,
                               offset_m: float = 0.0) -> float:
    """
    Airy disk 빔에서 원형 aperture가 수집하는 전력 비율

    Parameters:
        airy_radius_m: Airy disk 반경 (첫 번째 영점, ~1.22λz/D) [m]
        aperture_radius_m: 수신 aperture 반경 [m]
        offset_m: 빔 중심과 aperture 중심 간 거리 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    # 전체 전력 (무한대까지 적분, 실제로는 큰 반경까지)
    # Airy disk의 ~84%가 첫 번째 링 안에 있음
    def airy_radial(r):
        return airy_intensity(r, airy_radius_m) * 2 * np.pi * r

    # 전체 전력 (10배 Airy radius까지 적분)
    total_power, _ = integrate.quad(airy_radial, 0, 10 * airy_radius_m)

    if offset_m == 0:
        # 중심 정렬: 1D 적분
        collected_power, _ = integrate.quad(airy_radial, 0, aperture_radius_m)
        return collected_power / total_power if total_power > 0 else 0.0
    else:
        # Offset 있는 경우: 2D 적분
        r_ap = aperture_radius_m
        d = offset_m

        def integrand(r, theta):
            # Aperture 좌표에서 빔 중심 기준 좌표로 변환
            x = r * np.cos(theta) + d
            y = r * np.sin(theta)
            r_from_beam = np.sqrt(x ** 2 + y ** 2)
            return airy_intensity(r_from_beam, airy_radius_m) * r

        collected_power, _ = integrate.dblquad(
            integrand,
            0, 2 * np.pi,
            lambda theta: 0, lambda theta: r_ap
        )

        return collected_power / total_power if total_power > 0 else 0.0


# =============================================================================
# 통합 빔 프로파일 함수
# =============================================================================

def beam_power_in_circle(beam_radius_m: float,
                          aperture_radius_m: float,
                          offset_m: float = 0.0,
                          profile: Literal["uniform", "gaussian", "airy"] = "gaussian") -> float:
    """
    빔 프로파일에 따른 원형 aperture 전력 수집 비율

    Parameters:
        beam_radius_m: 빔 반경 [m] (Gaussian: 1/e², Uniform: edge, Airy: first null)
        aperture_radius_m: 수신 aperture 반경 [m]
        offset_m: 빔 중심과 aperture 중심 간 거리 [m]
        profile: 빔 프로파일 ("uniform", "gaussian", "airy")

    Returns:
        수집 전력 비율 (0~1)
    """
    if profile == "uniform":
        return uniform_beam_power_in_circle(beam_radius_m, aperture_radius_m, offset_m)
    elif profile == "gaussian":
        return gaussian_beam_power_in_circle(beam_radius_m, aperture_radius_m, offset_m)
    elif profile == "airy":
        return airy_beam_power_in_circle(beam_radius_m, aperture_radius_m, offset_m)
    else:
        raise ValueError(f"Unknown profile: {profile}. Use 'uniform', 'gaussian', or 'airy'.")


def circle_overlap_area(R: float, r: float, d: float) -> float:
    """
    두 원의 교집합 면적 계산 (Lens formula)

    Parameters:
        R: 첫 번째 원의 반경 (빔)
        r: 두 번째 원의 반경 (aperture)
        d: 두 원의 중심 간 거리 (offset)

    Returns:
        교집합 면적 [m²]
    """
    # 겹치지 않음
    if d >= R + r:
        return 0.0

    # 작은 원이 큰 원 안에 완전히 포함
    if d <= abs(R - r):
        return np.pi * min(R, r) ** 2

    # Lens formula
    # A = r²*arccos((d²+r²-R²)/(2dr)) + R²*arccos((d²+R²-r²)/(2dR))
    #     - 0.5*sqrt((R+r-d)(d+r-R)(d-r+R)(d+R+r))
    part1 = r ** 2 * np.arccos((d ** 2 + r ** 2 - R ** 2) / (2 * d * r))
    part2 = R ** 2 * np.arccos((d ** 2 + R ** 2 - r ** 2) / (2 * d * R))
    part3 = 0.5 * np.sqrt((R + r - d) * (d + r - R) * (d - r + R) * (d + R + r))

    return part1 + part2 - part3


def bistatic_loss(beam_radius_m: float,
                  offset_m: float,
                  rx_radius_m: float,
                  profile: Literal["uniform", "gaussian", "airy"] = "uniform") -> tuple[float, float]:
    """
    Bi-static 구조 손실 계산 (offset으로 인한 추가 손실)

    TX와 RX가 offset 만큼 떨어져 있어 발생하는 추가 손실
    빔 프로파일에 따라 다르게 계산

    Parameters:
        beam_radius_m: GS 위치에서 빔 반경 [m]
        offset_m: TX-RX offset 거리 [m]
        rx_radius_m: RX aperture 반경 [m]
        profile: 빔 프로파일 ("uniform", "gaussian", "airy")

    Returns:
        (efficiency_ratio, loss_dB)
    """
    if offset_m == 0:
        # offset 없으면 추가 손실 없음
        return 1.0, 0.0

    # 프로파일별 전력 수집 비율 계산
    # offset=0일 때 (기준)
    power_centered = beam_power_in_circle(beam_radius_m, rx_radius_m, 0.0, profile)

    # offset 있을 때
    power_offset = beam_power_in_circle(beam_radius_m, rx_radius_m, offset_m, profile)

    if power_centered <= 0:
        return 0.0, float('inf')

    # offset으로 인한 효율 감소 비율
    efficiency_ratio = power_offset / power_centered

    if efficiency_ratio <= 0:
        return 0.0, float('inf')

    loss_dB = -10.0 * np.log10(efficiency_ratio)

    return efficiency_ratio, loss_dB


def gaussian_beam_power_in_annulus(beam_radius_m: float,
                                    outer_radius_m: float,
                                    inner_radius_m: float) -> float:
    """
    Gaussian 빔에서 환형 aperture가 수집하는 전력 비율

    Parameters:
        beam_radius_m: Gaussian 빔 반경 [m]
        outer_radius_m: 외경 반경 [m]
        inner_radius_m: 내경 반경 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    if inner_radius_m >= outer_radius_m:
        return 0.0

    w_sq = beam_radius_m ** 2

    # P(r < R) = 1 - exp(-2*R²/w²)
    power_outer = 1.0 - np.exp(-2.0 * outer_radius_m ** 2 / w_sq)
    power_inner = 1.0 - np.exp(-2.0 * inner_radius_m ** 2 / w_sq)

    return power_outer - power_inner


def uniform_beam_power_in_annulus(beam_radius_m: float,
                                   outer_radius_m: float,
                                   inner_radius_m: float) -> float:
    """
    Uniform 빔에서 환형 aperture가 수집하는 전력 비율

    Parameters:
        beam_radius_m: 빔 반경 [m]
        outer_radius_m: 외경 반경 [m]
        inner_radius_m: 내경 반경 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    if inner_radius_m >= outer_radius_m:
        return 0.0

    beam_area = np.pi * beam_radius_m ** 2
    if beam_area <= 0:
        return 0.0

    # 환형 영역과 빔의 교집합
    # outer circle과 빔의 교집합 - inner circle과 빔의 교집합
    outer_overlap = circle_overlap_area(beam_radius_m, outer_radius_m, 0.0)
    inner_overlap = circle_overlap_area(beam_radius_m, inner_radius_m, 0.0)

    annulus_overlap = outer_overlap - inner_overlap
    return min(annulus_overlap / beam_area, 1.0)


def airy_beam_power_in_annulus(airy_radius_m: float,
                                outer_radius_m: float,
                                inner_radius_m: float) -> float:
    """
    Airy 빔에서 환형 aperture가 수집하는 전력 비율

    Parameters:
        airy_radius_m: Airy disk 반경 [m]
        outer_radius_m: 외경 반경 [m]
        inner_radius_m: 내경 반경 [m]

    Returns:
        수집 전력 비율 (0~1)
    """
    if inner_radius_m >= outer_radius_m:
        return 0.0

    def airy_radial(r):
        return airy_intensity(r, airy_radius_m) * 2 * np.pi * r

    # 전체 전력
    total_power, _ = integrate.quad(airy_radial, 0, 10 * airy_radius_m)
    if total_power <= 0:
        return 0.0

    # 환형 영역 전력
    power_outer, _ = integrate.quad(airy_radial, 0, outer_radius_m)
    power_inner, _ = integrate.quad(airy_radial, 0, inner_radius_m)

    return (power_outer - power_inner) / total_power


def beam_power_in_annulus(beam_radius_m: float,
                           outer_radius_m: float,
                           inner_radius_m: float,
                           profile: Literal["uniform", "gaussian", "airy"] = "gaussian") -> float:
    """
    빔 프로파일에 따른 환형 aperture 전력 수집 비율

    Parameters:
        beam_radius_m: 빔 반경 [m]
        outer_radius_m: 외경 반경 [m]
        inner_radius_m: 내경 반경 [m]
        profile: 빔 프로파일 ("uniform", "gaussian", "airy")

    Returns:
        수집 전력 비율 (0~1)
    """
    if profile == "uniform":
        return uniform_beam_power_in_annulus(beam_radius_m, outer_radius_m, inner_radius_m)
    elif profile == "gaussian":
        return gaussian_beam_power_in_annulus(beam_radius_m, outer_radius_m, inner_radius_m)
    elif profile == "airy":
        return airy_beam_power_in_annulus(beam_radius_m, outer_radius_m, inner_radius_m)
    else:
        raise ValueError(f"Unknown profile: {profile}")


def concentric_loss(beam_radius_m: float,
                    tx_inner_radius_m: float,
                    rx_outer_radius_m: float) -> tuple[float, float]:
    """
    Concentric 구조 손실 계산 (상대 손실)

    TX가 중앙에 위치하고 RX가 TX를 둘러싸는 환형 구조
    TX 내경으로 인해 환형 영역만 수신

    동일 외경의 원형 aperture 대비 상대적 손실을 계산
    (bistatic_loss와 동일한 방식)

    Parameters:
        beam_radius_m: GS 위치에서 빔 반경 [m]
        tx_inner_radius_m: TX 내경 (= RX 내경) [m]
        rx_outer_radius_m: RX 외경 [m]

    Returns:
        (efficiency_ratio, loss_dB): 원형 aperture 대비 상대 효율
    """
    # 기준: 동일 외경의 원형 aperture (inner hole 없음)
    power_full_circle = gaussian_beam_power_in_circle(
        beam_radius_m, rx_outer_radius_m, 0.0
    )

    # 환형 aperture
    power_annulus = gaussian_beam_power_in_annulus(
        beam_radius_m, rx_outer_radius_m, tx_inner_radius_m
    )

    if power_full_circle <= 0:
        return 0.0, float('inf')

    # 상대적 효율 (원형 대비 환형)
    efficiency_ratio = power_annulus / power_full_circle

    if efficiency_ratio <= 0:
        return 0.0, float('inf')

    loss_dB = -10.0 * np.log10(efficiency_ratio)

    return efficiency_ratio, loss_dB


def receiver_config_loss(beam_radius_m: float,
                         config: Literal["bistatic", "concentric"],
                         rx_outer_radius_m: float,
                         offset_m: float = 0.0,
                         tx_inner_radius_m: float = 0.0,
                         profile: Literal["uniform", "gaussian", "airy"] = "uniform") -> tuple[float, float]:
    """
    수신기 구조 손실 통합 함수

    Parameters:
        beam_radius_m: GS 위치에서 빔 반경 [m]
        config: "bistatic" 또는 "concentric"
        rx_outer_radius_m: RX 외경 반경 [m]
        offset_m: Bistatic TX-RX offset [m] (bistatic에서만 사용)
        tx_inner_radius_m: Concentric TX 내경 반경 [m] (concentric에서만 사용)
        profile: 빔 프로파일 ("uniform", "gaussian", "airy")

    Returns:
        (efficiency_ratio, loss_dB)
    """
    if config == "bistatic":
        return bistatic_loss(beam_radius_m, offset_m, rx_outer_radius_m, profile)
    elif config == "concentric":
        return concentric_loss(beam_radius_m, tx_inner_radius_m, rx_outer_radius_m)
    else:
        raise ValueError(f"Unknown config: {config}. Use 'bistatic' or 'concentric'.")


def downlink_collection_coefficient(
    beam_radius_m: float,
    config: Literal["bistatic", "concentric"],
    rx_outer_radius_m: float,
    offset_m: float = 0.0,
    tx_inner_radius_m: float = 0.0,
    profile: Literal["uniform", "gaussian", "airy"] = "gaussian"
) -> float:
    """
    다운링크 수광 계수 (h_pg) 통합 계산

    빔 프로파일과 수신기 구조를 고려하여 실제 수광 효율을 계산.
    기존의 h_pg × h_rx_config를 통합한 단일 계수.

    계산 방식:
    - Bistatic: beam_power_in_circle(beam_radius, rx_radius, offset, profile)
    - Concentric: beam_power_in_annulus(beam_radius, rx_outer, tx_inner, profile)

    Parameters:
        beam_radius_m: GS 위치에서 빔 반경 [m]
        config: "bistatic" 또는 "concentric"
        rx_outer_radius_m: RX aperture 반경 [m]
        offset_m: Bistatic TX-RX offset [m] (bistatic에서만 사용)
        tx_inner_radius_m: Concentric TX 내경 반경 [m] (concentric에서만 사용)
        profile: 빔 프로파일 ("uniform", "gaussian", "airy")

    Returns:
        h_pg: 수광 계수 (0~1)
    """
    if config == "bistatic":
        # Bistatic: 원형 aperture (offset 포함)
        return beam_power_in_circle(beam_radius_m, rx_outer_radius_m, offset_m, profile)
    elif config == "concentric":
        # Concentric: 환형 aperture
        return beam_power_in_annulus(beam_radius_m, rx_outer_radius_m, tx_inner_radius_m, profile)
    else:
        raise ValueError(f"Unknown config: {config}. Use 'bistatic' or 'concentric'.")
