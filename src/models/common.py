"""
공통 상수 및 유틸리티 함수
"""
import numpy as np
from typing import Union

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J·s

# Unit conversion helpers
def mrad_to_rad(mrad: float) -> float:
    """mrad to rad conversion"""
    return mrad * 1e-3

def rad_to_mrad(rad: float) -> float:
    """rad to mrad conversion"""
    return rad * 1e3

def nm_to_m(nm: float) -> float:
    """nm to m conversion"""
    return nm * 1e-9

def m_to_nm(m: float) -> float:
    """m to nm conversion"""
    return m * 1e9

def um_to_m(um: float) -> float:
    """μm to m conversion"""
    return um * 1e-6

def m_to_um(m: float) -> float:
    """m to μm conversion"""
    return m * 1e6

def cm_to_m(cm: float) -> float:
    """cm to m conversion"""
    return cm * 1e-2

def m_to_cm(m: float) -> float:
    """m to cm conversion"""
    return m * 1e2

def mm_to_m(mm: float) -> float:
    """mm to m conversion"""
    return mm * 1e-3

def m_to_mm(m: float) -> float:
    """m to mm conversion"""
    return m * 1e3

def deg_to_rad(deg: float) -> float:
    """degree to radian conversion"""
    return np.deg2rad(deg)

def rad_to_deg(rad: float) -> float:
    """radian to degree conversion"""
    return np.rad2deg(rad)

def urad_to_rad(urad: float) -> float:
    """μrad to rad conversion"""
    return urad * 1e-6

def rad_to_urad(rad: float) -> float:
    """rad to μrad conversion"""
    return rad * 1e6


# dB conversion functions
def linear_to_dB(linear: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    선형 값을 dB로 변환 (전력 비율)

    dB = 10 * log10(linear)
    """
    return 10.0 * np.log10(linear)


def dB_to_linear(dB: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    dB를 선형 값으로 변환 (전력 비율)

    linear = 10^(dB/10)
    """
    return np.power(10.0, dB / 10.0)


def dBm_to_W(dBm: float) -> float:
    """
    dBm을 Watt로 변환

    W = 10^((dBm - 30) / 10)
    """
    return np.power(10.0, (dBm - 30.0) / 10.0)


def W_to_dBm(W: float) -> float:
    """
    Watt를 dBm으로 변환

    dBm = 10 * log10(W) + 30
    """
    return 10.0 * np.log10(W) + 30.0


# Free Space Loss
def free_space_loss(wavelength_m: float, distance_m: float) -> tuple[float, float]:
    """
    자유 공간 손실 (Free Space Loss) 계산

    FSL = (λ / (4πR))²

    Parameters:
        wavelength_m: 파장 [m]
        distance_m: 거리 [m]

    Returns:
        (loss_linear, loss_dB): 선형 및 dB 값 (둘 다 양수, 손실을 나타냄)
    """
    fsl_linear = (wavelength_m / (4.0 * np.pi * distance_m)) ** 2
    fsl_dB = linear_to_dB(fsl_linear)
    return fsl_linear, fsl_dB


# Antenna/Aperture Gain
def aperture_gain(diameter_m: float, wavelength_m: float,
                  efficiency: float = 1.0) -> tuple[float, float]:
    """
    수신 Aperture 이득 계산

    G = η * (π * D / λ)²

    Parameters:
        diameter_m: 구경 직경 [m]
        wavelength_m: 파장 [m]
        efficiency: 효율 (0~1, default: 1.0)

    Returns:
        (gain_linear, gain_dB)
    """
    gain_linear = efficiency * (np.pi * diameter_m / wavelength_m) ** 2
    gain_dB = linear_to_dB(gain_linear)
    return gain_linear, gain_dB


def transmitter_gain(divergence_rad: float) -> tuple[float, float]:
    """
    송신기 이득 계산 (발산각 기반)

    G_tx = 32 / θ_div² (full angle e⁻² divergence 기준)

    Parameters:
        divergence_rad: 발산각 (half-angle) [rad]

    Returns:
        (gain_linear, gain_dB)

    Note:
        Full angle = 2 * half-angle
        수식의 θ_div는 full angle이므로 (2 * divergence_rad)²를 사용
    """
    full_angle = 2.0 * divergence_rad
    gain_linear = 32.0 / (full_angle ** 2)
    gain_dB = linear_to_dB(gain_linear)
    return gain_linear, gain_dB


def wave_number(wavelength_m: float) -> float:
    """
    파수 (wave number) 계산

    k = 2π / λ

    Parameters:
        wavelength_m: 파장 [m]

    Returns:
        k [rad/m]
    """
    return 2.0 * np.pi / wavelength_m
