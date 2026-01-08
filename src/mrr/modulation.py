"""
MQW (Multiple Quantum Well) 변조 효율
"""
import numpy as np


def mqw_modulation_efficiency(alpha_on: float, alpha_off: float) -> float:
    """
    MQW 변조 효율 계산

    수식:
        M = exp(-α_On) - exp(-α_Off)
          = exp(-α_Off) * (C_MQW - 1)

    여기서 C_MQW = exp(α_Off - α_On) = contrast ratio

    Parameters:
        alpha_on: ON 상태 흡수 계수
        alpha_off: OFF 상태 흡수 계수

    Returns:
        변조 효율 M (0~1)
    """
    return np.exp(-alpha_on) - np.exp(-alpha_off)


def mqw_modulation_efficiency_from_contrast(contrast_ratio: float,
                                            alpha_off: float = 0.1) -> float:
    """
    Contrast ratio로부터 변조 효율 계산

    M = exp(-α_Off) * (C_MQW - 1)

    Parameters:
        contrast_ratio: C_MQW = T_on / T_off
        alpha_off: OFF 상태 흡수 계수 (default: 0.1)

    Returns:
        변조 효율 M
    """
    return np.exp(-alpha_off) * (contrast_ratio - 1)


def modulation_loss_dB(modulation_efficiency: float) -> float:
    """
    변조 손실 [dB]

    Parameters:
        modulation_efficiency: 변조 효율 M (0~1)

    Returns:
        변조 손실 [dB] (양수)
    """
    if modulation_efficiency <= 0:
        return float('inf')

    return -10.0 * np.log10(modulation_efficiency)


def mrr_passive_loss_dB(ar_coating_loss_dB: float = 0.5,
                        reflectivity_loss_dB: float = 0.3) -> float:
    """
    MRR passive 손실 (AR 코팅 + 반사율)

    Parameters:
        ar_coating_loss_dB: AR 코팅 손실 [dB] (default: 0.5)
        reflectivity_loss_dB: 반사율 손실 [dB] (default: 0.3)

    Returns:
        총 passive 손실 [dB]
    """
    return ar_coating_loss_dB + reflectivity_loss_dB
