"""
대기 난류 모델 (Atmospheric Turbulence Models)

- Cn² 프로파일 (Hufnagel-Valley)
- Rytov Variance 계산
- 페이딩 분포 (Log-Normal, Gamma-Gamma)
"""
import numpy as np
from scipy import special
from scipy import integrate
from typing import Callable, Union, Literal


# =============================================================================
# Cn² 프로파일
# =============================================================================

def hufnagel_valley_cn2(height_m: float,
                        wind_speed: float = 21.0,
                        cn2_ground: float = 1.7e-14) -> float:
    """
    Hufnagel-Valley 5/7 Cn² 프로파일

    수식:
        C_n²(h) = 0.00594*(V/27)² * (10⁻⁵*h)^10 * exp(-h/1000)
                + 2.7×10⁻¹⁶ * exp(-h/1500)
                + C_n²(0) * exp(-h/100)

    Parameters:
        height_m: 고도 [m]
        wind_speed: 풍속 [m/s] (default: 21.0)
        cn2_ground: 지표면 Cn² [m^(-2/3)] (default: 1.7e-14)

    Returns:
        C_n² [m^(-2/3)]
    """
    h = height_m

    # 첫 번째 항: 성층권 기여
    term1 = 0.00594 * (wind_speed / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)

    # 두 번째 항: 대류권 상부 기여
    term2 = 2.7e-16 * np.exp(-h / 1500.0)

    # 세 번째 항: 지표면 근처 기여
    term3 = cn2_ground * np.exp(-h / 100.0)

    return term1 + term2 + term3


def constant_cn2(cn2_value: float) -> Callable[[float], float]:
    """
    상수 Cn² 프로파일 생성

    Parameters:
        cn2_value: 상수 Cn² 값 [m^(-2/3)]

    Returns:
        Cn² 함수 (고도를 입력받아 상수 반환)
    """
    return lambda h: cn2_value


# =============================================================================
# Rytov Variance 계산
# =============================================================================

def calculate_rytov_variance_constant(wavelength_m: float,
                                       distance_m: float,
                                       cn2: float,
                                       wave_type: Literal["spherical", "plane"] = "spherical") -> float:
    """
    상수 Cn² 가정 시 Rytov variance 계산

    수식:
        Uplink (Spherical):  σ_R² = 0.5 * C_n² * k^(7/6) * Z^(11/6)
        Downlink (Plane):    σ_R² = 1.23 * C_n² * k^(7/6) * Z^(11/6)

    Parameters:
        wavelength_m: 파장 [m]
        distance_m: 전파 거리 [m]
        cn2: 상수 Cn² 값 [m^(-2/3)]
        wave_type: "spherical" (uplink) 또는 "plane" (downlink)

    Returns:
        σ_R² (Rytov variance)
    """
    k = 2.0 * np.pi / wavelength_m  # wave number

    if wave_type == "spherical":
        # Uplink - 구면파
        coefficient = 0.5
    else:
        # Downlink - 평면파
        coefficient = 1.23

    sigma_r2 = coefficient * cn2 * (k ** (7.0 / 6.0)) * (distance_m ** (11.0 / 6.0))

    return sigma_r2


def calculate_rytov_variance_profile(wavelength_m: float,
                                      h_gs_m: float,
                                      h_uav_m: float,
                                      cn2_func: Callable[[float], float],
                                      direction: Literal["uplink", "downlink"] = "uplink",
                                      zenith_angle_deg: float = 0.0) -> float:
    """
    고도 프로파일 기반 Rytov variance 계산 (적분 수식)

    Uplink (GS → UAV, Spherical wave):
        σ_R² = 9*(2π/λ)^(7/6) * (Z/Z_hd)^(11/6)
             * ∫[C_n²(h) * (1 - (h-h_gs)/Z_hd)^(5/6) * (h-h_gs)^(5/6)] dh

    Parameters:
        wavelength_m: 파장 [m]
        h_gs_m: 지상국 고도 [m]
        h_uav_m: UAV 고도 [m]
        cn2_func: Cn²(h) 함수
        direction: "uplink" 또는 "downlink"
        zenith_angle_deg: 천정각 [deg] (0 = 수직)

    Returns:
        σ_R² (Rytov variance)
    """
    k = 2.0 * np.pi / wavelength_m
    Z_hd = h_uav_m - h_gs_m  # 고도 차이

    # Slant range 계산
    if zenith_angle_deg == 0:
        Z = Z_hd
    else:
        Z = Z_hd / np.cos(np.deg2rad(zenith_angle_deg))

    if direction == "uplink":
        # Spherical wave (GS → UAV)
        def integrand(h):
            xi = (h - h_gs_m) / Z_hd
            return cn2_func(h) * ((1 - xi) ** (5.0 / 6.0)) * ((h - h_gs_m) ** (5.0 / 6.0))

        result, _ = integrate.quad(integrand, h_gs_m, h_uav_m)
        sigma_r2 = 9.0 * (k ** (7.0 / 6.0)) * ((Z / Z_hd) ** (11.0 / 6.0)) * result

    else:
        # Plane wave (UAV → GS)
        # Downlink의 경우 weighting factor가 다름
        def integrand(h):
            xi = (h - h_gs_m) / Z_hd
            return cn2_func(h) * (xi ** (5.0 / 6.0)) * ((h - h_gs_m) ** (5.0 / 6.0))

        result, _ = integrate.quad(integrand, h_gs_m, h_uav_m)
        sigma_r2 = 9.0 * (k ** (7.0 / 6.0)) * ((Z / Z_hd) ** (11.0 / 6.0)) * result

    return sigma_r2


# =============================================================================
# Gamma-Gamma 분포 파라미터
# =============================================================================

def calculate_alpha_beta(sigma_r2: float,
                         wave_type: Literal["spherical", "plane"] = "plane") -> tuple[float, float]:
    """
    Gamma-Gamma 분포의 α, β 파라미터 계산

    Plane wave:
        α = [exp(0.49*σ_R² / (1 + 1.11*σ_R^(12/5))^(7/6)) - 1]^(-1)
        β = [exp(0.51*σ_R² / (1 + 0.69*σ_R^(12/5))^(5/6)) - 1]^(-1)

    Spherical wave:
        α = [exp(0.42*σ_R² / (1 + 0.90*σ_R^(12/5))^(7/6)) - 1]^(-1)
        β = [exp(0.56*σ_R² / (1 + 0.62*σ_R^(12/5))^(5/6)) - 1]^(-1)

    Parameters:
        sigma_r2: Rytov variance (σ_R²)
        wave_type: "plane" 또는 "spherical"

    Returns:
        (α, β)
    """
    sigma_r_12_5 = sigma_r2 ** (6.0 / 5.0)  # σ_R^(12/5) = (σ_R²)^(6/5)

    if wave_type == "plane":
        # Plane wave 계수
        alpha = 1.0 / (np.exp(0.49 * sigma_r2 / ((1 + 1.11 * sigma_r_12_5) ** (7.0 / 6.0))) - 1.0)
        beta = 1.0 / (np.exp(0.51 * sigma_r2 / ((1 + 0.69 * sigma_r_12_5) ** (5.0 / 6.0))) - 1.0)
    else:
        # Spherical wave 계수
        alpha = 1.0 / (np.exp(0.42 * sigma_r2 / ((1 + 0.90 * sigma_r_12_5) ** (7.0 / 6.0))) - 1.0)
        beta = 1.0 / (np.exp(0.56 * sigma_r2 / ((1 + 0.62 * sigma_r_12_5) ** (5.0 / 6.0))) - 1.0)

    return alpha, beta


# =============================================================================
# 페이딩 분포 샘플링
# =============================================================================

def sample_lognormal_fading(sigma_r2: float, n_samples: int,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Log-Normal 분포 페이딩 샘플링 (σ_R² < 1, weak-moderate turbulence)

    PDF:
        f_LN(h_a) = 1/(2*h_a*√(2π*σ_L²)) * exp(-(ln(h_a) + 2*σ_L²)² / (8*σ_L²))

    여기서 σ_L² ≈ σ_R² / 4

    Parameters:
        sigma_r2: Rytov variance (σ_R²)
        n_samples: 샘플 수
        rng: numpy random generator (optional)

    Returns:
        h_a 샘플 배열 (평균 ≈ 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_l2 = sigma_r2 / 4.0
    sigma_l = np.sqrt(sigma_l2)

    # Log-normal 분포: E[h] = 1이 되도록 mu 설정
    # ln(h) ~ N(mu, sigma²) → E[h] = exp(mu + sigma²/2)
    # E[h] = 1 → mu = -sigma²/2 = -2*sigma_l²
    mu = -2.0 * sigma_l2

    # 샘플링
    samples = rng.lognormal(mean=mu, sigma=2.0 * sigma_l, size=n_samples)

    return samples


def sample_gamma_gamma_fading(sigma_r2: float, n_samples: int,
                              wave_type: Literal["spherical", "plane"] = "plane",
                              rng: np.random.Generator = None) -> np.ndarray:
    """
    Gamma-Gamma 분포 페이딩 샘플링 (σ_R² >= 1, moderate-strong turbulence)

    h_a = h_x * h_y where h_x ~ Gamma(α, 1/α), h_y ~ Gamma(β, 1/β)

    Parameters:
        sigma_r2: Rytov variance (σ_R²)
        n_samples: 샘플 수
        wave_type: "plane" 또는 "spherical"
        rng: numpy random generator (optional)

    Returns:
        h_a 샘플 배열 (평균 ≈ 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    alpha, beta = calculate_alpha_beta(sigma_r2, wave_type)

    # Gamma(α, 1/α) 샘플링: shape=α, scale=1/α
    h_x = rng.gamma(shape=alpha, scale=1.0 / alpha, size=n_samples)

    # Gamma(β, 1/β) 샘플링: shape=β, scale=1/β
    h_y = rng.gamma(shape=beta, scale=1.0 / beta, size=n_samples)

    # 곱
    samples = h_x * h_y

    return samples


def sample_fading(sigma_r2: float, n_samples: int,
                  wave_type: Literal["spherical", "plane"] = "plane",
                  rng: np.random.Generator = None) -> np.ndarray:
    """
    Rytov variance에 따른 자동 분포 선택 페이딩 샘플링

    σ_R² < 1: Log-Normal
    σ_R² >= 1: Gamma-Gamma

    Parameters:
        sigma_r2: Rytov variance (σ_R²)
        n_samples: 샘플 수
        wave_type: "plane" 또는 "spherical" (Gamma-Gamma에서만 사용)
        rng: numpy random generator (optional)

    Returns:
        h_a 샘플 배열
    """
    if sigma_r2 < 1.0:
        return sample_lognormal_fading(sigma_r2, n_samples, rng)
    else:
        return sample_gamma_gamma_fading(sigma_r2, n_samples, wave_type, rng)


# =============================================================================
# 페이딩 PDF 계산
# =============================================================================

def lognormal_pdf(h: Union[float, np.ndarray], sigma_r2: float) -> Union[float, np.ndarray]:
    """
    Log-Normal 분포 PDF

    Parameters:
        h: 채널 계수 값
        sigma_r2: Rytov variance

    Returns:
        PDF 값
    """
    sigma_l2 = sigma_r2 / 4.0
    sigma_l = np.sqrt(sigma_l2)
    mu = -2.0 * sigma_l2

    h = np.asarray(h)
    pdf = np.zeros_like(h, dtype=float)
    mask = h > 0

    pdf[mask] = (1.0 / (h[mask] * 2.0 * sigma_l * np.sqrt(2.0 * np.pi))) * \
                np.exp(-((np.log(h[mask]) - mu) ** 2) / (2.0 * (2.0 * sigma_l) ** 2))

    return pdf


def gamma_gamma_pdf(h: Union[float, np.ndarray], sigma_r2: float,
                    wave_type: Literal["spherical", "plane"] = "plane") -> Union[float, np.ndarray]:
    """
    Gamma-Gamma 분포 PDF

    f_GG(h) = 2*(αβ)^((α+β)/2) / (Γ(α)*Γ(β)) * h^((α+β)/2 - 1) * K_{α-β}(2*√(αβ*h))

    Parameters:
        h: 채널 계수 값
        sigma_r2: Rytov variance
        wave_type: "plane" 또는 "spherical"

    Returns:
        PDF 값
    """
    alpha, beta = calculate_alpha_beta(sigma_r2, wave_type)

    h = np.asarray(h)
    pdf = np.zeros_like(h, dtype=float)
    mask = h > 0

    ab = alpha * beta
    ab_sum = alpha + beta
    ab_diff = alpha - beta

    coeff = 2.0 * (ab ** (ab_sum / 2.0)) / (special.gamma(alpha) * special.gamma(beta))

    pdf[mask] = coeff * (h[mask] ** (ab_sum / 2.0 - 1.0)) * \
                special.kv(ab_diff, 2.0 * np.sqrt(ab * h[mask]))

    return pdf


def scintillation_loss_dB(sigma_r2: float, probability: float = 0.99,
                          wave_type: Literal["spherical", "plane"] = "plane") -> float:
    """
    신틸레이션에 의한 페이딩 손실 추정 (확률 기반, 자동 분포 선택)

    σ_R² < 1: Log-Normal (weak-moderate turbulence)
    σ_R² >= 1: Gamma-Gamma (moderate-strong turbulence)

    Parameters:
        sigma_r2: Rytov variance
        probability: 누적 확률 (예: 0.99 = 99% 신뢰도)
        wave_type: "spherical" (uplink) 또는 "plane" (downlink)

    Returns:
        페이딩 손실 [dB] (양수 = 손실)
    """
    from scipy import stats
    from scipy import optimize

    if sigma_r2 <= 0:
        return 0.0

    if sigma_r2 < 1.0:
        # === Weak-Moderate Turbulence: Log-Normal ===
        sigma_l = np.sqrt(sigma_r2 / 4.0)

        # Log-normal 분위수: (1 - probability) 지점의 fade
        z = stats.norm.ppf(1 - probability)

        # 페이딩 값 (선형)
        # E[h] = 1이 되도록: mu = -2*sigma_l²
        fade_linear = np.exp(-2.0 * sigma_l ** 2 + 2.0 * sigma_l * z)

    else:
        # === Moderate-Strong Turbulence: Gamma-Gamma ===
        # Gamma-Gamma CDF 계산 (수치 적분)
        def gamma_gamma_cdf(h):
            if h <= 0:
                return 0.0
            # PDF 적분
            result, _ = integrate.quad(
                lambda x: gamma_gamma_pdf(x, sigma_r2, wave_type),
                0, h, limit=100
            )
            return result

        # (1 - probability) 분위수 찾기
        target_cdf = 1 - probability

        # 초기 추정: Log-Normal 근사로 시작점 설정
        sigma_l_approx = np.sqrt(sigma_r2 / 4.0)
        z_approx = stats.norm.ppf(target_cdf)
        h_init = np.exp(-2.0 * sigma_l_approx ** 2 + 2.0 * sigma_l_approx * z_approx)
        h_init = max(h_init, 1e-6)

        # Root finding: CDF(h) - target = 0
        try:
            result = optimize.brentq(
                lambda h: gamma_gamma_cdf(h) - target_cdf,
                1e-10, 10.0,  # 탐색 범위
                xtol=1e-6
            )
            fade_linear = result
        except ValueError:
            # Fallback: Log-Normal 근사
            fade_linear = h_init

    # dB 변환 (fade < 1이면 손실 > 0)
    if fade_linear <= 0:
        return 50.0  # 매우 큰 손실

    fade_dB = -10.0 * np.log10(fade_linear)

    return max(fade_dB, 0.0)  # 손실은 항상 양수
