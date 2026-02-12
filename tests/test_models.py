"""
MRR Link Budget Simulator 테스트
"""
import pytest
import numpy as np

from src.models.common import (
    nm_to_m, cm_to_m, mm_to_m, mrad_to_rad, deg_to_rad, urad_to_rad,
    linear_to_dB, dB_to_linear, dBm_to_W, W_to_dBm,
    free_space_loss, aperture_gain, transmitter_gain
)
from src.atmosphere.attenuation import (
    kim_attenuation, ijaz_attenuation, atmospheric_attenuation
)
from src.atmosphere.turbulence import (
    hufnagel_valley_cn2, calculate_rytov_variance_constant,
    sample_lognormal_fading, sample_gamma_gamma_fading
)
from src.mrr.efficiency import eta_mrr, mrr_m2, mrr_reflection_ratio, ccr_orientation_loss_dB, ccr_mean_h_mrr
from src.mrr.modulation import modulation_loss_dB, ccr_passive_loss_dB
from src.geometry.pointing import (
    calculate_beam_diameter_at_distance, tracking_error_loss,
    downlink_divergence
)
from src.geometry.receiver import bistatic_loss, concentric_loss
from src.models.antenna_model import AntennaModelParams, calculate_antenna_link_budget
from src.models.optical_model import OpticalModelParams, calculate_optical_channel_coefficient


class TestUnitConversions:
    """단위 변환 테스트"""

    def test_nm_to_m(self):
        assert nm_to_m(1550) == pytest.approx(1.55e-6)

    def test_cm_to_m(self):
        assert cm_to_m(10) == pytest.approx(0.1)

    def test_mrad_to_rad(self):
        assert mrad_to_rad(1) == pytest.approx(0.001)

    def test_dB_conversions(self):
        assert linear_to_dB(10) == pytest.approx(10)
        assert dB_to_linear(10) == pytest.approx(10)
        assert linear_to_dB(dB_to_linear(3)) == pytest.approx(3)

    def test_dBm_W_conversions(self):
        assert dBm_to_W(30) == pytest.approx(1.0)  # 30 dBm = 1W
        assert W_to_dBm(1.0) == pytest.approx(30)


class TestAtmosphericAttenuation:
    """대기 감쇠 모델 테스트"""

    def test_kim_high_visibility(self):
        # V = 23 km, clear day
        atten, atten_dB = kim_attenuation(1550, 23.0, 1.0)
        assert atten_dB > 0  # 양수 손실
        assert atten < 1.0   # 선형 감쇠

    def test_kim_low_visibility(self):
        # V = 1 km, hazy
        atten, atten_dB = kim_attenuation(1550, 1.0, 1.0)
        assert atten_dB > kim_attenuation(1550, 23.0, 1.0)[1]

    def test_ijaz_fog(self):
        # V < 1 km, fog
        atten, atten_dB = ijaz_attenuation(1.55, 0.5, 1.0, "fog")
        assert atten_dB > 0

    def test_ijaz_smoke(self):
        atten, atten_dB = ijaz_attenuation(1.55, 0.5, 1.0, "smoke")
        assert atten_dB > 0

    def test_atmospheric_model_selection(self):
        # V >= 1 km → Kim
        _, dB_kim = atmospheric_attenuation(1550, 5.0, 1.0)
        # V < 1 km → Ijaz
        _, dB_ijaz = atmospheric_attenuation(1550, 0.5, 1.0, "fog")
        # 안개가 더 많은 감쇠
        assert dB_ijaz > dB_kim


class TestTurbulence:
    """난류 모델 테스트"""

    def test_hufnagel_valley_cn2(self):
        # 지표면에서 가장 높음
        cn2_0 = hufnagel_valley_cn2(0)
        cn2_100 = hufnagel_valley_cn2(100)
        cn2_1000 = hufnagel_valley_cn2(1000)
        assert cn2_0 > cn2_100 > cn2_1000

    def test_rytov_variance(self):
        wavelength = 1.55e-6
        distance = 1000
        cn2 = 1e-14

        sigma_r2_spherical = calculate_rytov_variance_constant(
            wavelength, distance, cn2, "spherical"
        )
        sigma_r2_plane = calculate_rytov_variance_constant(
            wavelength, distance, cn2, "plane"
        )
        # Plane wave > Spherical wave
        assert sigma_r2_plane > sigma_r2_spherical

    def test_lognormal_sampling(self):
        samples = sample_lognormal_fading(0.5, 1000)
        assert len(samples) == 1000
        assert np.mean(samples) == pytest.approx(1.0, rel=0.1)

    def test_gamma_gamma_sampling(self):
        samples = sample_gamma_gamma_fading(2.0, 1000, "plane")
        assert len(samples) == 1000
        assert np.all(samples >= 0)


class TestMRR:
    """MRR 효율 테스트"""

    def test_eta_mrr_flat_region(self):
        # theta < knee → 1.0
        assert eta_mrr(0) == 1.0
        assert eta_mrr(1.0) == 1.0
        assert eta_mrr(2.0) == 1.0

    def test_eta_mrr_rolloff(self):
        # knee < theta < max → 0 < eta < 1
        eta = eta_mrr(2.5)
        assert 0 < eta < 1

    def test_eta_mrr_outside_fov(self):
        # theta >= max → 0
        assert eta_mrr(3.5) == 0.0

    def test_mrr_m2(self):
        # theta < knee → m2_min
        assert mrr_m2(0) == pytest.approx(1.3)
        # theta >= max → m2_max
        assert mrr_m2(3.5) == pytest.approx(3.4)

    def test_mrr_reflection_ratio(self):
        # Small angles → high ratio
        ratio = mrr_reflection_ratio(1.0, 1.0, 1.0)
        assert 0 < ratio < 1

    def test_modulation_loss(self):
        loss_dB = modulation_loss_dB(0.5)
        assert loss_dB == pytest.approx(3.01, rel=0.1)  # 50% → ~3 dB


class TestGeometry:
    """기하 계산 테스트"""

    def test_beam_diameter_far_field(self):
        # No TX diameter → far-field approximation
        diameter = calculate_beam_diameter_at_distance(0.001, 1000)
        assert diameter == pytest.approx(2.0)  # 2 * 0.001 * 1000

    def test_beam_diameter_with_tx(self):
        # With TX diameter
        diameter = calculate_beam_diameter_at_distance(0.001, 1000, 0.01)
        assert diameter == pytest.approx(2.01)  # 0.01 + 2*0.001*1000

    def test_tracking_error_loss(self):
        loss_linear, loss_dB = tracking_error_loss(100e-6, 1000, 0.5)
        assert loss_dB > 0
        assert 0 < loss_linear < 1

    def test_downlink_divergence(self):
        div = downlink_divergence(1.55e-6, 0.02, 1.5)
        assert div > 0


class TestReceiver:
    """수신부 구조 테스트"""

    def test_bistatic_loss(self):
        collected, loss_dB = bistatic_loss(1.0, 0.1, 0.08, profile="gaussian")
        assert 0 < collected < 1
        assert loss_dB > 0

    def test_concentric_loss(self):
        collected, loss_dB = concentric_loss(1.0, 0.025, 0.08)
        assert 0 < collected < 1
        assert loss_dB > 0


class TestAntennaModel:
    """Antenna 모델 통합 테스트"""

    def test_default_params(self):
        params = AntennaModelParams()
        result = calculate_antenna_link_budget(params)

        # 기본 파라미터로 링크 클로즈 가능
        assert result.link_margin_dB > 0
        assert result.receiver_power_dBm > params.receiver_sensitivity_dBm

    def test_long_distance(self):
        params = AntennaModelParams(distance_m=5000)
        result = calculate_antenna_link_budget(params)

        # 거리 증가 → 마진 감소
        params_short = AntennaModelParams(distance_m=1000)
        result_short = calculate_antenna_link_budget(params_short)

        assert result.link_margin_dB < result_short.link_margin_dB

    def test_high_tx_power(self):
        params_low = AntennaModelParams(P_tx_dBm=20)
        params_high = AntennaModelParams(P_tx_dBm=40)

        result_low = calculate_antenna_link_budget(params_low)
        result_high = calculate_antenna_link_budget(params_high)

        # 높은 전력 → 높은 마진
        assert result_high.link_margin_dB > result_low.link_margin_dB


class TestOpticalModel:
    """Optical 모델 통합 테스트"""

    def test_default_params(self):
        params = OpticalModelParams()
        result = calculate_optical_channel_coefficient(params)

        assert result.h_total > 0
        assert result.P_rx_W > 0

    def test_channel_coefficient_range(self):
        params = OpticalModelParams()
        result = calculate_optical_channel_coefficient(params)

        # 채널 계수는 0과 1 사이
        assert 0 < result.h_total < 1


class TestCCR:
    """CCR (Corner Cube Retroreflector) 테스트"""

    def test_ccr_orientation_loss(self):
        """CCR orientation loss는 양수, 합리적 범위"""
        loss = ccr_orientation_loss_dB(3.0)
        assert loss > 0
        assert loss < 10  # 합리적 범위

    def test_ccr_mean_h_mrr_table_ii(self):
        """Table II 검증: σ=1°→~0.96, σ=3°→~0.86, σ=5°→~0.83"""
        h_1 = ccr_mean_h_mrr(1.0)
        assert h_1 == pytest.approx(0.96, abs=0.02)

        h_3 = ccr_mean_h_mrr(3.0)
        assert h_3 == pytest.approx(0.86, abs=0.03)

        h_5 = ccr_mean_h_mrr(5.0)
        assert h_5 == pytest.approx(0.83, abs=0.05)

    def test_ccr_mean_h_mrr_zero_sigma(self):
        """σ=0일 때 h_MRR ≈ 1.0"""
        h_0 = ccr_mean_h_mrr(0.01)
        assert h_0 == pytest.approx(1.0, abs=0.01)

    def test_ccr_passive_loss(self):
        """CCR passive loss: 0.99^3 + AR = 예상값"""
        loss = ccr_passive_loss_dB(0.99, 0.5)
        expected_reflection = -10 * np.log10(0.99 ** 3)
        assert loss == pytest.approx(expected_reflection + 0.5, rel=0.01)

    def test_ccr_passive_loss_default(self):
        """기본값으로 계산 가능"""
        loss = ccr_passive_loss_dB()
        assert loss > 0
        assert loss < 2.0  # 합리적 범위

    def test_ccr_antenna_model(self):
        """CCR 모드에서 Antenna Model 정상 작동"""
        params = AntennaModelParams(
            reflector_type="ccr",
            ccr_surface_reflectivity=0.99,
            ccr_m2=1.05
        )
        result = calculate_antenna_link_budget(params)

        # L_orientation = 0 확인
        assert result.uplink.L_orientation_dB == 0.0

        # 정상적인 결과
        assert result.link_margin_dB != 0
        assert np.isfinite(result.receiver_power_dBm)

    def test_ccr_optical_model(self):
        """CCR 모드에서 Optical Model 정상 작동"""
        params = OpticalModelParams(
            reflector_type="ccr",
            ccr_surface_reflectivity=0.99,
            ccr_m2=1.05
        )
        result = calculate_optical_channel_coefficient(params)

        # h_orientation = 1.0 확인
        assert result.h_orientation == 1.0

        # 정상적인 결과
        assert result.h_total > 0
        assert result.P_rx_W > 0

    def test_mrr_backward_compatibility(self):
        """기본값(MRR)으로 기존 동작 유지"""
        # 기본 파라미터 (reflector_type 명시 없음 → "mrr")
        params_default = AntennaModelParams()
        result_default = calculate_antenna_link_budget(params_default)

        # 명시적 MRR 파라미터
        params_mrr = AntennaModelParams(reflector_type="mrr")
        result_mrr = calculate_antenna_link_budget(params_mrr)

        # 동일한 결과
        assert result_default.link_margin_dB == pytest.approx(result_mrr.link_margin_dB)
        assert result_default.receiver_power_dBm == pytest.approx(result_mrr.receiver_power_dBm)

    def test_ccr_vs_mrr_different(self):
        """CCR과 MRR이 다른 결과를 생성"""
        params_mrr = AntennaModelParams(reflector_type="mrr")
        params_ccr = AntennaModelParams(reflector_type="ccr")

        result_mrr = calculate_antenna_link_budget(params_mrr)
        result_ccr = calculate_antenna_link_budget(params_ccr)

        # 다른 결과
        assert result_mrr.link_margin_dB != result_ccr.link_margin_dB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
