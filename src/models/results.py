"""
계산 결과 데이터 구조
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BeamParameters:
    """빔 관련 중간 계산값 저장"""
    # 송신부
    tx_beam_diameter_m: Optional[float]  # D_beam_tx (optional, 송신부 초기 빔 직경)
    divergence_rad: float                # θ_div (항상 입력)

    # Uplink (MRR 위치)
    beam_diameter_at_mrr_m: float        # D_mrr = D_tx + 2*θ_div*Z (D_tx 있을 때)
                                         #       = 2*θ_div*Z (D_tx 없을 때, far-field)
    beam_footprint_area_m2: float        # π * (D_mrr/2)²

    # Downlink (GS 위치)
    mrr_m2_factor: float                 # MRR M² (각도 의존적)
    downlink_divergence_rad: float       # θ_down = M²_mrr * 4λ / (π*D_eff)
    beam_diameter_at_gs_m: float         # GS 위치에서 다운링크 빔 직경


@dataclass
class UplinkBudget:
    """Uplink 링크 버짓 상세 [dB]"""
    P_tx_dBm: float                      # 송신 전력
    L_tx_optics_dB: float                # 송신 광학계 손실
    G_tx_dB: float                       # 송신기 이득
    L_FSL_dB: float                      # 자유 공간 손실
    L_atm_dB: float                      # 대기 감쇠 (Kim/Ijaz)
    L_scint_dB: float                    # 신틸레이션 손실 (평균)
    L_tracking_dB: float                 # 추적 오차 손실
    L_orientation_dB: float              # UAV 자세 오차 손실 (eta_mrr)
    L_AR_coating_dB: float               # MRR AR 코팅 손실
    G_rx_mrr_dB: float                   # MRR 수신 이득

    @property
    def P_mrr_in_dBm(self) -> float:
        """MRR 입력 전력 [dBm]"""
        return (self.P_tx_dBm
                - self.L_tx_optics_dB
                + self.G_tx_dB
                + self.L_FSL_dB          # FSL은 음수
                - self.L_atm_dB
                - self.L_scint_dB
                - self.L_tracking_dB
                - self.L_orientation_dB
                - self.L_AR_coating_dB
                + self.G_rx_mrr_dB)

    def to_dict(self) -> dict[str, float]:
        """항목별 딕셔너리 반환"""
        return {
            "P_tx [dBm]": self.P_tx_dBm,
            "L_tx_optics [dB]": -self.L_tx_optics_dB,
            "G_tx [dB]": self.G_tx_dB,
            "L_FSL [dB]": self.L_FSL_dB,
            "L_atm [dB]": -self.L_atm_dB,
            "L_scint [dB]": -self.L_scint_dB,
            "L_tracking [dB]": -self.L_tracking_dB,
            "L_orientation [dB]": -self.L_orientation_dB,
            "L_AR_coating [dB]": -self.L_AR_coating_dB,
            "G_rx_mrr [dB]": self.G_rx_mrr_dB,
            "P_mrr_in [dBm]": self.P_mrr_in_dBm,
        }


@dataclass
class DownlinkBudget:
    """Downlink 링크 버짓 상세 [dB]"""
    P_mrr_in_dBm: float                  # MRR 입력 전력 (Uplink 결과)
    G_mrr_rx_dB: float                   # MRR 반사 이득 (retro-reflection)
    L_modulation_dB: float               # 변조 손실
    L_mrr_passive_dB: float              # MRR passive 손실
    L_FSL_dB: float                      # 자유 공간 손실
    L_atm_dB: float                      # 대기 감쇠
    L_scint_dB: float                    # 신틸레이션 손실
    G_rx_gs_dB: float                    # GS 수신 이득
    L_rx_config_dB: float                # 수신부 구조 손실 (bistatic/concentric)
    L_rx_optics_dB: float                # 수신 광학계 손실

    @property
    def P_rx_dBm(self) -> float:
        """GS 수신 전력 [dBm]"""
        return (self.P_mrr_in_dBm
                + self.G_mrr_rx_dB
                - self.L_modulation_dB
                - self.L_mrr_passive_dB
                + self.L_FSL_dB          # FSL은 음수
                - self.L_atm_dB
                - self.L_scint_dB
                + self.G_rx_gs_dB
                - self.L_rx_config_dB
                - self.L_rx_optics_dB)

    def to_dict(self) -> dict[str, float]:
        """항목별 딕셔너리 반환"""
        return {
            "P_mrr_in [dBm]": self.P_mrr_in_dBm,
            "G_mrr_rx [dB]": self.G_mrr_rx_dB,
            "L_modulation [dB]": -self.L_modulation_dB,
            "L_mrr_passive [dB]": -self.L_mrr_passive_dB,
            "L_FSL [dB]": self.L_FSL_dB,
            "L_atm [dB]": -self.L_atm_dB,
            "L_scint [dB]": -self.L_scint_dB,
            "G_rx_gs [dB]": self.G_rx_gs_dB,
            "L_rx_config [dB]": -self.L_rx_config_dB,
            "L_rx_optics [dB]": -self.L_rx_optics_dB,
            "P_rx [dBm]": self.P_rx_dBm,
        }


@dataclass
class LinkBudgetResult:
    """전체 링크 버짓 결과"""
    # 빔 파라미터
    beam: BeamParameters

    # Uplink/Downlink 상세
    uplink: UplinkBudget
    downlink: DownlinkBudget

    # 최종 결과
    receiver_sensitivity_dBm: float
    link_margin_dB: float = field(init=False)

    # Optical model 결과 (선택적)
    channel_coefficient_h: Optional[float] = None
    snr_linear: Optional[float] = None

    def __post_init__(self):
        """링크 마진 계산"""
        self.link_margin_dB = self.downlink.P_rx_dBm - self.receiver_sensitivity_dBm

    @property
    def receiver_power_dBm(self) -> float:
        """수신 전력 [dBm]"""
        return self.downlink.P_rx_dBm

    def get_full_breakdown(self) -> dict[str, dict[str, float]]:
        """전체 breakdown 딕셔너리 반환"""
        return {
            "uplink": self.uplink.to_dict(),
            "downlink": self.downlink.to_dict(),
            "summary": {
                "P_rx [dBm]": self.receiver_power_dBm,
                "Sensitivity [dBm]": self.receiver_sensitivity_dBm,
                "Link Margin [dB]": self.link_margin_dB,
            }
        }


@dataclass
class OpticalModelResult:
    """Optical Model 채널 계수 결과 (Gaussian 빔 전파)"""
    # 빔 파라미터
    beam: BeamParameters

    # 송신 광학계 (Antenna Model L_tx_optics에 대응)
    eta_tx: float            # 송신 광학계 효율 (0~1)

    # Uplink 계수 (선형) - h_orientation 포함
    h_geometric: float       # Uplink 기하 손실 (aperture/footprint, pointing 미포함)
    h_tracking: float        # Uplink 추적 오차 손실 (pointing error)
    h_pu: float              # h_geometric * h_tracking (기하+pointing)
    h_lgu: float             # Uplink 대기 감쇠
    h_aug: float             # Uplink 난류 페이딩 (마진 모드: <1, 평균 모드: =1)
    h_orientation: float     # UAV 자세 오차 손실 (Antenna Model L_orientation에 대응)
    sigma_r2_uplink: float   # Uplink Rytov variance
    L_scint_up_dB: float     # Uplink 신틸레이션 손실 [dB] (마진 모드에서만 적용)
    h_uplink: float          # h_pu * h_lgu * h_aug * h_orientation

    # MRR 계수 (단순화: 변조만)
    h_modulation: float      # 변조 효율 M (0~1)
    h_MRR: float             # = h_modulation (변조 효율만)

    # Downlink 계수
    h_pg: float              # Downlink 수광 계수 (beam profile + rx config 통합)
    h_lgu_down: float        # Downlink 대기 감쇠
    h_agu: float             # Downlink 난류 페이딩 (마진 모드: <1, 평균 모드: =1)
    sigma_r2_downlink: float # Downlink Rytov variance
    L_scint_down_dB: float   # Downlink 신틸레이션 손실 [dB] (마진 모드에서만 적용)
    h_downlink: float        # h_pg * h_lgu_down * h_agu

    # 수신 광학계 (Antenna Model L_rx_optics에 대응)
    eta_rx: float            # 수신 광학계 효율 (0~1)
    eta_optics: float        # eta_tx * eta_rx (총 광학계 효율)

    # 전체 채널 계수
    h_total: float           # h_uplink * h_MRR * h_downlink * eta_optics

    # 전력 [W]
    P_tx_W: float
    P_rx_W: float

    @property
    def h_total_dB(self) -> float:
        """전체 채널 계수 [dB]"""
        if self.h_total <= 0:
            return float('-inf')
        import numpy as np
        return 10.0 * np.log10(self.h_total)

    @property
    def P_rx_dBm(self) -> float:
        """수신 전력 [dBm]"""
        if self.P_rx_W <= 0:
            return float('-inf')
        from src.models.common import W_to_dBm
        return W_to_dBm(self.P_rx_W)

    def to_dict(self) -> dict[str, float]:
        """채널 계수 딕셔너리 반환"""
        return {
            # 송신 광학계
            "η_tx (tx optics)": self.eta_tx,
            # Uplink (h_orientation 포함)
            "h_geometric (uplink geom)": self.h_geometric,
            "h_tracking (pointing error)": self.h_tracking,
            "h_pu (geom×tracking)": self.h_pu,
            "h_lgu (uplink atm)": self.h_lgu,
            "h_aug (uplink turb)": self.h_aug,
            "h_orientation (UAV attitude)": self.h_orientation,
            "σ_R² (uplink)": self.sigma_r2_uplink,
            "L_scint_up [dB]": self.L_scint_up_dB,
            "h_uplink": self.h_uplink,
            # MRR (변조만)
            "h_modulation (= h_MRR)": self.h_modulation,
            # Downlink
            "h_pg (downlink collection)": self.h_pg,
            "h_lgu_down (downlink atm)": self.h_lgu_down,
            "h_agu (downlink turb)": self.h_agu,
            "σ_R² (downlink)": self.sigma_r2_downlink,
            "L_scint_down [dB]": self.L_scint_down_dB,
            "h_downlink": self.h_downlink,
            # 수신 광학계
            "η_rx (rx optics)": self.eta_rx,
            "η_optics (total optics)": self.eta_optics,
            # Total
            "h_total": self.h_total,
            "h_total [dB]": self.h_total_dB,
            "P_tx [W]": self.P_tx_W,
            "P_rx [W]": self.P_rx_W,
            "P_rx [dBm]": self.P_rx_dBm,
        }
