"""
MRR Link Budget Simulator - Streamlit Application

MQW 기반 Modulating Retro-Reflector FSO 통신 링크 버짓 시뮬레이터
"""
import streamlit as st
import numpy as np
import pandas as pd

from src.models.antenna_model import AntennaModelParams, calculate_antenna_link_budget
from src.models.optical_model import OpticalModelParams, calculate_optical_channel_coefficient, optical_to_antenna_comparison
from src.models.common import dBm_to_W, W_to_dBm
from src.simulation.monte_carlo import MonteCarloConfig, run_monte_carlo_antenna
from src.simulation.parameter_sweep import (
    sweep_distance, sweep_visibility, sweep_orientation_error,
    sweep_tracking_error, sweep_mrr_diameter, sweep_divergence,
    sweep_distance_optical, sweep_visibility_optical, sweep_orientation_error_optical,
    sweep_tracking_error_optical, sweep_mrr_diameter_optical, sweep_divergence_optical
)
from src.visualization.waterfall import (
    create_uplink_waterfall, create_downlink_waterfall,
    create_full_link_waterfall
)
from src.visualization.pdf_plots import (
    plot_channel_coefficient_pdf, plot_received_power_distribution,
    plot_fading_comparison, plot_outage_vs_threshold
)
from src.visualization.sweep_plots import (
    plot_sweep_link_margin, plot_sweep_with_mc
)

# 페이지 설정
st.set_page_config(
    page_title="MRR Link Budget Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 MRR Link Budget Simulator")
st.markdown("MQW/CCR 기반 Modulating Retro-Reflector FSO 통신 링크 버짓 분석 도구")


# === 사이드바: 파라미터 입력 ===
with st.sidebar:
    st.header("Parameters")

    # === 모델 선택 ===
    st.subheader("Model Selection")
    model_type = st.radio(
        "Analysis Model",
        ["Antenna Model (dB)", "Optical Model (Linear)", "Both (Comparison)"],
        index=0,
        help="Antenna: dB 기반 링크 버짓 / Optical: Gaussian 빔 전파 채널 계수"
    )
    st.divider()

    # === 송신부 ===
    with st.expander("📡 Transmitter", expanded=True):
        P_tx_dBm = st.number_input("TX Power [dBm]", 0.0, 50.0, 30.0, 1.0)
        wavelength_nm = st.selectbox("Wavelength [nm]", [850, 1310, 1550], index=2)
        theta_div_full_urad = st.number_input("Full Divergence θ [μrad]", 10.0, 10000.0, 1000.0, 10.0,
                                        help="Full-angle beam divergence (e⁻² diameter)")
        theta_div_full_mrad = theta_div_full_urad / 1000.0

        use_tx_diameter = st.checkbox("TX Beam Diameter 설정")
        tx_beam_diameter_mm = None
        if use_tx_diameter:
            tx_beam_diameter_mm = st.number_input("TX Beam Diameter [mm]", 1.0, 100.0, 10.0, 1.0)
        L_tx_optics_dB = st.number_input("TX Optics Loss [dB]", 0.0, 10.0, 2.0, 0.5)

    # === 링크 ===
    with st.expander("🔗 Link Parameters", expanded=True):
        distance_m = st.number_input("Distance [m]", 100, 10000, 1000, 50)
        visibility_km = st.number_input("Visibility [km]", 0.1, 100.0, 10.0, 0.5)
        use_fog = st.checkbox("Fog/Smoke Condition")
        fog_condition = None
        if use_fog:
            fog_condition = st.selectbox("Condition Type", ["fog", "smoke"])

    # === Reflector ===
    with st.expander("🔲 Reflector Parameters", expanded=True):
        # Reflector Type 선택
        reflector_type_label = st.radio(
            "Reflector Type",
            ["MRR (MQW Modulation)", "CCR (Corner Cube)"],
            horizontal=True,
            help="MRR: MQW 변조 기반 / CCR: Corner Cube 3회 반사 기반"
        )
        is_ccr = reflector_type_label.startswith("CCR")

        mrr_diameter_mm = st.number_input("Reflector Diameter [mm]", 1.0, 50.0, 20.0, 1.0)
        mrr_diameter_cm = mrr_diameter_mm / 10.0

        sigma_orientation_deg = st.number_input("Orientation σ [deg]", 0.0, 15.0, 3.0, 0.01,
                                               help="UAV attitude fluctuation standard deviation")

        if is_ccr:
            # === CCR 전용 파라미터 ===
            st.markdown("**CCR Settings**")
            col_ccr1, col_ccr2 = st.columns(2)
            with col_ccr1:
                ccr_surface_reflectivity = st.number_input(
                    "Surface Reflectivity", 0.90, 1.00, 0.99, 0.01,
                    help="CCR 표면 반사율 (3회 반사 적용)"
                )
            with col_ccr2:
                ccr_m2_value = st.number_input(
                    "CCR M²", 1.0, 3.0, 1.05, 0.05,
                    help="CCR 빔 품질 인자 (고정값, 각도 비의존)"
                )

            # 계산된 값 표시
            from src.mrr.efficiency import ccr_mean_h_mrr, ccr_orientation_loss_dB
            from src.mrr.modulation import ccr_passive_loss_dB
            mean_h = ccr_mean_h_mrr(sigma_orientation_deg)
            clip_loss = ccr_orientation_loss_dB(sigma_orientation_deg)
            passive_loss = ccr_passive_loss_dB(ccr_surface_reflectivity)
            bounce_loss = -10*np.log10(ccr_surface_reflectivity**3)
            st.info(
                f"**CCR Computed Values:**\n"
                f"- 3-Bounce Loss: {bounce_loss:.3f} dB (reflectivity³ = {ccr_surface_reflectivity**3:.6f})\n"
                f"- Mean h_MRR: {mean_h:.4f} ({clip_loss:.2f} dB geometric clipping)\n"
                f"- Total Passive Loss: {passive_loss:.3f} dB (3-bounce + AR)"
            )

            # MQW 파라미터는 CCR에서 사용하지 않으므로 기본값 설정
            use_mqw_params = False
            alpha_off = 0.1
            c_mqw = 3.0
            modulation_efficiency = 0.5
            use_strehl = False
            strehl_ratio = 0.8
            mrr_m2_min = 1.3
            mrr_m2_max = 3.4
            mrr_knee_deg = 2.12
            mrr_max_deg = 3.2
            mrr_ar_coating_loss = 0.5
            mrr_passive_loss = 0.3

        else:
            # === MRR 전용 파라미터 (기존 동일) ===
            ccr_surface_reflectivity = 0.99
            ccr_m2_value = 1.05

            # MQW 변조 파라미터
            st.markdown("**Modulation Settings**")
            use_mqw_params = st.checkbox("Use MQW Parameters (α_off, C_MQW)")

            if use_mqw_params:
                col_mqw1, col_mqw2 = st.columns(2)
                with col_mqw1:
                    alpha_off = st.number_input("α_off (OFF absorption)", 0.01, 1.0, 0.1, 0.01,
                                               help="OFF state absorption coefficient")
                with col_mqw2:
                    c_mqw = st.number_input("C_MQW (Contrast ratio)", 1.1, 10.0, 3.0, 0.1,
                                           help="Modulation contrast ratio = T_on/T_off")
                calc_mod_eff = np.exp(-alpha_off) * (c_mqw - 1)
                if calc_mod_eff > 0:
                    st.info(f"M = exp(-α_off) × (C_MQW - 1) = **{calc_mod_eff:.4f}** ({-10*np.log10(calc_mod_eff):.2f} dB)")
                modulation_efficiency = calc_mod_eff
            else:
                modulation_efficiency = st.number_input("Modulation Efficiency M", 0.1, 1.0, 0.5, 0.05)
                alpha_off = 0.1
                c_mqw = 3.0

            # M² / Strehl 입력 방식 선택
            st.markdown("**M² (Beam Quality)**")
            use_strehl = st.radio(
                "Select input type",
                ["M² (직접 입력)", "Strehl Ratio"],
                horizontal=True,
                help="M² = 1/√Strehl"
            ) == "Strehl Ratio"

            col_m2_1, col_m2_2 = st.columns(2)
            with col_m2_1:
                if use_strehl:
                    strehl_ratio = st.number_input("Strehl Ratio", 0.1, 1.0, 0.8, 0.05,
                                                   help="빔 품질 (1.0 = 완벽한 빔)")
                    calc_m2 = 1.0 / np.sqrt(strehl_ratio)
                    st.info(f"M² = 1/√{strehl_ratio:.2f} = **{calc_m2:.2f}**")
                    mrr_m2_min = calc_m2
                else:
                    mrr_m2_min = st.number_input("M² Min", 1.0, 3.0, 1.3, 0.1,
                                                 help="최소 M² 값 (angle=0°)")
                    strehl_ratio = 0.8
            with col_m2_2:
                mrr_m2_max = st.number_input("M² Max", 2.0, 5.0, 3.4, 0.1,
                                             help="최대 M² 값 (angle=max°)")

            with st.expander("Advanced MRR Settings"):
                mrr_knee_deg = st.number_input("Knee Angle [deg]", 0.5, 5.0, 2.12, 0.1)
                mrr_max_deg = st.number_input("Max Angle [deg]", 1.0, 10.0, 3.2, 0.1)
                mrr_ar_coating_loss = st.number_input("AR Coating Loss [dB]", 0.0, 2.0, 0.5, 0.1)
                mrr_passive_loss = st.number_input("Passive Loss [dB]", 0.0, 2.0, 0.3, 0.1)

    # === 수신부 ===
    with st.expander("📶 Receiver", expanded=True):
        rx_diameter_cm = st.number_input("RX Diameter [cm]", 1.0, 50.0, 16.0, 0.5)
        L_rx_optics_dB = st.number_input("RX Optics Loss [dB]", 0.0, 10.0, 3.0, 0.5)
        receiver_sensitivity_dBm = st.number_input("Sensitivity [dBm]", -60.0, -20.0, -40.0, 1.0)

        # 빔 프로파일 선택
        st.markdown("**Beam Profile**")
        beam_profile = st.radio(
            "Downlink Beam Profile",
            ["uniform", "gaussian", "airy"],
            index=0,
            horizontal=True,
            help="Uniform: Top-hat / Gaussian: exp(-2r²/w²) / Airy: [2J₁(x)/x]²"
        )

        rx_config = st.radio("RX Configuration", ["bistatic", "concentric"])
        if rx_config == "bistatic":
            rx_offset_cm = st.number_input("TX-RX Offset [cm]", 0.0, 100.0, 10.0, 1.0)
            tx_inner_diameter_cm = 0.0
        else:
            rx_offset_cm = 0.0
            tx_inner_diameter_cm = st.number_input("TX Inner Diameter [cm]", 0.1, 10.0, 5.0, 0.5)

    # === 추적 ===
    with st.expander("🎯 Tracking", expanded=True):
        tracking_offset_deg = st.number_input(
            "Tracking Offset [deg]", 0.0, 5.0, 0.0057, 0.001, format="%.4f",
            help="지향 오차 각도. offset = angle × distance"
        )
        tracking_offset_urad = tracking_offset_deg * 17453.3  # deg → μrad

    # === 난류 ===
    with st.expander("🌫️ Turbulence", expanded=True):
        use_altitude_profile = st.checkbox("Use Altitude Profile (HV Model)")
        if use_altitude_profile:
            h_gs_m = st.number_input("GS Altitude [m]", 0.0, 1000.0, 0.0, 10.0)
            h_uav_m = st.number_input("UAV Altitude [m]", 10.0, 1000.0, 100.0, 10.0)
            wind_speed_ms = st.number_input("Wind Speed [m/s]", 0.0, 50.0, 21.0, 1.0)
            cn2_constant = 5e-15
        else:
            cn2_exp = st.number_input("Cn² Exponent (10^x)", -17, -12, -15, 1)
            cn2_constant = 10 ** cn2_exp
            h_gs_m = 0.0
            h_uav_m = 100.0
            wind_speed_ms = 21.0
        use_turbulence = st.checkbox(
            "Apply Turbulence Loss",
            value=True,
            help="난류 손실 적용 (σ_R² < 1: Log-Normal, ≥ 1: Gamma-Gamma)"
        )
        scint_probability = st.number_input("Scintillation Probability", 0.500, 0.999, 0.990, 0.001, format="%.3f")

    # === Monte Carlo ===
    with st.expander("🎲 Monte Carlo", expanded=False):
        enable_mc = st.checkbox("Enable MC Simulation")
        mc_samples = st.selectbox("Sample Count", [1000, 10000, 100000], index=1)


# === 파라미터 객체 생성 ===
# Antenna Model 파라미터
antenna_params = AntennaModelParams(
    P_tx_dBm=P_tx_dBm,
    L_tx_optics_dB=L_tx_optics_dB,
    wavelength_nm=wavelength_nm,
    theta_div_full_mrad=theta_div_full_mrad,
    tx_beam_diameter_mm=tx_beam_diameter_mm,
    distance_m=distance_m,
    visibility_km=visibility_km,
    fog_condition=fog_condition,
    mrr_diameter_cm=mrr_diameter_cm,
    sigma_orientation_deg=sigma_orientation_deg,
    mrr_knee_deg=mrr_knee_deg,
    mrr_max_deg=mrr_max_deg,
    mrr_m2_min=mrr_m2_min,
    mrr_m2_max=mrr_m2_max,
    use_strehl=use_strehl,
    strehl_ratio=strehl_ratio,
    use_mqw_params=use_mqw_params,
    alpha_off=alpha_off,
    c_mqw=c_mqw,
    modulation_efficiency=modulation_efficiency,
    mrr_ar_coating_loss_dB=mrr_ar_coating_loss,
    mrr_passive_loss_dB=mrr_passive_loss,
    rx_diameter_cm=rx_diameter_cm,
    L_rx_optics_dB=L_rx_optics_dB,
    rx_config=rx_config,
    rx_offset_cm=rx_offset_cm,
    tx_inner_diameter_cm=tx_inner_diameter_cm,
    beam_profile=beam_profile,
    receiver_sensitivity_dBm=receiver_sensitivity_dBm,
    tracking_offset_urad=tracking_offset_urad,
    use_turbulence=use_turbulence,
    use_altitude_profile=use_altitude_profile,
    cn2_constant=cn2_constant,
    h_gs_m=h_gs_m,
    h_uav_m=h_uav_m,
    wind_speed_ms=wind_speed_ms,
    scint_probability=scint_probability,
    reflector_type="ccr" if is_ccr else "mrr",
    ccr_surface_reflectivity=ccr_surface_reflectivity,
    ccr_m2=ccr_m2_value,
)

# Optical Model 파라미터
optical_params = OpticalModelParams(
    P_tx_W=dBm_to_W(P_tx_dBm),
    eta_tx=10**(-L_tx_optics_dB/10),  # dB loss to efficiency
    wavelength_nm=wavelength_nm,
    theta_div_full_mrad=theta_div_full_mrad,
    tx_beam_diameter_mm=tx_beam_diameter_mm,
    distance_m=distance_m,
    visibility_km=visibility_km,
    fog_condition=fog_condition,
    mrr_diameter_cm=mrr_diameter_cm,
    sigma_orientation_deg=sigma_orientation_deg,
    mrr_knee_deg=mrr_knee_deg,
    mrr_max_deg=mrr_max_deg,
    mrr_m2_min=mrr_m2_min,
    mrr_m2_max=mrr_m2_max,
    use_strehl=use_strehl,
    strehl_ratio=strehl_ratio,
    use_mqw_params=use_mqw_params,
    alpha_off=alpha_off,
    c_mqw=c_mqw,
    modulation_efficiency=modulation_efficiency,

    rx_diameter_cm=rx_diameter_cm,
    eta_rx=10**(-L_rx_optics_dB/10),
    rx_config=rx_config,
    rx_offset_cm=rx_offset_cm,
    tx_inner_diameter_cm=tx_inner_diameter_cm,
    beam_profile=beam_profile,
    tracking_offset_urad=tracking_offset_urad,
    use_altitude_profile=use_altitude_profile,
    cn2_constant=cn2_constant,
    h_gs_m=h_gs_m,
    h_uav_m=h_uav_m,
    wind_speed_ms=wind_speed_ms,
    use_turbulence=use_turbulence,
    scint_probability=scint_probability,
    reflector_type="ccr" if is_ccr else "mrr",
    ccr_surface_reflectivity=ccr_surface_reflectivity,
    ccr_m2=ccr_m2_value,
)


# === 모델 계산 ===
antenna_result = None
optical_result = None

if model_type in ["Antenna Model (dB)", "Both (Comparison)"]:
    antenna_result = calculate_antenna_link_budget(antenna_params)

if model_type in ["Optical Model (Linear)", "Both (Comparison)"]:
    optical_result = calculate_optical_channel_coefficient(optical_params)


# === 메인 영역: 탭 구조 ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Link Budget",
    "📉 Waterfall Chart",
    "🎲 Monte Carlo",
    "📈 Parameter Sweep",
    "📋 Detailed Breakdown"
])


# === Tab 1: Link Budget Analysis ===
with tab1:
    st.header("Link Budget Analysis")

    if model_type == "Both (Comparison)":
        optical_comparison = optical_to_antenna_comparison(optical_result, receiver_sensitivity_dBm)

        # 요약 메트릭 비교
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            margin_diff = antenna_result.link_margin_dB - optical_comparison['link_margin_dB']
            st.metric("Link Margin Diff", f"{margin_diff:.2f} dB",
                      help="Antenna - Optical")
        with col2:
            st.metric("Antenna P_rx", f"{antenna_result.receiver_power_dBm:.2f} dBm")
        with col3:
            st.metric("Optical P_rx", f"{optical_comparison['P_rx_dBm']:.2f} dBm")

        st.divider()

        # Uplink 비교
        st.subheader("Uplink Budget Comparison")
        # Optical P_mrr_in 계산 (새 필드 사용)
        optical_P_mrr_in = (optical_comparison['P_tx_dBm']
                           + optical_comparison['eta_tx_dB']
                           + optical_comparison['h_geometric_dB']
                           + optical_comparison['h_lgu_dB']
                           - optical_result.L_scint_up_dB
                           + optical_comparison['h_tracking_dB']
                           + optical_comparison['h_orientation_dB'])
        uplink_compare = {
            "Parameter": [
                "P_tx [dBm]",
                "L_tx_optics / η_tx [dB]",
                "G_tx / h_geometric [dB]",
                "L_FSL [dB]",
                "L_atm / h_lgu [dB]",
                "L_scint [dB]",
                "L_tracking / h_tracking [dB]",
                "L_orientation / h_orientation [dB]",
                "G_rx_mrr [dB]",
                "P_mrr_in [dBm]",
            ],
            "Antenna": [
                f"{antenna_result.uplink.P_tx_dBm:.2f}",
                f"{-antenna_result.uplink.L_tx_optics_dB:.2f}",
                f"{antenna_result.uplink.G_tx_dB:.2f}",
                f"{antenna_result.uplink.L_FSL_dB:.2f}",
                f"{-antenna_result.uplink.L_atm_dB:.2f}",
                f"{-antenna_result.uplink.L_scint_dB:.2f}",
                f"{-antenna_result.uplink.L_tracking_dB:.2f}",
                f"{-antenna_result.uplink.L_orientation_dB:.2f}",
                f"{antenna_result.uplink.G_rx_mrr_dB:.2f}",
                f"{antenna_result.uplink.P_mrr_in_dBm:.2f}",
            ],
            "Optical": [
                f"{optical_comparison['P_tx_dBm']:.2f}",
                f"{optical_comparison['eta_tx_dB']:.2f}",
                f"{optical_comparison['h_geometric_dB']:.2f}",
                "-",
                f"{optical_comparison['h_lgu_dB']:.2f}",
                f"{-optical_result.L_scint_up_dB:.2f}",
                f"{optical_comparison['h_tracking_dB']:.2f}",
                f"{optical_comparison['h_orientation_dB']:.2f}",
                "-",
                f"{optical_P_mrr_in:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(uplink_compare), use_container_width=True, hide_index=True)

        st.divider()

        # Downlink 비교
        st.subheader("Downlink Budget Comparison")
        downlink_compare = {
            "Parameter": [
                "P_mrr_in [dBm]",
                "G_mrr_rx [dB]",
                "L_modulation / h_MRR [dB]",
                "L_FSL [dB]",
                "L_atm / h_lgu [dB]",
                "L_scint [dB]",
                "G_rx_gs / h_pg [dB]",
                "L_rx_config [dB]",
                "L_rx_optics / η_rx [dB]",
                "P_rx [dBm]",
            ],
            "Antenna": [
                f"{antenna_result.uplink.P_mrr_in_dBm:.2f}",
                f"{antenna_result.downlink.G_mrr_rx_dB:.2f}",
                f"{-antenna_result.downlink.L_modulation_dB:.2f}",
                f"{antenna_result.downlink.L_FSL_dB:.2f}",
                f"{-antenna_result.downlink.L_atm_dB:.2f}",
                f"{-antenna_result.downlink.L_scint_dB:.2f}",
                f"{antenna_result.downlink.G_rx_gs_dB:.2f}",
                f"{-antenna_result.downlink.L_rx_config_dB:.2f}",
                f"{-antenna_result.downlink.L_rx_optics_dB:.2f}",
                f"{antenna_result.downlink.P_rx_dBm:.2f}",
            ],
            "Optical": [
                f"{optical_P_mrr_in:.2f}",
                "-",
                f"{optical_comparison['h_MRR_dB']:.2f}",
                "-",
                f"{optical_comparison['h_lgu_down_dB']:.2f}",
                f"{-optical_result.L_scint_down_dB:.2f}",
                f"{optical_comparison['h_pg_dB']:.2f}",
                "(in h_pg)",
                f"{optical_comparison['eta_rx_dB']:.2f}",
                f"{optical_comparison['P_rx_dBm']:.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(downlink_compare), use_container_width=True, hide_index=True)

    elif model_type == "Antenna Model (dB)":
        # 요약 메트릭
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Link Margin", f"{antenna_result.link_margin_dB:.2f} dB",
                      delta="✓" if antenna_result.link_margin_dB > 0 else "✗")
        with col2:
            st.metric("RX Power", f"{antenna_result.receiver_power_dBm:.2f} dBm")
        with col3:
            st.metric("Sensitivity", f"{receiver_sensitivity_dBm:.1f} dBm")
        with col4:
            st.metric("Distance", f"{distance_m} m")

        st.divider()

        # Uplink/Downlink 테이블
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uplink Budget")
            uplink_data = antenna_result.uplink.to_dict()
            df_uplink = pd.DataFrame({
                "Parameter": list(uplink_data.keys()),
                "Value": [f"{v:.2f}" for v in uplink_data.values()]
            })
            st.dataframe(df_uplink, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Downlink Budget")
            downlink_data = antenna_result.downlink.to_dict()
            df_downlink = pd.DataFrame({
                "Parameter": list(downlink_data.keys()),
                "Value": [f"{v:.2f}" for v in downlink_data.values()]
            })
            st.dataframe(df_downlink, use_container_width=True, hide_index=True)

    else:  # Optical Model
        # Optical Model 결과 (dB 변환)
        optical_comparison = optical_to_antenna_comparison(optical_result, receiver_sensitivity_dBm)

        # 요약 메트릭
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Link Margin", f"{optical_comparison['link_margin_dB']:.2f} dB",
                      delta="✓" if optical_comparison['link_margin_dB'] > 0 else "✗")
        with col2:
            st.metric("RX Power", f"{optical_comparison['P_rx_dBm']:.2f} dBm")
        with col3:
            st.metric("Sensitivity", f"{receiver_sensitivity_dBm:.1f} dBm")
        with col4:
            st.metric("Distance", f"{distance_m} m")

        st.divider()

        # Uplink/Downlink 테이블 (Antenna Model과 동일한 형식)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uplink Budget")
            # P_mrr_in 계산 (새 필드 사용)
            P_mrr_in_dBm = (optical_comparison['P_tx_dBm']
                           + optical_comparison['eta_tx_dB']
                           + optical_comparison['h_geometric_dB']
                           + optical_comparison['h_lgu_dB']
                           - optical_result.L_scint_up_dB
                           + optical_comparison['h_tracking_dB']
                           + optical_comparison['h_orientation_dB'])
            uplink_data = {
                "P_tx [dBm]": optical_comparison['P_tx_dBm'],
                "η_tx [dB]": optical_comparison['eta_tx_dB'],
                "h_geometric [dB]": optical_comparison['h_geometric_dB'],
                "h_lgu (atm) [dB]": optical_comparison['h_lgu_dB'],
                "L_scint [dB]": -optical_result.L_scint_up_dB,
                "h_tracking [dB]": optical_comparison['h_tracking_dB'],
                "h_orientation [dB]": optical_comparison['h_orientation_dB'],
                "P_mrr_in [dBm]": P_mrr_in_dBm,
            }
            df_uplink = pd.DataFrame({
                "Parameter": list(uplink_data.keys()),
                "Value": [f"{v:.2f}" for v in uplink_data.values()]
            })
            st.dataframe(df_uplink, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Downlink Budget")
            downlink_data = {
                "P_mrr_in [dBm]": P_mrr_in_dBm,
                "h_modulation (=h_MRR) [dB]": optical_comparison['h_MRR_dB'],
                "h_pg (collection) [dB]": optical_comparison['h_pg_dB'],
                "h_lgu (atm) [dB]": optical_comparison['h_lgu_down_dB'],
                "L_scint [dB]": -optical_result.L_scint_down_dB,
                "η_rx [dB]": optical_comparison['eta_rx_dB'],
                "P_rx [dBm]": optical_comparison['P_rx_dBm'],
            }
            df_downlink = pd.DataFrame({
                "Parameter": list(downlink_data.keys()),
                "Value": [f"{v:.2f}" for v in downlink_data.values()]
            })
            st.dataframe(df_downlink, use_container_width=True, hide_index=True)

        # 전체 채널 계수 요약
        st.divider()
        st.subheader("Total Channel Coefficient")
        total_col1, total_col2, total_col3 = st.columns(3)
        with total_col1:
            st.metric("h_total (linear)", f"{optical_result.h_total:.2e}")
        with total_col2:
            st.metric("h_total [dB]", f"{optical_comparison['h_total_dB']:.2f}")
        with total_col3:
            st.metric("P_rx [W]", f"{optical_result.P_rx_W:.2e}")

    # Beam 파라미터 (공통)
    st.divider()
    st.subheader("Beam Parameters")
    if antenna_result:
        beam = antenna_result.beam
    else:
        beam = optical_result.beam

    # TX Beam Diameter 정보 표시
    if beam.tx_beam_diameter_m is not None:
        st.info(f"📏 TX Beam Diameter: {beam.tx_beam_diameter_m*1000:.1f} mm → Beam @ MRR: {beam.beam_diameter_at_mrr_m*1000:.1f} mm")
    else:
        st.info(f"📏 TX Beam Diameter: Not set (far-field approximation) → Beam @ MRR: {beam.beam_diameter_at_mrr_m*1000:.1f} mm")

    beam_col1, beam_col2, beam_col3, beam_col4 = st.columns(4)
    with beam_col1:
        st.metric("Full Divergence", f"{beam.divergence_rad*1000:.2f} mrad")
    with beam_col2:
        st.metric("Beam @ MRR", f"{beam.beam_diameter_at_mrr_m:.3f} m")
    with beam_col3:
        st.metric("MRR M²", f"{beam.mrr_m2_factor:.2f}")
    with beam_col4:
        st.metric("Beam @ GS", f"{beam.beam_diameter_at_gs_m:.3f} m")


# === Tab 2: Waterfall Chart ===
with tab2:
    st.header("Waterfall Charts")

    if antenna_result:
        chart_type = st.radio(
            "Chart Type",
            ["Uplink", "Downlink", "Full Link"],
            horizontal=True
        )

        if chart_type == "Uplink":
            fig = create_uplink_waterfall(antenna_result.uplink.to_dict())
        elif chart_type == "Downlink":
            fig = create_downlink_waterfall(antenna_result.downlink.to_dict())
        else:
            fig = create_full_link_waterfall(
                antenna_result.uplink.to_dict(),
                antenna_result.downlink.to_dict(),
                receiver_sensitivity_dBm
            )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waterfall chart is only available for Antenna Model. Select 'Antenna Model (dB)' or 'Both (Comparison)'.")


# === Tab 3: Monte Carlo ===
with tab3:
    st.header("Monte Carlo Simulation")

    if enable_mc:
        if antenna_result is None:
            st.warning("Monte Carlo simulation requires Antenna Model. Please select 'Antenna Model (dB)' or 'Both (Comparison)'.")
        else:
            with st.spinner(f"Running Monte Carlo simulation ({mc_samples:,} samples)..."):
                mc_config = MonteCarloConfig(
                    n_samples=mc_samples,
                    seed=42,
                    include_pointing=True,
                    include_scintillation=True,
                    include_orientation=True
                )
                mc_result = run_monte_carlo_antenna(antenna_params, mc_config)

            # 통계 요약
            st.subheader("Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Mean P_rx", f"{mc_result.P_rx_mean_dBm:.2f} dBm")
            with stat_col2:
                st.metric("Std Dev", f"{mc_result.P_rx_std_dB:.2f} dB")
            with stat_col3:
                st.metric("Outage Prob", f"{mc_result.outage_probability:.2%}")
            with stat_col4:
                st.metric("h (1%)", f"{mc_result.h_percentile_1:.4f}")

            st.divider()

            # 분포 그래프
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Received Power Distribution")
                fig_prx = plot_received_power_distribution(
                    mc_result.P_rx_dBm_samples,
                    receiver_sensitivity_dBm
                )
                st.plotly_chart(fig_prx, use_container_width=True)

            with col2:
                st.subheader("Channel Coefficient Distribution")
                fig_h = plot_channel_coefficient_pdf(
                    mc_result.h_samples,
                    show_theoretical=True,
                    sigma_r2=mc_result.sigma_r2_uplink + mc_result.sigma_r2_downlink
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # 페이딩 컴포넌트
            st.subheader("Fading Components")
            fig_fading = plot_fading_comparison(
                mc_result.h_uplink_samples,
                mc_result.h_downlink_samples,
                mc_result.h_samples
            )
            st.plotly_chart(fig_fading, use_container_width=True)

            # Outage vs Threshold
            st.subheader("Outage vs Threshold")
            fig_outage = plot_outage_vs_threshold(mc_result.P_rx_dBm_samples)
            st.plotly_chart(fig_outage, use_container_width=True)

    else:
        st.info("Enable Monte Carlo simulation in the sidebar to see results.")


# === Tab 4: Parameter Sweep ===
with tab4:
    st.header("Parameter Sweep Analysis")

    if antenna_result is None and optical_result is None:
        st.warning("Parameter sweep requires a model result. Please run a calculation first.")
    else:
        sweep_param = st.selectbox(
            "Select Parameter to Sweep",
            ["Distance", "Visibility", "Orientation Error", "Tracking Error", "MRR Diameter", "Divergence"]
        )

        sweep_col1, sweep_col2 = st.columns(2)
        with sweep_col1:
            sweep_min = st.number_input("Min Value", value=100.0)
        with sweep_col2:
            sweep_max = st.number_input("Max Value", value=2000.0)

        sweep_points = st.number_input("Number of Points", 10, 100, 50)
        sweep_mc = st.checkbox("Include MC Analysis", value=False)

        if st.button("Run Sweep"):
            sweep_values = np.linspace(sweep_min, sweep_max, sweep_points)
            mc_config = MonteCarloConfig(n_samples=1000) if sweep_mc else None

            with st.spinner("Running parameter sweep..."):
                # Antenna Model sweep
                antenna_sweep_result = None
                if antenna_result is not None:
                    if sweep_param == "Distance":
                        antenna_sweep_result = sweep_distance(antenna_params, sweep_values, sweep_mc, mc_config)
                    elif sweep_param == "Visibility":
                        antenna_sweep_result = sweep_visibility(antenna_params, sweep_values, sweep_mc, mc_config)
                    elif sweep_param == "Orientation Error":
                        antenna_sweep_result = sweep_orientation_error(antenna_params, sweep_values, sweep_mc, mc_config)
                    elif sweep_param == "Tracking Error":
                        antenna_sweep_result = sweep_tracking_error(antenna_params, sweep_values, sweep_mc, mc_config)
                    elif sweep_param == "MRR Diameter":
                        antenna_sweep_result = sweep_mrr_diameter(antenna_params, sweep_values, sweep_mc, mc_config)
                    else:  # Divergence
                        antenna_sweep_result = sweep_divergence(antenna_params, sweep_values, sweep_mc, mc_config)

                # Optical Model sweep
                optical_sweep_result = None
                if optical_result is not None:
                    if sweep_param == "Distance":
                        optical_sweep_result = sweep_distance_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)
                    elif sweep_param == "Visibility":
                        optical_sweep_result = sweep_visibility_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)
                    elif sweep_param == "Orientation Error":
                        optical_sweep_result = sweep_orientation_error_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)
                    elif sweep_param == "Tracking Error":
                        optical_sweep_result = sweep_tracking_error_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)
                    elif sweep_param == "MRR Diameter":
                        optical_sweep_result = sweep_mrr_diameter_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)
                    else:  # Divergence
                        optical_sweep_result = sweep_divergence_optical(optical_params, sweep_values, receiver_sensitivity_dBm, sweep_mc, mc_config)

            # Plot results
            import plotly.graph_objects as go

            if model_type == "Both (Comparison)" and antenna_sweep_result and optical_sweep_result:
                # 두 모델 비교 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=antenna_sweep_result.parameter_values,
                    y=antenna_sweep_result.link_margin_dB,
                    mode='lines',
                    name='Antenna Model',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=optical_sweep_result.parameter_values,
                    y=optical_sweep_result.link_margin_dB,
                    mode='lines',
                    name='Optical Model',
                    line=dict(color='red')
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Link Margin = 0 dB")
                fig.update_layout(
                    title=f"Link Margin vs {sweep_param} (Model Comparison)",
                    xaxis_title=f"{sweep_param} [{antenna_sweep_result.parameter_unit}]",
                    yaxis_title="Link Margin [dB]",
                    legend=dict(x=0.02, y=0.98)
                )
                st.plotly_chart(fig, use_container_width=True)

                # 비교 테이블
                st.subheader("Sweep Results Comparison")
                sweep_df = pd.DataFrame({
                    f"{antenna_sweep_result.parameter_name} [{antenna_sweep_result.parameter_unit}]": antenna_sweep_result.parameter_values,
                    "Antenna Link Margin [dB]": antenna_sweep_result.link_margin_dB,
                    "Optical Link Margin [dB]": optical_sweep_result.link_margin_dB,
                    "Difference [dB]": antenna_sweep_result.link_margin_dB - optical_sweep_result.link_margin_dB
                })
                st.dataframe(sweep_df, use_container_width=True)

            else:
                # 단일 모델 결과
                sweep_result = antenna_sweep_result if antenna_sweep_result else optical_sweep_result
                model_name = "Antenna Model" if antenna_sweep_result else "Optical Model"

                if sweep_mc:
                    fig = plot_sweep_with_mc(sweep_result)
                else:
                    fig = plot_sweep_link_margin(sweep_result)

                st.plotly_chart(fig, use_container_width=True)

                # 결과 테이블
                st.subheader(f"Sweep Results ({model_name})")
                sweep_df = pd.DataFrame({
                    f"{sweep_result.parameter_name} [{sweep_result.parameter_unit}]": sweep_result.parameter_values,
                    "Link Margin [dB]": sweep_result.link_margin_dB,
                    "P_rx [dBm]": sweep_result.P_rx_dBm
                })
                if sweep_mc and sweep_result.outage_probability is not None:
                    sweep_df["Outage Probability"] = sweep_result.outage_probability
                st.dataframe(sweep_df, use_container_width=True)


# === Tab 5: Detailed Breakdown ===
with tab5:
    st.header("Detailed Link Budget Breakdown")

    if antenna_result:
        # Uplink 상세
        st.subheader("Uplink Components (Antenna Model)")
        st.markdown("""
        | Component | Formula | Value |
        |-----------|---------|-------|
        | TX Power | P_tx | {:.2f} dBm |
        | TX Optics Loss | L_tx | -{:.2f} dB |
        | TX Gain | G_tx = 32/θ² | {:.2f} dB |
        | Free Space Loss | L_FSL = 20·log₁₀(4πZ/λ) | {:.2f} dB |
        | Atmospheric | Kim/Ijaz model | -{:.2f} dB |
        | Scintillation | From Rytov variance | -{:.2f} dB |
        | Tracking Error | L_tr = 8.686·d_p²/w_z² | -{:.2f} dB |
        | Orientation | From η_mrr | -{:.2f} dB |
        | AR Coating | | -{:.2f} dB |
        | MRR Gain | G_mrr = 10·log₁₀(πD²/(λ²)) | {:.2f} dB |
        | **MRR Input** | | **{:.2f} dBm** |
        """.format(
            antenna_result.uplink.P_tx_dBm,
            antenna_result.uplink.L_tx_optics_dB,
            antenna_result.uplink.G_tx_dB,
            antenna_result.uplink.L_FSL_dB,
            antenna_result.uplink.L_atm_dB,
            antenna_result.uplink.L_scint_dB,
            antenna_result.uplink.L_tracking_dB,
            antenna_result.uplink.L_orientation_dB,
            antenna_result.uplink.L_AR_coating_dB,
            antenna_result.uplink.G_rx_mrr_dB,
            antenna_result.uplink.P_mrr_in_dBm
        ))

        st.divider()

        # Downlink 상세
        st.subheader("Downlink Components (Antenna Model)")
        modulation_label = "CCR Geometric Clipping" if is_ccr else "Modulation Loss"
        modulation_formula = "CCR h_MRR clipping" if is_ccr else "10·log₁₀(M)"
        st.markdown("""
        | Component | Formula | Value |
        |-----------|---------|-------|
        | MRR Input | From uplink | {:.2f} dBm |
        | MRR Gain | Retro-reflection | {:.2f} dB |
        | """ + modulation_label + """ | """ + modulation_formula + """ | -{:.2f} dB |
        | Passive Loss | | -{:.2f} dB |
        | Free Space Loss | L_FSL | {:.2f} dB |
        | Atmospheric | Same as uplink | -{:.2f} dB |
        | Scintillation | | -{:.2f} dB |
        | RX Gain | G_rx = 10·log₁₀(πD²/(λ²)) | {:.2f} dB |
        | RX Config Loss | Bistatic/Concentric | -{:.2f} dB |
        | RX Optics Loss | | -{:.2f} dB |
        | **RX Power** | | **{:.2f} dBm** |
        """.format(
            antenna_result.downlink.P_mrr_in_dBm,
            antenna_result.downlink.G_mrr_rx_dB,
            antenna_result.downlink.L_modulation_dB,
            antenna_result.downlink.L_mrr_passive_dB,
            antenna_result.downlink.L_FSL_dB,
            antenna_result.downlink.L_atm_dB,
            antenna_result.downlink.L_scint_dB,
            antenna_result.downlink.G_rx_gs_dB,
            antenna_result.downlink.L_rx_config_dB,
            antenna_result.downlink.L_rx_optics_dB,
            antenna_result.downlink.P_rx_dBm
        ))

    if optical_result:
        st.divider()
        st.subheader("Optical Model Channel Coefficients")

        def to_dB(val):
            return f"{10*np.log10(val):.2f}" if val > 0 else "-∞"

        st.markdown(f"""
        | Component | Value (Linear) | Value (dB) |
        |-----------|----------------|------------|
        | η_tx (TX Optics) | {optical_result.eta_tx:.6e} | {to_dB(optical_result.eta_tx)} |
        | h_geometric (Uplink Geom) | {optical_result.h_geometric:.6e} | {to_dB(optical_result.h_geometric)} |
        | h_tracking (Pointing) | {optical_result.h_tracking:.6e} | {to_dB(optical_result.h_tracking)} |
        | h_lgu (Uplink Atm) | {optical_result.h_lgu:.6e} | {to_dB(optical_result.h_lgu)} |
        | h_orientation (UAV Attitude) | {optical_result.h_orientation:.6e} | {to_dB(optical_result.h_orientation)} |
        | h_modulation (=h_MRR) | {optical_result.h_modulation:.6e} | {to_dB(optical_result.h_modulation)} |
        | h_pg (DL Collection) | {optical_result.h_pg:.6e} | {to_dB(optical_result.h_pg)} |
        | η_rx (RX Optics) | {optical_result.eta_rx:.6e} | {to_dB(optical_result.eta_rx)} |
        | **h_total** | **{optical_result.h_total:.6e}** | **{to_dB(optical_result.h_total)}** |
        """)

    st.divider()

    # 최종 요약
    st.subheader("Summary")
    if antenna_result:
        st.markdown(f"""
        **Antenna Model:**
        - Received Power: {antenna_result.receiver_power_dBm:.2f} dBm
        - Receiver Sensitivity: {receiver_sensitivity_dBm:.1f} dBm
        - Link Margin: {antenna_result.link_margin_dB:.2f} dB
        - Status: {"✅ Link Closed" if antenna_result.link_margin_dB > 0 else "❌ Link Not Closed"}
        """)

    if optical_result:
        optical_comparison = optical_to_antenna_comparison(optical_result, receiver_sensitivity_dBm)
        st.markdown(f"""
        **Optical Model:**
        - Received Power: {optical_comparison['P_rx_dBm']:.2f} dBm
        - h_total: {optical_result.h_total:.6e} ({optical_result.h_total_dB:.2f} dB)
        - Link Margin: {optical_comparison['link_margin_dB']:.2f} dB
        - Status: {"✅ Link Closed" if optical_comparison['link_margin_dB'] > 0 else "❌ Link Not Closed"}
        """)


# === Footer ===
st.divider()
st.caption("MRR Link Budget Simulator | MQW/CCR-based Modulating Retro-Reflector FSO Communication")
