"""
확률 분포 그래프 (PDF, CDF)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional


def plot_channel_coefficient_pdf(
    h_samples: np.ndarray,
    title: str = "Channel Coefficient Distribution",
    bins: int = 100,
    show_theoretical: bool = False,
    sigma_r2: Optional[float] = None
) -> go.Figure:
    """
    채널 계수 PDF 히스토그램

    Parameters:
        h_samples: 채널 계수 샘플
        title: 차트 제목
        bins: 히스토그램 빈 수
        show_theoretical: 이론적 PDF 오버레이 여부
        sigma_r2: Rytov variance (이론 PDF용)

    Returns:
        Plotly Figure
    """
    # 유효한 샘플만 사용
    valid_samples = h_samples[h_samples > 0]

    fig = go.Figure()

    # 히스토그램
    fig.add_trace(go.Histogram(
        x=valid_samples,
        nbinsx=bins,
        histnorm='probability density',
        name='Simulated',
        marker_color='rgba(55, 128, 191, 0.6)',
        opacity=0.7
    ))

    # 이론적 PDF (Log-Normal, 약한 난류)
    if show_theoretical and sigma_r2 is not None and sigma_r2 < 1.0:
        x_theory = np.linspace(0.01, np.percentile(valid_samples, 99), 200)
        # Log-Normal: σ_I² = exp(σ_R²) - 1 ≈ σ_R² for weak turbulence
        sigma_I = np.sqrt(sigma_r2)
        # PDF: (1/(h*σ_I*sqrt(2π))) * exp(-(ln(h)+σ_I²/2)² / (2*σ_I²))
        mu_ln = -sigma_I**2 / 2  # E[h] = 1
        pdf_theory = (1 / (x_theory * sigma_I * np.sqrt(2*np.pi))) * \
                     np.exp(-(np.log(x_theory) - mu_ln)**2 / (2 * sigma_I**2))

        fig.add_trace(go.Scatter(
            x=x_theory,
            y=pdf_theory,
            mode='lines',
            name='Log-Normal Theory',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Channel Coefficient h",
        yaxis_title="Probability Density",
        height=400,
        showlegend=True
    )

    return fig


def plot_channel_coefficient_cdf(
    h_samples: np.ndarray,
    title: str = "Channel Coefficient CDF",
    threshold_values: Optional[list[float]] = None
) -> go.Figure:
    """
    채널 계수 CDF

    Parameters:
        h_samples: 채널 계수 샘플
        title: 차트 제목
        threshold_values: 마커로 표시할 임계값 리스트

    Returns:
        Plotly Figure
    """
    valid_samples = h_samples[h_samples > 0]
    sorted_h = np.sort(valid_samples)
    cdf = np.arange(1, len(sorted_h) + 1) / len(sorted_h)

    fig = go.Figure()

    # CDF
    fig.add_trace(go.Scatter(
        x=sorted_h,
        y=cdf,
        mode='lines',
        name='CDF',
        line=dict(color='blue', width=2)
    ))

    # 임계값 마커
    if threshold_values:
        for thresh in threshold_values:
            idx = np.searchsorted(sorted_h, thresh)
            if idx < len(cdf):
                prob = cdf[idx]
                fig.add_trace(go.Scatter(
                    x=[thresh],
                    y=[prob],
                    mode='markers+text',
                    name=f'h={thresh:.3f}',
                    marker=dict(size=10, color='red'),
                    text=[f'{prob:.2%}'],
                    textposition='top right'
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Channel Coefficient h",
        yaxis_title="CDF P(H < h)",
        height=400,
        showlegend=True
    )

    return fig


def plot_received_power_distribution(
    P_rx_dBm_samples: np.ndarray,
    sensitivity_dBm: float,
    title: str = "Received Power Distribution",
    bins: int = 100
) -> go.Figure:
    """
    수신 전력 분포 히스토그램

    Parameters:
        P_rx_dBm_samples: 수신 전력 샘플 [dBm]
        sensitivity_dBm: 수신 감도 [dBm]
        title: 차트 제목
        bins: 히스토그램 빈 수

    Returns:
        Plotly Figure
    """
    valid_samples = P_rx_dBm_samples[np.isfinite(P_rx_dBm_samples)]

    fig = go.Figure()

    # 히스토그램
    fig.add_trace(go.Histogram(
        x=valid_samples,
        nbinsx=bins,
        histnorm='probability density',
        name='P_rx Distribution',
        marker_color='rgba(55, 128, 191, 0.6)',
        opacity=0.7
    ))

    # 수신 감도 라인
    fig.add_vline(
        x=sensitivity_dBm,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Sensitivity: {sensitivity_dBm} dBm",
        annotation_position="top right"
    )

    # Outage 영역 표시
    fig.add_vrect(
        x0=valid_samples.min() if len(valid_samples) > 0 else sensitivity_dBm - 20,
        x1=sensitivity_dBm,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Outage",
        annotation_position="top left"
    )

    # 통계 표시
    mean_power = np.mean(valid_samples)
    std_power = np.std(valid_samples)
    outage_prob = np.sum(P_rx_dBm_samples < sensitivity_dBm) / len(P_rx_dBm_samples)

    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=f"Mean: {mean_power:.1f} dBm<br>Std: {std_power:.1f} dB<br>Outage: {outage_prob:.2%}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title=title,
        xaxis_title="Received Power [dBm]",
        yaxis_title="Probability Density",
        height=400,
        showlegend=True
    )

    return fig


def plot_fading_comparison(
    h_uplink: np.ndarray,
    h_downlink: np.ndarray,
    h_total: np.ndarray,
    title: str = "Fading Components Comparison"
) -> go.Figure:
    """
    페이딩 컴포넌트 비교 (Uplink, Downlink, Total)

    Parameters:
        h_uplink: Uplink 페이딩 샘플
        h_downlink: Downlink 페이딩 샘플
        h_total: 전체 페이딩 샘플
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Uplink Fading", "Downlink Fading", "Total Fading"]
    )

    # Uplink
    fig.add_trace(
        go.Histogram(
            x=h_uplink[h_uplink > 0],
            nbinsx=50,
            histnorm='probability density',
            name='Uplink',
            marker_color='rgba(55, 128, 191, 0.6)'
        ),
        row=1, col=1
    )

    # Downlink
    fig.add_trace(
        go.Histogram(
            x=h_downlink[h_downlink > 0],
            nbinsx=50,
            histnorm='probability density',
            name='Downlink',
            marker_color='rgba(50, 171, 96, 0.6)'
        ),
        row=1, col=2
    )

    # Total
    fig.add_trace(
        go.Histogram(
            x=h_total[h_total > 0],
            nbinsx=50,
            histnorm='probability density',
            name='Total',
            marker_color='rgba(219, 64, 82, 0.6)'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title=title,
        height=350,
        showlegend=False
    )

    fig.update_xaxes(title_text="h", row=1, col=1)
    fig.update_xaxes(title_text="h", row=1, col=2)
    fig.update_xaxes(title_text="h", row=1, col=3)

    return fig


def plot_outage_vs_threshold(
    P_rx_dBm_samples: np.ndarray,
    threshold_range: Optional[tuple[float, float]] = None,
    title: str = "Outage Probability vs Threshold"
) -> go.Figure:
    """
    임계값에 따른 outage probability

    Parameters:
        P_rx_dBm_samples: 수신 전력 샘플 [dBm]
        threshold_range: (min, max) 임계값 범위 [dBm]
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    valid_samples = P_rx_dBm_samples[np.isfinite(P_rx_dBm_samples)]

    if threshold_range is None:
        threshold_range = (np.percentile(valid_samples, 1), np.percentile(valid_samples, 99))

    thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
    outage_probs = [np.mean(valid_samples < t) for t in thresholds]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=outage_probs,
        mode='lines',
        name='Outage Probability',
        line=dict(color='blue', width=2)
    ))

    # 일반적인 목표 outage 라인
    for target in [0.01, 0.05, 0.10]:
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{target:.0%}",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Threshold [dBm]",
        yaxis_title="Outage Probability",
        yaxis_type="log",
        height=400,
        showlegend=True
    )

    return fig


def plot_ber_vs_snr(
    snr_dB_range: tuple[float, float] = (-10, 30),
    modulation: str = "OOK",
    title: str = "BER vs SNR"
) -> go.Figure:
    """
    SNR에 따른 BER 곡선

    Parameters:
        snr_dB_range: SNR 범위 [dB]
        modulation: 변조 방식 ("OOK" or "BPSK")
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    from scipy.special import erfc

    snr_dB = np.linspace(snr_dB_range[0], snr_dB_range[1], 200)
    snr_linear = 10 ** (snr_dB / 10)

    fig = go.Figure()

    if modulation.upper() == "OOK":
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
        name = "OOK"
    else:  # BPSK
        ber = 0.5 * erfc(np.sqrt(snr_linear))
        name = "BPSK"

    fig.add_trace(go.Scatter(
        x=snr_dB,
        y=ber,
        mode='lines',
        name=name,
        line=dict(width=2)
    ))

    # 목표 BER 라인
    for target_ber in [1e-3, 1e-6, 1e-9]:
        fig.add_hline(
            y=target_ber,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"BER={target_ber:.0e}",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="SNR [dB]",
        yaxis_title="BER",
        yaxis_type="log",
        height=400,
        showlegend=True
    )

    return fig
