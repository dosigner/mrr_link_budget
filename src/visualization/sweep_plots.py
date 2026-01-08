"""
Parameter Sweep 그래프
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

from ..simulation.parameter_sweep import SweepResult


def plot_sweep_link_margin(
    sweep_result: SweepResult,
    title: Optional[str] = None,
    margin_threshold: float = 0.0
) -> go.Figure:
    """
    파라미터 스윕 - 링크 마진 그래프

    Parameters:
        sweep_result: SweepResult 인스턴스
        title: 차트 제목
        margin_threshold: 마진 임계값 [dB]

    Returns:
        Plotly Figure
    """
    if title is None:
        title = f"Link Margin vs {sweep_result.parameter_name}"

    fig = go.Figure()

    # 링크 마진
    fig.add_trace(go.Scatter(
        x=sweep_result.parameter_values,
        y=sweep_result.link_margin_dB,
        mode='lines+markers',
        name='Link Margin',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    # 임계값 라인
    fig.add_hline(
        y=margin_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {margin_threshold} dB",
        annotation_position="right"
    )

    # 교차점 표시
    crossing = sweep_result.get_crossing_value("link_margin_dB", margin_threshold)
    if crossing is not None:
        fig.add_vline(
            x=crossing,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Critical: {crossing:.2f} {sweep_result.parameter_unit}",
            annotation_position="top"
        )

    fig.update_layout(
        title=title,
        xaxis_title=f"{sweep_result.parameter_name} [{sweep_result.parameter_unit}]",
        yaxis_title="Link Margin [dB]",
        height=400,
        showlegend=True
    )

    return fig


def plot_sweep_with_mc(
    sweep_result: SweepResult,
    title: Optional[str] = None
) -> go.Figure:
    """
    파라미터 스윕 - 링크 마진 + Outage probability

    Parameters:
        sweep_result: SweepResult 인스턴스 (MC 결과 포함)
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    if title is None:
        title = f"Link Margin & Outage vs {sweep_result.parameter_name}"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 링크 마진 (왼쪽 Y축)
    fig.add_trace(
        go.Scatter(
            x=sweep_result.parameter_values,
            y=sweep_result.link_margin_dB,
            mode='lines+markers',
            name='Link Margin',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )

    # Outage probability (오른쪽 Y축)
    if sweep_result.outage_probability is not None:
        fig.add_trace(
            go.Scatter(
                x=sweep_result.parameter_values,
                y=sweep_result.outage_probability,
                mode='lines+markers',
                name='Outage Probability',
                line=dict(color='red', width=2, dash='dash')
            ),
            secondary_y=True
        )

    fig.update_layout(
        title=title,
        height=450,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    fig.update_xaxes(title_text=f"{sweep_result.parameter_name} [{sweep_result.parameter_unit}]")
    fig.update_yaxes(title_text="Link Margin [dB]", secondary_y=False)
    fig.update_yaxes(title_text="Outage Probability", secondary_y=True, type="log")

    return fig


def plot_sweep_comparison(
    sweep_results: List[SweepResult],
    labels: List[str],
    title: str = "Parameter Comparison",
    metric: str = "link_margin_dB"
) -> go.Figure:
    """
    다중 스윕 결과 비교

    Parameters:
        sweep_results: SweepResult 리스트
        labels: 각 결과의 라벨
        title: 차트 제목
        metric: 비교할 메트릭 ("link_margin_dB", "P_rx_dBm", "outage_probability")

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, (result, label) in enumerate(zip(sweep_results, labels)):
        values = getattr(result, metric, None)
        if values is not None:
            fig.add_trace(go.Scatter(
                x=result.parameter_values,
                y=values,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

    # Y축 라벨
    y_labels = {
        "link_margin_dB": "Link Margin [dB]",
        "P_rx_dBm": "Received Power [dBm]",
        "outage_probability": "Outage Probability"
    }

    fig.update_layout(
        title=title,
        xaxis_title=f"{sweep_results[0].parameter_name} [{sweep_results[0].parameter_unit}]",
        yaxis_title=y_labels.get(metric, metric),
        height=450,
        showlegend=True
    )

    if metric == "outage_probability":
        fig.update_yaxes(type="log")

    return fig


def plot_2d_heatmap(
    param1_grid: np.ndarray,
    param2_grid: np.ndarray,
    link_margin_grid: np.ndarray,
    param1_name: str,
    param1_unit: str,
    param2_name: str,
    param2_unit: str,
    title: str = "2D Parameter Sweep - Link Margin"
) -> go.Figure:
    """
    2D 파라미터 스윕 히트맵

    Parameters:
        param1_grid: 첫 번째 파라미터 그리드
        param2_grid: 두 번째 파라미터 그리드
        link_margin_grid: 링크 마진 그리드
        param1_name: 첫 번째 파라미터 이름
        param1_unit: 첫 번째 파라미터 단위
        param2_name: 두 번째 파라미터 이름
        param2_unit: 두 번째 파라미터 단위
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        x=param1_grid[0, :],
        y=param2_grid[:, 0],
        z=link_margin_grid,
        colorscale='RdYlGn',
        colorbar=dict(title="Link Margin [dB]")
    ))

    # 0 dB 등고선
    fig.add_trace(go.Contour(
        x=param1_grid[0, :],
        y=param2_grid[:, 0],
        z=link_margin_grid,
        contours=dict(
            start=0,
            end=0,
            size=0.1,
            coloring='lines'
        ),
        line=dict(color='black', width=2),
        showscale=False,
        name='0 dB'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{param1_name} [{param1_unit}]",
        yaxis_title=f"{param2_name} [{param2_unit}]",
        height=500
    )

    return fig


def plot_sensitivity_analysis(
    base_value: float,
    parameter_names: List[str],
    low_margins: List[float],
    high_margins: List[float],
    title: str = "Sensitivity Analysis (Tornado Chart)"
) -> go.Figure:
    """
    민감도 분석 토네이도 차트

    Parameters:
        base_value: 기준 링크 마진 값
        parameter_names: 파라미터 이름 리스트
        low_margins: 낮은 값에서의 마진 리스트
        high_margins: 높은 값에서의 마진 리스트
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    # 영향도 순서로 정렬
    impacts = [abs(h - l) for h, l in zip(high_margins, low_margins)]
    sorted_indices = np.argsort(impacts)[::-1]

    sorted_names = [parameter_names[i] for i in sorted_indices]
    sorted_low = [low_margins[i] - base_value for i in sorted_indices]
    sorted_high = [high_margins[i] - base_value for i in sorted_indices]

    fig = go.Figure()

    # 낮은 값 영향 (음수 방향)
    fig.add_trace(go.Bar(
        y=sorted_names,
        x=sorted_low,
        orientation='h',
        name='Low Value',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))

    # 높은 값 영향 (양수 방향)
    fig.add_trace(go.Bar(
        y=sorted_names,
        x=sorted_high,
        orientation='h',
        name='High Value',
        marker_color='rgba(219, 64, 82, 0.7)'
    ))

    # 기준선
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title=f"Change in Link Margin [dB] (Base: {base_value:.1f} dB)",
        yaxis_title="Parameter",
        barmode='overlay',
        height=max(400, len(parameter_names) * 30),
        showlegend=True
    )

    return fig


def plot_distance_vs_visibility(
    distances: np.ndarray,
    visibilities: np.ndarray,
    link_margins: np.ndarray,
    title: str = "Link Margin: Distance vs Visibility"
) -> go.Figure:
    """
    거리 vs 가시거리 2D 히트맵

    Parameters:
        distances: 거리 배열 [m]
        visibilities: 가시거리 배열 [km]
        link_margins: 링크 마진 그리드 [dB]
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    return plot_2d_heatmap(
        *np.meshgrid(distances, visibilities),
        link_margins,
        "Distance", "m",
        "Visibility", "km",
        title
    )
