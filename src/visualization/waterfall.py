"""
Waterfall (Cascade) 차트
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional


def create_waterfall_chart(
    labels: list[str],
    values: list[float],
    title: str = "Link Budget Waterfall",
    measure_types: Optional[list[str]] = None,
    y_label: str = "Power [dBm] / Gain/Loss [dB]"
) -> go.Figure:
    """
    Waterfall 차트 생성

    Parameters:
        labels: 항목 라벨 리스트
        values: 값 리스트 (양수: 이득, 음수: 손실)
        title: 차트 제목
        measure_types: Plotly measure 타입 ("relative", "total", "absolute")
        y_label: Y축 라벨

    Returns:
        Plotly Figure
    """
    if measure_types is None:
        # 첫 항목과 마지막 항목은 absolute/total
        measure_types = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]

    # 색상 설정
    colors = []
    for val, mtype in zip(values, measure_types):
        if mtype == "total":
            colors.append("rgba(100, 149, 237, 0.8)")  # 파란색 (총합)
        elif val >= 0:
            colors.append("rgba(46, 204, 113, 0.8)")   # 초록색 (이득)
        else:
            colors.append("rgba(231, 76, 60, 0.8)")    # 빨간색 (손실)

    fig = go.Figure(go.Waterfall(
        name="Link Budget",
        orientation="v",
        measure=measure_types,
        x=labels,
        textposition="outside",
        text=[f"{v:+.2f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "rgba(46, 204, 113, 0.8)"}},
        decreasing={"marker": {"color": "rgba(231, 76, 60, 0.8)"}},
        totals={"marker": {"color": "rgba(100, 149, 237, 0.8)"}}
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title=y_label,
        showlegend=False,
        height=500,
        xaxis_tickangle=-45
    )

    return fig


def create_uplink_waterfall(uplink_dict: dict[str, float]) -> go.Figure:
    """
    Uplink 링크 버짓 waterfall 차트

    Parameters:
        uplink_dict: UplinkBudget.to_dict() 결과

    Returns:
        Plotly Figure
    """
    labels = list(uplink_dict.keys())
    values = list(uplink_dict.values())

    # measure 타입 설정
    measures = []
    for i, label in enumerate(labels):
        if i == 0:  # P_tx
            measures.append("absolute")
        elif "P_mrr_in" in label:  # 합계
            measures.append("total")
        else:
            measures.append("relative")

    return create_waterfall_chart(
        labels=labels,
        values=values,
        title="Uplink Link Budget",
        measure_types=measures
    )


def create_downlink_waterfall(downlink_dict: dict[str, float]) -> go.Figure:
    """
    Downlink 링크 버짓 waterfall 차트

    Parameters:
        downlink_dict: DownlinkBudget.to_dict() 결과

    Returns:
        Plotly Figure
    """
    labels = list(downlink_dict.keys())
    values = list(downlink_dict.values())

    # measure 타입 설정
    measures = []
    for i, label in enumerate(labels):
        if i == 0:  # P_mrr_in
            measures.append("absolute")
        elif "P_rx" in label:  # 합계
            measures.append("total")
        else:
            measures.append("relative")

    return create_waterfall_chart(
        labels=labels,
        values=values,
        title="Downlink Link Budget",
        measure_types=measures
    )


def create_full_link_waterfall(
    uplink_dict: dict[str, float],
    downlink_dict: dict[str, float],
    sensitivity_dBm: float
) -> go.Figure:
    """
    전체 링크 버짓 waterfall 차트 (Uplink + Downlink + Margin)

    Parameters:
        uplink_dict: UplinkBudget.to_dict() 결과
        downlink_dict: DownlinkBudget.to_dict() 결과
        sensitivity_dBm: 수신 감도 [dBm]

    Returns:
        Plotly Figure
    """
    labels = []
    values = []
    measures = []

    # Uplink (P_tx 시작, P_mrr_in 전까지)
    for label, value in uplink_dict.items():
        if "P_mrr_in" not in label:
            labels.append(f"UL: {label}")
            values.append(value)
            if label.startswith("P_tx"):
                measures.append("absolute")
            else:
                measures.append("relative")

    # MRR 입력 (중간 합계)
    labels.append("MRR Input")
    values.append(uplink_dict.get("P_mrr_in [dBm]", 0))
    measures.append("total")

    # Downlink (P_mrr_in 이후부터 P_rx 전까지)
    skip_first = True
    for label, value in downlink_dict.items():
        if skip_first:
            skip_first = False
            continue
        if "P_rx" not in label:
            labels.append(f"DL: {label}")
            values.append(value)
            measures.append("relative")

    # 최종 수신 전력
    P_rx = downlink_dict.get("P_rx [dBm]", 0)
    labels.append("RX Power")
    values.append(P_rx)
    measures.append("total")

    # 링크 마진
    margin = P_rx - sensitivity_dBm
    labels.append(f"Margin (vs {sensitivity_dBm} dBm)")
    values.append(margin)
    measures.append("relative")

    return create_waterfall_chart(
        labels=labels,
        values=values,
        title="Full Link Budget (Uplink → MRR → Downlink)",
        measure_types=measures
    )


def create_comparison_bar_chart(
    antenna_result: dict,
    optical_result: dict,
    title: str = "Antenna vs Optical Model Comparison"
) -> go.Figure:
    """
    Antenna vs Optical 모델 비교 막대 차트

    Parameters:
        antenna_result: Antenna 모델 결과 딕셔너리
        optical_result: Optical 모델 결과 딕셔너리 (dB 변환)
        title: 차트 제목

    Returns:
        Plotly Figure
    """
    categories = list(antenna_result.keys())
    antenna_values = list(antenna_result.values())
    optical_values = [optical_result.get(k, 0) for k in categories]

    fig = go.Figure(data=[
        go.Bar(name='Antenna Model', x=categories, y=antenna_values,
               marker_color='rgba(55, 128, 191, 0.7)'),
        go.Bar(name='Optical Model', x=categories, y=optical_values,
               marker_color='rgba(219, 64, 82, 0.7)')
    ])

    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Component",
        yaxis_title="Value [dB]",
        xaxis_tickangle=-45,
        height=500
    )

    return fig
