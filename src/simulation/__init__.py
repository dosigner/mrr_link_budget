"""
Simulation module - Monte Carlo and Parameter Sweep
"""
from .monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo_antenna,
    run_monte_carlo_optical,
    calculate_ber_ook,
    calculate_ber_bpsk,
    calculate_required_fade_margin
)

from .parameter_sweep import (
    SweepResult,
    sweep_distance,
    sweep_visibility,
    sweep_orientation_error,
    sweep_tracking_error,
    sweep_mrr_diameter,
    sweep_tx_power,
    sweep_divergence,
    sweep_cn2,
    sweep_generic,
    sweep_2d
)

__all__ = [
    # Monte Carlo
    "MonteCarloConfig",
    "MonteCarloResult",
    "run_monte_carlo_antenna",
    "run_monte_carlo_optical",
    "calculate_ber_ook",
    "calculate_ber_bpsk",
    "calculate_required_fade_margin",
    # Parameter Sweep
    "SweepResult",
    "sweep_distance",
    "sweep_visibility",
    "sweep_orientation_error",
    "sweep_tracking_error",
    "sweep_mrr_diameter",
    "sweep_tx_power",
    "sweep_divergence",
    "sweep_cn2",
    "sweep_generic",
    "sweep_2d"
]
