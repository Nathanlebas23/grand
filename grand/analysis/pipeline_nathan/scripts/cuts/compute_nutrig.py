"""
Load NUTRIG cuts from a file and apply them to a dataset.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_FLT_CACHE: Dict[str, Any] = {}
_NUTRIG_AVAILABLE: Optional[bool] = None

# Default location of the nutrig package relative to this file.
_DEFAULT_NUTRIG_PATH = Path(__file__).parent.parent / "nutrig"


def is_flt_available(nutrig_path: Path = _DEFAULT_NUTRIG_PATH) -> bool:
    """
    Check whether the nutrig.flt package can be imported, caching the result.
    """
    global _NUTRIG_AVAILABLE

    if _NUTRIG_AVAILABLE is not None:
        return _NUTRIG_AVAILABLE

    try:
        sys.path.insert(0, str(nutrig_path))
        import nutrig.flt.template_FLT  # noqa: F401
        _NUTRIG_AVAILABLE = True
    except ImportError:
        _NUTRIG_AVAILABLE = False

    return _NUTRIG_AVAILABLE


def get_flt_instance(
    templates_npz_path: Optional[str] = None,
    force_reload: bool = False,
    nutrig_path: Path = _DEFAULT_NUTRIG_PATH,
):
    """
    Get an instance of TemplateFLT, loading templates from a .npz file.
    """
    if not is_flt_available(nutrig_path):
        raise ImportError(
            "The nutrig.flt package is required for the FLT method. "
            "Make sure nutrig/ is on your PYTHONPATH."
        )

    from nutrig.flt.template_FLT import TemplateFLT

    # Cache key (no desampling factor needed, it's fixed in TemplateFLT)
    cache_key = templates_npz_path

    # Return the cached instance if available
    if not force_reload and cache_key in _FLT_CACHE:
        return _FLT_CACHE[cache_key]

    # Create a new instance
    flt = TemplateFLT()

    # IMPORTANT: configure the sampling rates BEFORE load_templates().
    # Pipeline templates are at 500 MHz (not 2 GHz, the nutrig default),
    # so we use factor=1 (no desampling).
    # Reference: nutrig notebook test_template_FLT.ipynb
    flt.set_sampling_rates(adc_sampling_rate=500, sim_sampling_rate=500)

    # Configure the correlation window.
    # A wider window [-50, 50] would let us find the peak even if
    # pre_trigger_sample isn't exactly on the maximum (given that
    # _template_peak_sample=30).
    # flt.set_corr_window([-50, 50])
    flt.set_corr_window([-10, 10])

    # Load the templates (uses the _desampling_factor configured above)
    flt.load_templates(templates_npz_path)

    # Cache the instance
    _FLT_CACHE[cache_key] = flt

    return flt


def compute_flt_correlation(
    trace: np.ndarray,
    pre_trigger_sample: int,
    templates_npz_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute the correlation using the FLT method.

    Definitions:
    - trace: The input signal trace to be analyzed.
    - pre_trigger_sample: The sample index before the trigger to consider for correlation.
    - templates_npz_path: Path to the .npz file containing the templates. If None, the default path will be used.
    - kwargs: Additional keyword arguments to pass to the FLT correlation function.

    Output:
    - A dictionary containing the correlation results, which may include:
        - 'rho_max': Maximum correlation value.
        - 'best_idx': Index of the best matching template.
        - 'best_position': Sample index of the best match in the trace.
        - 'ts': Time shift corresponding to the best match.
        - 't_corr': Array of time indices for the correlation window.
        - 'corr_vals_window_best': Correlation values for the best template in the correlation window.
        - 'template_best': The best matching template.
        - 'time_best': Time index of the best match.
        - 'corr_best': Correlation value for the best match.
    Raises:
    - ImportError: If the FLT method is not available (nutrig package not installed).
    - ValueError: If the trace is empty or if pre_trigger_sample is out of bounds.
    """
    if not is_flt_available():
        raise ImportError(
            "Needs NUTRIG package. "
            "Please add nutrig/ to your PYTHONPATH or install it."
        )

    if len(trace) == 0:
        raise ValueError("trace cannot be empty")
    if not (0 <= pre_trigger_sample < len(trace)):
        raise ValueError(
            f"pre_trigger_sample ({pre_trigger_sample}) should be in [0, {len(trace)})"
        )

    flt = get_flt_instance(templates_npz_path=templates_npz_path)

    # Convert the trace if needed (FLT expects float64)
    trace_float = trace.astype(np.float64) if trace.dtype != np.float64 else trace

    # Call the FLT method.
    # Note: template_fit() mutates flt's internal state (corr_best, time_best, etc.)
    flt.template_fit(trace_float, pre_trigger_sample)

    corr_window_start, corr_vals_window = flt._cross_corr_desampled(trace_float, pre_trigger_sample)

    t_corr = np.arange(corr_window_start, corr_window_start + corr_vals_window.shape[-1])
    corr_vals_window_best = corr_vals_window[
        flt.idx_template_best_fit, flt.idx_templates_desampled_best[flt.idx_template_best_fit]
    ]

    # Extract the results
    best_template_idx = int(flt.idx_template_best_fit)

    template_best = flt.templates[flt.idx_template_best_fit]
    time_best = flt.time_best[flt.idx_template_best_fit]
    corr_best = flt.ts

    result = {
        'rho_max': float(flt.corr_best[best_template_idx]),
        'best_idx': best_template_idx,
        'best_position': int(flt.time_best[best_template_idx]),
        'ts': float(flt.ts),
        't_corr': t_corr,
        'corr_vals_window_best': corr_vals_window_best,
        'template_best': template_best,
        'time_best': int(time_best),
        'corr_best': float(corr_best),
        'method': 'flt',
        # Extra info for further analysis
        'all_corr': flt.corr_best.copy(),
        'all_times': flt.time_best.copy(),
    }

    return result


def compute_correlation(
    trace: np.ndarray,
    templates: np.ndarray,
    pre_trigger_sample: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute the correlation between a trace and a set of templates.

    Definitions:
    - trace: The input signal trace to be analyzed.
    - templates: A 2D array where each row is a template to correlate with the trace.
    - pre_trigger_sample: Required. Specifies the sample index before the trigger to consider for correlation.
    - kwargs: Additional keyword arguments to pass to the correlation function.

    Output:
    - A dictionary containing the correlation results, which may include:
        - 'rho_x': The correlation values for the x-axis.
        - 'rho_y': The correlation values for the y-axis.
    Raises:
    - ImportError: If the nutrig package (FLT method) is not available.
    - ValueError: If pre_trigger_sample is missing.
    """
    if not is_flt_available():
        raise ImportError(
            "The 'flt' method requires the nutrig package. "
            "Add nutrig/ to your PYTHONPATH."
        )

    if pre_trigger_sample is None:
        raise ValueError("pre_trigger_sample is required")

    # Call the FLT backend (handles templates internally via the .npz file)
    result = compute_flt_correlation(
        trace=trace,
        pre_trigger_sample=pre_trigger_sample,
        **kwargs,
    )
    return result


def compute_rho_event_score(
    rho_x_arr: np.ndarray,
    rho_y_arr: np.ndarray,
    positions: list,
) -> Tuple[float, float, int]:
    """
    Calculate the event score based on rho_x and rho_y values at specified positions.

    Definitions:
    - rho_x_arr: Array of rho_x values for the event.
    - rho_y_arr: Array of rho_y values for the event.
    - positions: List of indices where the event is considered valid.

    Output:
    - rho_min: Minimum of the maximum rho values at the specified positions.
    - rho_mean: Mean of the maximum rho values at the specified positions.
    - count: Number of valid positions considered in the calculation.
    If no valid positions are found, returns (nan, nan, 0).
    """
    rho_maxs = []

    for pos in positions:
        rx, ry = rho_x_arr[pos], rho_y_arr[pos]
        rx_ok = np.isfinite(rx)
        ry_ok = np.isfinite(ry)

        if rx_ok and ry_ok:
            rho_maxs.append(max(rx, ry))
        elif rx_ok:
            rho_maxs.append(rx)
        elif ry_ok:
            rho_maxs.append(ry)

    if not rho_maxs:
        return float("nan"), float("nan"), 0

    rho_maxs = np.asarray(rho_maxs, dtype=float)
    return float(np.min(rho_maxs)), float(np.mean(rho_maxs)), len(rho_maxs)