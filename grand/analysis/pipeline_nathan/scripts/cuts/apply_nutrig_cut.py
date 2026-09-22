"""
Event-level NUTRIG FLT rho computation and cut.

Reproduces the historical per-channel FLT correlation implementation
(rho_x/rho_y computed independently per channel with that channel's own
peak sample, rho_max = max(rho_x, rho_y) per antenna) on top of the current
pipeline's ADC traces, plus the event-level rho_min/rho_mean cut.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import grand.analysis.signals.extraction as ext
from grand.analysis.pipeline_nathan.scripts.cuts.compute_nutrig import (
    is_flt_available,
    compute_flt_correlation,
    compute_rho_event_score,
)

# Keys kept from each per-channel FLT result dict - just enough for the
# sanity-plot template overlay (rescale_template_for_trace) and a title;
# the bulky all_corr/all_times/t_corr diagnostic arrays are dropped.
_RESULT_KEYS_TO_KEEP = ("rho_max", "best_position", "template_best", "ts")

# Could use get_peak_time_adc() from extraction.py, but that function is not
def _peak_idx_per_channel(channel_trace: np.ndarray) -> int:
    """argmax(|channel_trace|) - independent per-channel peak sample,
    matching the historical precompute_precise_time_and_peaks()'s X/Y logic."""
    return int(np.argmax(np.abs(channel_trace)))


def compute_nutrig_event(
    e,
    templates_npz_path,
    nutrig_src_path,
) -> Dict[str, Any]:
    """Compute per-antenna NUTRIG FLT rho_x/rho_y/rho_max and the event-level
    rho_min/rho_mean/n_valid score for every DU in e.voltages (same order).

    A global NUTRIG/configuration failure (package not importable) raises
    ImportError immediately - it is a setup problem, not a physically-bad
    event, and must not be silently turned into a NaN/cut. A local, per-
    channel failure (bad peak sample, empty trace) is caught and only
    leaves that channel's rho as NaN; compute_rho_event_score() then falls
    back to the other channel for that antenna.
    """
    if not is_flt_available(nutrig_path=nutrig_src_path):
        raise ImportError(
            f"nutrig.flt package not available (nutrig_src_path={nutrig_src_path!r}). "
            "Check paths.nutrig_src_path in config.yaml."
        )

    n_antennas = len(e.voltages)
    ADC_traces = np.array(
        [ext.convert_voltage_to_ADC(v.trace, channels=[0, 1, 2]) for v in e.voltages]
    )

    rho_x = np.full(n_antennas, np.nan, dtype=float)
    rho_y = np.full(n_antennas, np.nan, dtype=float)
    result_x: List[Optional[Dict[str, Any]]] = [None] * n_antennas
    result_y: List[Optional[Dict[str, Any]]] = [None] * n_antennas

    for i in range(n_antennas):
        try:
            peak_idx_x = _peak_idx_per_channel(ADC_traces[i, 0])
            rx = compute_flt_correlation(
                ADC_traces[i, 0],
                pre_trigger_sample=peak_idx_x,
                templates_npz_path=templates_npz_path,
            )
            rho_x[i] = rx["rho_max"]
            result_x[i] = {k: rx[k] for k in _RESULT_KEYS_TO_KEEP}
        except ValueError:
            pass

        try:
            peak_idx_y = _peak_idx_per_channel(ADC_traces[i, 1])
            ry = compute_flt_correlation(
                ADC_traces[i, 1],
                pre_trigger_sample=peak_idx_y,
                templates_npz_path=templates_npz_path,
            )
            rho_y[i] = ry["rho_max"]
            result_y[i] = {k: ry[k] for k in _RESULT_KEYS_TO_KEEP}
        except ValueError:
            pass

    rho_max = np.full(n_antennas, np.nan, dtype=float)
    for i in range(n_antennas):
        rx_ok = np.isfinite(rho_x[i])
        ry_ok = np.isfinite(rho_y[i])
        if rx_ok and ry_ok:
            rho_max[i] = max(rho_x[i], rho_y[i])
        elif rx_ok:
            rho_max[i] = rho_x[i]
        elif ry_ok:
            rho_max[i] = rho_y[i]

    rho_min, rho_mean, n_valid = compute_rho_event_score(
        rho_x, rho_y, positions=list(range(n_antennas))
    )

    return {
        "rho_x": rho_x,
        "rho_y": rho_y,
        "rho_max": rho_max,
        "rho_min": rho_min,
        "rho_mean": rho_mean,
        "n_valid": n_valid,
        "ADC_traces": ADC_traces,
        "result_x": result_x,
        "result_y": result_y,
    }


def passes_nutrig_cut(
    nutrig_result: Dict[str, Any],
    rho_min_threshold: float,
    rho_mean_threshold: float,
) -> bool:
    """PASS iff every antenna had at least one valid channel (n_valid equals
    the antenna count - an antenna with both X and Y invalid fails the whole
    event, rather than being silently excluded from rho_min/rho_mean) and
    both event-level scores clear their threshold.

    NaN-safe by construction: `nan >= threshold` is False in NumPy, so a
    NaN rho_min/rho_mean (e.g. n_valid == 0) correctly fails the cut.
    """
    return (
        nutrig_result["n_valid"] == len(nutrig_result["rho_max"])
        and nutrig_result["rho_min"] >= rho_min_threshold
        and nutrig_result["rho_mean"] >= rho_mean_threshold
    )


def rescale_template_for_trace(
    trace: np.ndarray,
    template: np.ndarray,
    best_position: int,
    template_peak_sample: int = 30,
) -> dict:
    """
    Reproduit exactement le pipeline FLT pour affichage diagnostic.

    Le FLT decale la fenetre de trace par -template_peak_sample avant la
    correlation, et normalise la trace par max(|trace|). Cette fonction
    applique les memes operations pour produire un template rescale
    coherent avec le calcul de correlation.

    Args:
        trace: Trace brute complete (1D, unites ADC)
        template: Template selectionne (1D, L2-normalise)
        best_position: Index retourne par FLT (corr_window_start + argmax)
        template_peak_sample: Decalage interne FLT (defaut 30, cf. nutrig)

    Returns:
        Dictionnaire contenant:
            - 'trace_slice': slice de trace brute aligne avec le template
            - 'scaled_template': template rescale en unites brutes (pour overlay)
            - 'trace_start': index de debut dans la trace complete
            - 'trace_end': index de fin dans la trace complete
            - 'norm_factor': facteur de normalisation max(|trace|)
            - 'a_norm': coefficient LS dans l'espace normalise
            - 'template_unit': template L2-unitaire
            - 'trace_slice_norm': slice normalise (comme vu par le FLT)
            - 'scaled_template_norm': template rescale dans l'espace normalise
    """
    n_template = len(template)

    # Le FLT decale la fenetre par -template_peak_sample
    #    best_position = corr_window_start + argmax, sans le decalage de -30
    #    Le slice reel utilise dans la correlation commence a :
    trace_start = best_position - template_peak_sample
    trace_end = trace_start + n_template

    # Securite: si le template sort des bornes de la trace, on ne produit pas
    # un overlay tronque/faux - on signale et on n'affiche rien.
    if trace_start < 0 or trace_end > len(trace):
        print(
            f"[rescale_template_for_trace] WARNING: template window "
            f"[{trace_start}, {trace_end}) sort des bornes de la trace "
            f"(longueur {len(trace)}) - overlay non affiche."
        )
        return {
            'valid': False,
            'trace_slice': None,
            'scaled_template': None,
            'trace_start': trace_start,
            'trace_end': trace_end,
            'norm_factor': None,
            'a_norm': None,
            'template_unit': None,
            'trace_slice_norm': None,
            'scaled_template_norm': None,
        }

    trace_slice_raw = trace[trace_start:trace_end]

    # FLT normalise la trace entiere par max(|trace|) avant correlation
    trace_abs_max = np.max(np.abs(trace))
    norm_factor = trace_abs_max if trace_abs_max > 0 else 1.0
    trace_slice_norm = trace_slice_raw / norm_factor

    # Le template est L2-normalise (deja le cas dans les .npz)
    template_l2 = np.linalg.norm(template)
    template_unit = template / template_l2 if template_l2 > 0 else template

    # Least-squares fit dans l'espace normalise (coherent avec FLT)
    a_norm = np.dot(trace_slice_norm, template_unit) / np.dot(template_unit, template_unit)
    scaled_template_norm = a_norm * template_unit

    # Reconvertir en unites brutes pour superposition sur le plot
    scaled_template_raw = scaled_template_norm * norm_factor

    return {
        'valid': True,
        'trace_slice': trace_slice_raw,
        'scaled_template': scaled_template_raw,
        'trace_start': trace_start,
        'trace_end': trace_end,
        'norm_factor': norm_factor,
        'a_norm': a_norm,
        'template_unit': template_unit,
        'trace_slice_norm': trace_slice_norm,
        'scaled_template_norm': scaled_template_norm,
    }
