import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import grand.analysis.signals.extraction as ext
from grand.analysis.pipeline_nathan.scripts.cuts.apply_nutrig_cut import rescale_template_for_trace


#-------------------------------
# Load GRANDlib and nl_style
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)

sys.path.append(config['paths']['grandlib_path'])
sys.path.append(config['paths']['nl_style_path'])

from nl_style import set_style , NL_COLORS

set_style()


## Le peak time est celui obtenue par reconstruction.py
## 

def plot_traces(e, output_dir, peak_times=None, ADC_traces=None, nutrig_result=None, t0_ns=None):
    """
    Plot X/Y/Z ADC traces for every triggered antenna.

    peak_times, if given, is the per-antenna array (seconds, same order as e.voltages)
    computed by reconstruction.py's reconstruct_event() - overlaid as a vertical line
    marking the reconstructed NUTRIG peak time, converted to each antenna's own local
    (0-based) time axis.

    ADC_traces, if given, is the (n_antennas, 3, n_samples) array already computed by
    compute_nutrig_event() - reused here instead of recomputing the voltage->ADC
    conversion. If None, this function converts e.voltages itself, so plot_traces()
    still works standalone (e.g. without going through compute_nutrig_event() first).

    nutrig_result, if given, is the dict returned by compute_nutrig_event() - used for
    the per-DU rho_x/rho_y/rho_max title and the FLT best-fit template overlay on the
    X/Y channels (via result_x/result_y + rescale_template_for_trace).
    """

  
    n_antennas = len(e.antennas)

    # t0_rel_ns matches reconstruction.py's own t0 computation exactly, needed to convert
    # peak_times (relative to the event's earliest-triggering antenna) back to each
    # antenna's local 0-based trace axis.

    if peak_times is not None:
        # t0_all_ns = np.array([v.t0.astype('int64') for v in e.voltages])
        # t0_rel_ns = t0_all_ns - t0_all_ns.min()
        # Must be EXACTLY the t0 that reconstruct_event used to build peak_times, otherwise
        # the subtraction below mixes two different time references. In simulation that is
        # compute_t0_sims(tadc) (epoch-anchored), whereas compute_t0(e.tvoltage) anchors on
        # min(du_seconds) and min(du_nanoseconds) taken independently - when the DUs straddle
        # a second boundary the two differ by ~1e9 ns, which threw the axvline to -1e9 ns and
        # autoscaled the whole trace into the right edge of the plot.
        t0_rel_ns = (np.asarray(t0_ns, dtype=float) if t0_ns is not None
                     else ext.compute_t0(e.tvoltage))

    fig, axs = plt.subplots(
        n_antennas,
        1,
        figsize=(12, 3 * n_antennas),
        sharex=True,
        squeeze=False,
    )

    try:
        axs = axs[:, 0]

        # Tracks whether each FLT-template legend entry has been added yet - anchored to
        # the first antenna where that channel's overlay is actually drawn, rather than
        # unconditionally i == 0 (whose own channel can be the one that failed locally).
        _template_legend_added = {"X": False, "Y": False}

        for i in range(n_antennas):
            v = e.voltages[i]
            if ADC_traces is not None:
                traces = np.asarray(ADC_traces[i])
            else:
                traces = np.asarray(ext.convert_voltage_to_ADC(v.trace, channels=[0, 1, 2]))

            n_samples = traces.shape[-1] # 512 points
            t_ns = np.arange(n_samples) * v.t_bin_size # 1024 ns

            axs[i].plot(t_ns, traces[0], label="X")
            axs[i].plot(t_ns, traces[1], label="Y")
            axs[i].plot(t_ns, traces[2], label="Z")

            if peak_times is not None:
                peak_local_ns = peak_times[i] * 1e9 - t0_rel_ns[i]
                axs[i].axvline(
                    peak_local_ns,
                    color=NL_COLORS['black'],
                    linestyle="--",
                    label="NUTRIG peak time" if i == 0 else None,
                )

            title = f"DU {v.du_id}"

            if nutrig_result is not None:
                rho_x_i = nutrig_result["rho_x"][i]
                rho_y_i = nutrig_result["rho_y"][i]
                rho_max_i = nutrig_result["rho_max"][i]
                title += f" — rhoX={rho_x_i:.2f}, rhoY={rho_y_i:.2f}, rhoMax={rho_max_i:.2f}"

                # Template overlay: rescale_template_for_trace expects the same raw ADC
                # units these traces already are in (no unit conversion needed here).
                for channel_idx, channel_label, result_list, color in (
                    (0, "X", nutrig_result["result_x"], NL_COLORS['blue']),
                    (1, "Y", nutrig_result["result_y"], NL_COLORS['orange']),
                ):
                    result_i = result_list[i]
                    if result_i is None:
                        continue
                    rescaled = rescale_template_for_trace(
                        traces[channel_idx], result_i["template_best"], result_i["best_position"]
                    )
                    if not rescaled["valid"]:
                        continue
                    window_ns = np.arange(rescaled["trace_start"], rescaled["trace_end"]) * v.t_bin_size
                    axs[i].plot(
                        window_ns,
                        rescaled["scaled_template"],
                        linestyle="--",
                        color=color,
                        label=(
                            f"FLT template {channel_label}"
                            if not _template_legend_added[channel_label]
                            else None
                        ),
                    )
                    _template_legend_added[channel_label] = True

            axs[i].set_title(title)
            axs[i].set_ylabel("Voltage [ADC]")
            axs[i].legend()

        axs[-1].set_xlabel("Time [ns]")

        fig.suptitle(
            f"Event {e.event_number} - Run {e.run_number}"
        )
        fig.tight_layout()

        plot_path = (
            output_dir
            / f"event_{e.event_number}_run_{e.run_number}_traces.png"
        )

        fig.savefig(plot_path, dpi=200)

    finally:
        plt.close(fig)