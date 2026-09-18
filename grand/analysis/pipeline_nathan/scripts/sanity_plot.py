import matplotlib.pyplot as plt
import numpy as np


def plot_traces(e, output_dir, peak_times=None):
    """Plot X/Y/Z ADC traces for every triggered antenna.

    peak_times, if given, is the per-antenna array (seconds, same order as e.voltages)
    computed by reconstruction.py's reconstruct_event() - overlaid as a vertical line
    marking the reconstructed NUTRIG peak time, converted to each antenna's own local
    (0-based) time axis.
    """

    n_antennas = len(e.antennas)

    # t0_rel_ns matches reconstruction.py's own t0 computation exactly, needed to convert
    # peak_times (relative to the event's earliest-triggering antenna) back to each
    # antenna's local 0-based trace axis.
    if peak_times is not None:
        t0_all_ns = np.array([v.t0.astype('int64') for v in e.voltages])
        t0_rel_ns = t0_all_ns - t0_all_ns.min()

    fig, axs = plt.subplots(
        n_antennas,
        1,
        figsize=(12, 3 * n_antennas),
        sharex=True,
        squeeze=False,
    )
    axs = axs[:, 0]

    for i in range(n_antennas):
        v = e.voltages[i]
        traces = np.asarray(v.trace)  # (3, n_samples): axis 0 = X/Y/Z, raw voltage in uV

        n_samples = traces.shape[-1]
        t_ns = np.arange(n_samples) * v.t_bin_size

        axs[i].plot(t_ns, traces[0], label="X")
        axs[i].plot(t_ns, traces[1], label="Y")
        axs[i].plot(t_ns, traces[2], label="Z")

        if peak_times is not None:
            peak_local_ns = peak_times[i] * 1e9 - t0_rel_ns[i]
            axs[i].axvline(
                peak_local_ns,
                color="k",
                linestyle="--",
                label="NUTRIG peak time" if i == 0 else None,
            )

        axs[i].set_title(f"DU {v.du_id}")
        axs[i].set_ylabel("Voltage [uV]")
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

    fig.savefig(plot_path)
    plt.close(fig)