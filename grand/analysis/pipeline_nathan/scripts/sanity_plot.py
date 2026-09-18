import matplotlib.pyplot as plt
import numpy as np


def plot_traces(e, output_dir):
    """Plot X/Y/Z ADC traces for every triggered antenna."""

    n_antennas = len(e.antennas)

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