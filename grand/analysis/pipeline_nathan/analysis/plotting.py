import sys
sys.path.append("/home/lpnhe/grand")

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_ADC_vs_omega(t_recons, omega_cr_mean, w, adf_f, fsuptit, ftit_adf, plot_dir):
    """Plot ADC peak amplitude vs angle ω and ADF model."""
    plt.figure()
    # Peak ADC values measured at each triggered DU
    plt.plot(np.rad2deg(t_recons.omega), np.asarray(t_recons.peak_amps), 'ob', label="max ADC @ DU")
    plt.errorbar(np.rad2deg(t_recons.omega), np.asarray(t_recons.peak_amps), 0.075 * np.asarray(t_recons.peak_amps), fmt='None', marker='o', markerfacecolor='b')
    # ADF amplitude at each DU
    plt.plot(np.rad2deg(t_recons.omega), t_recons.adf_amplitude, '+r', label="ADF fit @ DU")
    # Mean Cherenkov angle for all eta values
    plt.plot([np.rad2deg(omega_cr_mean), np.rad2deg(omega_cr_mean)], [0, max(np.asarray(t_recons.peak_amps)) * 1.5], label="mean Cherenkov angle")
    # Mean ADF model
    plt.plot(np.rad2deg(w), adf_f, "--r", label="mean ADF model")
    plt.ylim([0, max(t_recons.adf_amplitude) * 1.5])
    plt.xlim([0, 1.6])
    plt.xlabel("$\omega$ (deg)")
    plt.legend(loc="best")
    plt.suptitle(fsuptit, fontsize=10)
    plt.title(ftit_adf)
    plt.ylabel("Voltage (ADC)")
    plt.savefig(f"{plot_dir}/ADC_vs_omega_event_{t_recons.event_number}_run_{t_recons.run_number}.png")


def plot_footprint(t_recons, antenna_position, Xants, K, Xcore, x_ell, omega_cr_mean, fsuptit, ftit_adf, plot_dir):
    """Plot the 2D footprint of the event on the ground."""
    distm = 5000  # distance in meters
    xmin, xmax = -distm, distm
    ymin, ymax = -distm, distm

    plt.figure()
    # Triggered DUs: color and size proportional to amplitude
    sc = plt.scatter(-Xants[:, 1], Xants[:, 0], c=t_recons.adf_amplitude, cmap='viridis', s=np.asarray(t_recons.adf_amplitude), label="Triggered DUs")
    # All DUs on site
    plt.scatter(antenna_position['x'], antenna_position['y'], marker='+', color='red', label="DUs on site")
    plt.colorbar(sc, label='Peak amplitude (ADC)')
    # Shower direction arrow
    plt.arrow(-Xcore[1] + K[1] * 1e3, Xcore[0] - K[0] * 1e3, -K[1] * 1e3, K[0] * 1e3, head_width=1, head_length=1, fc='black', ec='black')
    # Footprint: core position (black dot) and Cherenkov ellipse (dashed line)
    plt.plot(-Xcore[1], Xcore[0], 'ko')
    plt.plot(-x_ell[:, 1], x_ell[:, 0], '--k', linewidth=2)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.grid(True)
    plt.suptitle(fsuptit, fontsize=10)
    plt.title(ftit_adf)
    plt.legend()
    plt.axis('equal')
    plt.ylim([xmin, xmax])
    plt.xlim([ymin, ymax])
    plt.subplots_adjust(left=0.15)
    plt.savefig(f"{plot_dir}/footprint_event_{t_recons.event_number}_run_{t_recons.run_number}.png")


def plot_timing_residuals(t_recons, timing: dict, plot_dir) -> None:
    """Plot measured-vs-model peak times and residuals for PWF and SWF, one figure per event."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    for ax, t_model_ns, chi2r, label in (
        (axs[0, 0], timing["t_pwf_ns"], timing["chi2_pwf_reduced"], "PWF"),
        (axs[0, 1], timing["t_swf_ns"], timing["chi2_swf_reduced"], "SWF"),
    ):
        ax.scatter(timing["t_exp_ns"], t_model_ns)
        for i, idx in enumerate(timing["antenna_index"]):
            ax.annotate(str(idx), (timing["t_exp_ns"][i], t_model_ns[i]), fontsize=7)
        vmin = min(timing["t_exp_ns"].min(), t_model_ns.min())
        vmax = max(timing["t_exp_ns"].max(), t_model_ns.max())
        ax.plot([vmin, vmax], [vmin, vmax], "--", color="gray")
        ax.set_xlabel("t_exp [ns]")
        ax.set_ylabel("t_rec [ns]")
        ax.set_title(f"{label}  chi2/ndf = {chi2r:.2f}")

    sigma_t_ns = timing["sigma_t_ns"]
    for ax, resid_ns, label in (
        (axs[1, 0], timing["resid_pwf_ns"], "PWF"),
        (axs[1, 1], timing["resid_swf_ns"], "SWF"),
    ):
        ax.scatter(timing["t_exp_ns"], resid_ns)
        ax.axhline(0, linestyle="--", color="gray")
        ax.axhline(sigma_t_ns, linestyle=":", color="gray", label=f"+-{sigma_t_ns:.0f} ns")
        ax.axhline(-sigma_t_ns, linestyle=":", color="gray")
        ax.set_xlabel("t_exp [ns]")
        ax.set_ylabel("t_exp - t_rec [ns]")
        ax.set_title(f"{label} residuals")
        ax.legend(fontsize=8)

    fig.suptitle(f"Event {t_recons.event_number} - Run {t_recons.run_number}")
    fig.tight_layout()
    fig.savefig(Path(plot_dir) / f"timing_residuals_event_{t_recons.event_number}_run_{t_recons.run_number}.png")
    plt.close(fig)


def plot_event(t_recons, results: dict, antenna_position, plot_dir) -> None:
    """Produce the ADC-vs-omega and footprint plots for one already-analyzed event."""
    plot_ADC_vs_omega(
        t_recons, results["omega_cr_mean"], results["w"], results["adf_f"],
        results["fsuptit"], results["ftit_adf"], plot_dir,
    )
    plot_footprint(
        t_recons, antenna_position, results["Xants"], results["K"], results["Xcore"],
        results["x_ell"], results["omega_cr_mean"], results["fsuptit"], results["ftit_adf"], plot_dir,
    )
