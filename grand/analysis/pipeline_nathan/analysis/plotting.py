import sys

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_ADC_vs_omega(
    t_recons,
    omega_cr_mean,
    w,
    adf_f,
    fsuptit,
    ftit_adf,
    plot_dir,
    ):
    """Plot ADC peak amplitude vs angle omega and ADF model."""

    fig, ax = plt.subplots()

    omega_deg = np.rad2deg(t_recons.omega)
    peak_amps = np.asarray(t_recons.peak_amps)
    adf_amplitude = np.asarray(t_recons.adf_amplitude)

    ax.plot(
        omega_deg,
        peak_amps,
        "ob",
        label="max ADC @ DU",
        zorder=10,
    )

    ax.errorbar(
        omega_deg,
        peak_amps,
        yerr=0.075 * peak_amps,
        fmt="None",
        marker="o",
        markerfacecolor="b",
    )

    ax.plot(
        omega_deg,
        adf_amplitude,
        "+r",
        label="ADF fit @ DU",
    )

    ax.axvline(
        np.rad2deg(omega_cr_mean),
        label="mean Cherenkov angle",
        color=NL_COLORS["blue"]
    )

    ax.annotate(
        f"$\\omega_c = {np.rad2deg(omega_cr_mean):.2f} ^\\circ$",
        xy=(np.rad2deg(omega_cr_mean), 0.5 * adf_amplitude.max()),
        xytext=(np.rad2deg(omega_cr_mean) - 0.02, 0.5 * adf_amplitude.max()), 
        rotation=90,
        va="center",
        color=NL_COLORS["blue"],
    )

    ax.plot(
        np.rad2deg(w),
        adf_f,
        "--r",
        label="mean ADF model",
    )

    ax.set_ylim(0, adf_amplitude.max() * 1.5)
    omega_max_plot = 1.1 * np.nanmax(omega_deg)
    ax.set_xlim(0, omega_max_plot)

    ax.set_xlabel(r"$\omega$ [deg]")
    ax.set_ylabel("Voltage (ADC)")
    ax.legend(frameon=False, loc="best")

    fig.suptitle(fsuptit, fontsize=10)
    ax.set_title(ftit_adf)

    fig.savefig(plot_dir/ f"ADC_vs_omega_event_{t_recons.event_number}_run_{t_recons.run_number}.png")

    plt.close(fig)

def plot_footprint(
    t_recons,
    antenna_position,
    Xants,
    K,
    Xcore,
    x_ell,
    omega_cr_mean,
    omega_ell_min,
    omega_ell_max,
    x_ell_min,
    x_ell_max,
    fsuptit,
    ftit_adf,
    plot_dir,
):
    """Plot the 2D footprint of the event on the ground."""

    distm = 5  # distance in k_meters
    xmin, xmax = -distm, distm
    ymin, ymax = -distm, distm

    fig, ax = plt.subplots()

    # Triggered DUs: color and size proportional to amplitude
    amplitudes = np.asarray(t_recons.adf_amplitude)

    sc = ax.scatter(
        Xants[:, 1] / 1000,
        Xants[:, 0] / 1000,
        c=amplitudes,
        cmap="Reds",
        s=amplitudes,
        label="Triggered DUs",
    )

    # All DUs on site
    ax.scatter(
        -antenna_position["y"] / 1000,
        antenna_position["x"] / 1000,
        marker="+",
        color=NL_COLORS["black"],
        label="DUs on site",
    )

    fig.colorbar(sc, ax=ax, label="Peak amplitude (ADC)")

    # Shower direction arrow
    ax.arrow(
        (Xcore[1] - K[1] * 3000) / 1000 ,
        (Xcore[0] - K[0] * 3000) / 1000,
        K[1],
        K[0],
        head_width=0.1,
        head_length=0.15,
        fc="black",
        ec="black",
        length_includes_head=True,
    )

    # Core position
    ax.plot(
        Xcore[1] / 1000,
        Xcore[0] / 1000,
        "ko",
        label="Core",
        markersize=5,
        zorder=1,
    )

    # Cherenkov ellipse
    ax.plot(
        x_ell[:, 1] / 1000,
        x_ell[:, 0] / 1000,
        "--k",
        linewidth=1.5,
        label="Cherenkov ellipse",
    )

    if x_ell_min is not None and x_ell_max is not None:

        # Coordinates in the plotting frame
        x_min = x_ell_min[:, 1] / 1000
        y_min = x_ell_min[:, 0] / 1000

        x_max = x_ell_max[:, 1] / 1000
        y_max = x_ell_max[:, 0] / 1000

        x_min_closed = np.r_[x_min, x_min[0]]
        y_min_closed = np.r_[y_min, y_min[0]]

        x_max_closed = np.r_[x_max, x_max[0]]
        y_max_closed = np.r_[y_max, y_max[0]]
                

        # Light fill between the two ellipses
        if (
            np.all(np.isfinite(x_min))
            and np.all(np.isfinite(y_min))
            and np.all(np.isfinite(x_max))
            and np.all(np.isfinite(y_max))
        ):

            ax.fill(
                np.concatenate([x_min_closed, x_max_closed[::-1]]),
                np.concatenate([y_min_closed, y_max_closed[::-1]]),
                color=NL_COLORS["orange"],
                alpha=0.2,
                linewidth=0,
                label=(
                    rf"$\omega \in "
                    rf"\left[{np.rad2deg(omega_ell_min):.1f}, "
                    rf"{np.rad2deg(omega_ell_max):.1f}\right]^\circ$"
                ),
            )

        ax.plot(
            x_min_closed,
            y_min_closed,
            "--",
            color=NL_COLORS["orange"],
            linewidth=1,
        )

        ax.plot(
            x_max_closed,
            y_max_closed,
            "--",
            color=NL_COLORS["orange"],
            linewidth=1,
        )
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    ax.set_title(ftit_adf)
    fig.suptitle(fsuptit, fontsize=10)

    # fig.suptitle(
    # f"Event {t_recons.event_number} — Run {t_recons.run_number}",
    # fontsize=12,
    # y=0.90,
    # )

    # -----------------------------------------------
    # Add reconstruction results as text on the plot
    # -----------------------------------------------

    # ndf = t_recons.du_count - 4
    # chi2_adf_reduced = (
    #     t_recons.chi2_adf / ndf
    #     if ndf > 0
    #     else np.nan
    # )

    # reco_text = (
    #     rf"$\theta = {np.rad2deg(t_recons.zenith_adf):.1f}^\circ$"
    #     "\n"
    #     rf"$\phi = {np.rad2deg(t_recons.azimuth_adf):.1f}^\circ$"
    #     "\n"
    #     rf"$\chi^2_{{\rm ADF}}/\mathrm{{ndf}} = {chi2_adf_reduced:.2f}$"
    #     "\n"
    #     rf"$\omega_c = {np.rad2deg(omega_cr_mean):.2f}^\circ$"
    # )

    # ax.text(
    #     0.05,
    #     0.95,
    #     reco_text,
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=9,
    # )

    ax.legend(frameon=False, loc="best", fontsize=9)

    fig.savefig(
        plot_dir
        / f"footprint_event_{t_recons.event_number}_run_{t_recons.run_number}.png",
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_event(t_recons, results: dict, antenna_position, plot_dir) -> None:
    """Produce the ADC-vs-omega and footprint plots for one already-analyzed event."""
    plot_ADC_vs_omega(
        t_recons, results["omega_cr_mean"], results["w"], results["adf_f"],
        results["fsuptit"], results["ftit_adf"], plot_dir,
    )
    plot_footprint(
        t_recons, antenna_position, results["Xants"], results["K"], results["Xcore"],
        results["x_ell"], results["omega_cr_mean"], results['omega_ell_min'], results['omega_ell_max'], results['x_ell_min'], results['x_ell_max'], results["fsuptit"], results["ftit_adf"], plot_dir,
    )

def plot_timing_residuals(t_recons, timing: dict, plot_dir) -> None:
    """Plot measured-vs-model peak times and residuals for PWF and SWF, one figure per event."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    for ax, t_model_ns, chi2r, label in (
        (axs[0, 0], timing["t_pwf_ns"], timing["chi2_pwf_reduced"], "PWF"),
        (axs[0, 1], timing["t_swf_ns"], timing["chi2_swf_reduced"], "SWF"),
    ):
        ax.scatter(timing["t_exp_ns"], t_model_ns)
        for i, idx in enumerate(timing["antenna_index"]):
            ax.annotate(str(idx), (timing["t_exp_ns"][i] + 10, t_model_ns[i]), fontsize=7)
        vmin = min(timing["t_exp_ns"].min(), t_model_ns.min())
        vmax = max(timing["t_exp_ns"].max(), t_model_ns.max())
        ax.plot([vmin, vmax], [vmin, vmax], "--", color="gray")
        ax.set_xlabel(r"$\rm t_{\rm exp}$ [ns]")
        ax.set_ylabel(r"$\rm t_{\rm rec}$ [ns]")
        ax.set_title(fr"\rm {label}  $\chi^2 / \rm ndf$ = {chi2r:.2f}")

    sigma_t_ns = timing["sigma_t_ns"]
    for ax, resid_ns, label in (
        (axs[1, 0], timing["resid_pwf_ns"], "PWF"),
        (axs[1, 1], timing["resid_swf_ns"], "SWF"),
    ):
        ax.scatter(timing["t_exp_ns"], resid_ns)
        ax.axhline(0, linestyle="--", color="gray")
        ax.axhline(sigma_t_ns, linestyle=":", color="gray", label=f"${sigma_t_ns:.0f}$ ns")
        ax.axhline(-sigma_t_ns, linestyle=":", color="gray")
        ax.set_xlabel(r"$\rm t_{\rm exp}$ [ns]")
        ax.set_ylabel(r"$\rm t_{\rm exp} - t_{\rm rec}$ [ns]")
        ax.grid(True)
        ax.set_title(f"{label} residuals")
        ax.legend(fontsize=8)

    fig.suptitle(f"Event {t_recons.event_number} - Run {t_recons.run_number}")
    fig.tight_layout()
    fig.savefig(plot_dir/ f"timing_residuals_event_{t_recons.event_number}_run_{t_recons.run_number}.png")
    plt.close(fig)

