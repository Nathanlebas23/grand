import sys

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

#-------------------------------
# Load GRANDlib and nl_style
#-------------------------------
from grand.analysis.pipeline_nathan.scripts.loading import load_config

config_path = Path(__file__).parent.parent / "config.yaml"
config = load_config(config_path)


logger = logging.getLogger("grand.process")

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
    """Plot ADC peak amplitude vs angle omega, the ADF model, and their residuals.

    peak_amps    : amplitudes mesurees dans les traces ADC (simulees en mode simulation),
                   ce n'est PAS une amplitude de verite MC.
    adf_amplitude : modele ADF evalue aux memes DUs.
    """

    fig, (ax, ax_res) = plt.subplots(
        2, 1, sharex=True, figsize=(7.4, 6.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    omega_deg = np.rad2deg(t_recons.omega)
    peak_amps = np.asarray(t_recons.peak_amps)
    adf_amplitude = np.asarray(t_recons.adf_amplitude)

    ax.plot(
        omega_deg,
        peak_amps,
        "ob",
        label="Simulated ADC data",
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

    ax.set_ylabel("Voltage [ADC]")
    ax.legend(frameon=False, loc="best")

    # ----- panneau de residus : donnees ADC simulees - modele ADF -----
    residuals = peak_amps - adf_amplitude

    ax_res.plot(omega_deg, residuals, "ob")
    ax_res.axhline(0, linestyle="--", color=NL_COLORS["black"])
    ax_res.set_xlabel(r"$\omega$ [deg]")
    ax_res.set_ylabel("Residuals [ADC]")

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
    fill_ratio,
    fsuptit,
    ftit_adf,
    plot_dir,
    is_simulation=False,
    dead_du_ids=None,
):
    """Plot the 2D footprint of the event on the ground.

    dead_du_ids : DU de la table de reference n'apparaissant dans aucun evenement du
    fichier ROOT (donnees reelles uniquement). None -> comportement d'origine, une
    seule couche "DUs on site".
    """

    if is_simulation:
        distm = 20  # distance in k_meters
        xmin, xmax = -distm, distm
        ymin, ymax = -distm, distm
    else:    
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

    if  not is_simulation:
        dead_mask = (
            antenna_position["DU_id"].isin(dead_du_ids)
            if dead_du_ids
            else np.zeros(len(antenna_position), dtype=bool)
        )

        alive = antenna_position[~dead_mask]
        ax.scatter(
            -alive["y"] / 1000,
            alive["x"] / 1000,
            marker="+",
            color=NL_COLORS["black"],
            label=f"DUs on site ({len(alive)})",
        )

        if np.any(dead_mask):
            dead = antenna_position[dead_mask]
            ax.scatter(
                -dead["y"] / 1000,
                dead["x"] / 1000,
                marker="x",
                color=NL_COLORS["red"],
                alpha=0.7,
                label=f"Dead DUs ({len(dead)})",
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

    ax.text(
    0.02,
    0.98,
    f"Fill ratio = {fill_ratio:.2f}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    )   

    ax.legend(frameon=False, loc="best", fontsize=9)

    fig.savefig(
        plot_dir
        / f"footprint_event_{t_recons.event_number}_run_{t_recons.run_number}.png",
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_event(t_recons, results: dict, antenna_position, plot_dir, is_simulation, dead_du_ids=None) -> None:
    """Produce the ADC-vs-omega and footprint plots for one already-analyzed event.

    dead_du_ids : liste des DU de la table de reference absentes de tout le fichier ROOT
    (donnees reelles uniquement, cf. loading.get_dead_du_ids). None -> couche non tracee.
    """
    plot_ADC_vs_omega(
        t_recons, results["omega_cr_mean"], results["w"], results["adf_f"],
        results["fsuptit"], results["ftit_adf"], plot_dir,
    )
    plot_footprint(
        t_recons, antenna_position, results["Xants"], results["K"], results["Xcore"],
        results["x_ell"], results["omega_cr_mean"], results['omega_ell_min'], results['omega_ell_max'], results['x_ell_min'], results['x_ell_max'], results['fill_ratio'], results["fsuptit"], results["ftit_adf"], plot_dir, is_simulation,
        dead_du_ids=dead_du_ids,
    )

    if results.get("simulation_comparison") is not None:
        plot_simulation_comparison(
            t_recons,
            results["simulation_comparison"],
            plot_dir,
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


def plot_simulation_comparison(
    t_recons,
    comparison,
    plot_dir,
):
    """Erreurs de reconstruction vis-a-vis de la verite MC.

    Deux panneaux seulement, limites a ce qui est physiquement valide :
    erreur angulaire (PWF/SWF/ADF) et energie. Pas de coeur (reperes MC et reco pas
    encore reconcilies), pas de Xmax (xmax_pos_shc est dans le repere gerbe),
    pas de vecteur direction (branche vide dans cette production).
    """

    if comparison is None:
        return

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    # ----- panneau 1 : erreur angulaire par methode -----
    labels = ["PWF", "SWF", "ADF"]
    angular_errors = [
        comparison["angular_error_pwf_deg"],
        comparison["angular_error_swf_deg"],
        comparison["angular_error_adf_deg"],
    ]

    axs[0].bar(labels, angular_errors, color=NL_COLORS["blue"])
    axs[0].set_ylabel(r"Angular error $\Delta\Psi$ [deg]")
    axs[0].set_title("Direction reconstruction")
    axs[0].grid(axis="y", alpha=0.3)

    # ----- panneau 2 : energie -----
    ratio = comparison["energy_ratio"]
    rel_percent = 100.0 * comparison["energy_relative_error"]

    axs[1].bar([r"$(E_{\rm reco}-E_{\rm sim})/E_{\rm sim}$"],
               [rel_percent], color=NL_COLORS["orange"])
    axs[1].axhline(0, linestyle="--", color=NL_COLORS["black"])
    axs[1].set_ylabel("Relative energy error [%]")
    axs[1].set_title(rf"$E_{{\rm reco}}/E_{{\rm sim}} = {ratio:.2f}$")
    axs[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Simulation vs reconstruction - "
        f"Event {t_recons.event_number} - Run {t_recons.run_number}"
    )
    fig.tight_layout()
    fig.savefig(
        plot_dir
        / (
            f"simulation_comparison_event_"
            f"{t_recons.event_number}_"
            f"run_{t_recons.run_number}.png"
        )
    )
    plt.close(fig)
